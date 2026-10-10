"""Simulated head RGB-D camera (Intel RealSense D435i on the G1 head, MJCF camera `head_d435` at `d435_link`).

Rendering runs in a separate **process** (its own GIL, GL context and ROS node `g1_head_camera`), so the 500 Hz
Python physics loop never waits for it. A render thread in the sim process starved the physics loop through the
GIL (RTF 0.6, measured). Hand-off per frame:
  physics thread: maybe_capture(d, t) every step; when a frame is due (camera_rate) and the render process reports a
                  subscriber and is idle, copy qpos + time into shared memory and set an Event (a few microseconds)
  render process: mj_kinematics on its own MjData, render colour (+ depth if subscribed), publish
    /g1/head_camera/image_raw    sensor_msgs/Image rgb8, 640x360 (16:9 like the Isaac camera: fovy 42 deg, fx=fy=468.9)
    /g1/head_camera/depth        sensor_msgs/Image 16UC1, millimetres (0 = no return), aligned to colour
    /g1/head_camera/camera_info  sensor_msgs/CameraInfo (pinhole from the MJCF fovy; no distortion)
frame_id d435_color_optical_frame (z forward, x right, y down); the sim node publishes the static TF
d435_link -> d435_color_optical_frame. Frames are stamped with sim time.
"""
import array
import math
import multiprocessing as mp
import os
import time

import numpy as np
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

OPTICAL_FRAME = "d435_color_optical_frame"


def _u8(buf):
    """uint8[] field value without rosidl's per-element validation (which costs ~50 ms per 640x480 frame)."""
    a = array.array("B")
    a.frombytes(buf)
    return a


def _render_main(model_path, cam_name, w, h, qpos_arr, t_val, go, busy, wanted, stop, stats, domain_env, static_arr,
                 frame_id=OPTICAL_FRAME):
    os.environ.update(domain_env)
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import CameraInfo, Image
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    rack = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tool_rack")
    rack = rack if (rack >= 0 and m.body_jntnum[rack] == 0) else -1     # static rack: pose comes from the sim
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, w)
    m.vis.global_.offheight = max(m.vis.global_.offheight, h)
    rclpy.init(args=[])            # not the parent's argv (it carries the sim node's __node remap and params)
    node = rclpy.create_node("g1_head_camera")
    log = node.get_logger()
    try:
        rgb_r = mujoco.Renderer(m, h, w)
        dep_r = mujoco.Renderer(m, h, w)
        dep_r.enable_depth_rendering()
    except Exception as e:  # noqa: BLE001
        log.error(f"cannot create an offscreen GL context ({e}); set MUJOCO_GL=egl or osmesa")
        return
    qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
    pub_rgb = node.create_publisher(Image, "/g1/head_camera/image_raw", qos)
    pub_dep = node.create_publisher(Image, "/g1/head_camera/depth", qos)
    pub_info = node.create_publisher(CameraInfo, "/g1/head_camera/camera_info", qos)
    fovy = math.radians(float(m.cam_fovy[cam]))
    f = (h / 2.0) / math.tan(fovy / 2.0)
    K = [f, 0.0, w / 2.0, 0.0, f, h / 2.0, 0.0, 0.0, 1.0]
    log.info(f"head camera {w}x{h} ready (renders only while subscribed)")
    q = np.frombuffer(qpos_arr.get_obj(), dtype=np.float64)
    last_check = 0.0
    parent = os.getppid()
    while not stop.is_set():
        now = time.monotonic()
        if os.getppid() != parent:          # the sim died without stopping us (SIGKILL): do not linger
            break
        if now - last_check > 0.2:
            n_rgb, n_dep = pub_rgb.get_subscription_count(), pub_dep.get_subscription_count()
            wanted.value = 1 if (n_rgb + n_dep) > 0 else 0
            last_check = now
        if not go.wait(timeout=0.2):
            continue
        go.clear()
        try:
            t0 = time.perf_counter()
            with qpos_arr.get_lock():
                d.qpos[:] = q
                t = t_val.value
                if rack >= 0 and static_arr[7] != 0.0:
                    m.body_pos[rack] = static_arr[0:3]
                    m.body_quat[rack] = static_arr[3:7]
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            rgb_r.update_scene(d, camera=cam)
            rgb = rgb_r.render()
            depth = None
            if pub_dep.get_subscription_count() > 0:
                dep_r.update_scene(d, camera=cam)
                depth = dep_r.render()
            im = Image()
            im.header.stamp.sec, im.header.stamp.nanosec = divmod(int(round(t * 1e9)), 1000000000)
            im.header.frame_id = frame_id
            im.height, im.width, im.encoding, im.is_bigendian, im.step = h, w, "rgb8", 0, 3 * w
            im.data = _u8(rgb.tobytes())
            pub_rgb.publish(im)
            if depth is not None:
                mm = np.where((depth > 0.1) & (depth < 10.0), depth * 1000.0, 0.0).astype(np.uint16)
                dm = Image()
                dm.header = im.header
                dm.height, dm.width, dm.encoding, dm.is_bigendian, dm.step = h, w, "16UC1", 0, 2 * w
                dm.data = _u8(mm.tobytes())
                pub_dep.publish(dm)
            ci = CameraInfo()
            ci.header = im.header
            ci.height, ci.width, ci.distortion_model = h, w, "plumb_bob"
            ci.d = [0.0] * 5
            ci.k = K
            ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            ci.p = [K[0], 0.0, K[2], 0.0, 0.0, K[4], K[5], 0.0, 0.0, 0.0, 1.0, 0.0]
            pub_info.publish(ci)
            with stats.get_lock():
                stats[0] += 1
                stats[1] = 0.9 * stats[1] + 0.1 * 1e3 * (time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            log.warn(f"head camera frame failed: {e}")
        finally:
            busy.value = 0
    rgb_r.close()
    dep_r.close()
    node.destroy_node()
    rclpy.try_shutdown()


class HeadCamera:
    def __init__(self, node, m, cam_name="head_d435", width=640, height=360, rate=15.0, model_path=None,
                 frame_id=OPTICAL_FRAME):
        import mujoco
        self.frame_id = frame_id
        self.node = node
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name) < 0:
            raise ValueError(f"model has no camera '{cam_name}'")
        if not model_path:
            raise ValueError("HeadCamera needs the scene path (the render process loads its own model)")
        self.args = (model_path, cam_name, int(width), int(height))
        self.period = 1.0 / float(rate)
        ctx = mp.get_context("spawn")
        self.ctx = ctx
        self.qpos = ctx.Array("d", m.nq)
        self.t = ctx.Value("d", 0.0, lock=False)
        self.go, self.stop_ev = ctx.Event(), ctx.Event()
        self.busy = ctx.Value("i", 0, lock=False)
        self.wanted = ctx.Value("i", 0, lock=False)
        self.stats = ctx.Array("d", 2)         # frames rendered, render ms (EMA)
        self.static = ctx.Array("d", 8, lock=False)   # static tool_rack pos(3) quat(4) valid-flag (object offsets)
        self.rack = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tool_rack")
        self.rack = self.rack if (self.rack >= 0 and m.body_jntnum[self.rack] == 0) else -1
        self.m = m
        self.proc = None
        self.next_t = 0.0
        self.win = (time.monotonic(), 0.0)
        self.hz = 0.0
        st = TransformStamped()
        st.header.frame_id = "d435_link"
        st.child_frame_id = frame_id
        st.transform.rotation.x, st.transform.rotation.y = -0.5, 0.5
        st.transform.rotation.z, st.transform.rotation.w = -0.5, 0.5
        self.static_tf = StaticTransformBroadcaster(node)
        self.static_tf.sendTransform(st)

    def start(self):
        env = {k: v for k, v in os.environ.items() if k.startswith(("ROS_", "RMW_", "CYCLONEDDS", "MUJOCO_GL"))}
        self.proc = self.ctx.Process(target=_render_main, daemon=True, name="g1_head_camera",
                                     args=self.args + (self.qpos, self.t, self.go, self.busy, self.wanted,
                                                       self.stop_ev, self.stats, env, self.static, self.frame_id))
        self.proc.start()

    def stop(self):
        self.stop_ev.set()
        if self.proc is not None:
            self.proc.join(timeout=3.0)
            if self.proc.is_alive():
                self.proc.terminate()

    def maybe_capture(self, d, t):
        """Physics thread, every step. A few microseconds unless a frame is due."""
        if t < self.next_t:
            return
        self.next_t = t + self.period
        if not self.wanted.value or self.busy.value:
            return
        q = np.frombuffer(self.qpos.get_obj(), dtype=np.float64)
        with self.qpos.get_lock():
            q[:] = d.qpos
            self.t.value = float(t)
            if self.rack >= 0:
                self.static[0:3] = self.m.body_pos[self.rack]
                self.static[3:7] = self.m.body_quat[self.rack]
                self.static[7] = 1.0
        self.busy.value = 1
        self.go.set()

    def rate_measured(self):
        now = time.monotonic()
        t0, n0 = self.win
        n = self.stats[0]
        if now - t0 >= 2.0:
            self.hz = round((n - n0) / (now - t0), 1)
            self.win = (now, n)
        return self.hz

    def render_ms(self):
        return round(self.stats[1], 2)
