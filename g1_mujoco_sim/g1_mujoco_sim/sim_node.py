"""MuJoCo simulation of the G1 speaking the real robot's low-level interface (docs/interfaces.md 3.1).

Physics at sim_dt (0.002 s, 500 Hz), paced to wall-clock real time. Every step:
  * applies the latest /lowcmd like the G1 motor driver: tau = tau_ff + kp (q_des - q) + kd (dq_des - dq),
    clipped to the joint torque limit (motors with mode 0 produce no torque)
  * publishes /lowstate (IMU = imu_in_pelvis, motors 0..28, tick = sim ms) and /clock
Also: /joint_states, /odom + TF odom->pelvis (ground truth), hands (/g1/hand/*/cmd -> Revo2 position actuators,
/g1/hand/*/state), MID-360 /livox/lidar (PointCloud2, livox_ros_driver2 point format, frame mid360_link) and
/scan (LaserScan in laser_link = mid360_link rolled back by pi so z is up; static TF mid360_link->laser_link),
/sim/status (JSON: rtf, step time, ...), services /sim/reset (Trigger) and /sim/harness (SetBool).

Autonomy additions (docs/interfaces.md 6):
  * head RGB-D camera (MJCF camera head_d435 at d435_link): /g1/head_camera/{image_raw (rgb8), depth (16UC1, mm,
    aligned to colour), camera_info}, frame d435_color_optical_frame (static TF from d435_link). Rendered in a
    separate process (head_camera.py) from a qpos snapshot, only while someone subscribes (camera_rate, 15 Hz).
  * /sim/objects (String JSON, GROUND TRUTH = step-1 cheat): pose + type of every free object, the surgeon
    stand-in's hand and the simulated surgeon grip (g1_mujoco_sim.scene_objects). /sim/reset_objects (Trigger)
    puts them back, shifted by parameter object_offset [dx, dy, dyaw] (rack + instruments, for trials).
  * robot stubs (stubs.py): /g1/estop/heartbeat (simulated hardware e-stop, 10 Hz; /sim/estop presses it,
    /sim/estop_link kills the receiver), /g1/indicator/light and /g1/speech/say (logged).
  * fixed_base:=true welds the pelvis at the spawn pose, fixed_base_height above the floor (arm-skill development
    while the placeholder policy cannot take large reaches); no harness then. /sim/fixed_base_goal
    (geometry_msgs/PoseStamped, world frame; CHEAT) glides the welded base to another stance in a straight line
    (stands in for walking in fixed-base mode; the mission uses it instead of Nav2 there).

Harness ("elastic band", like unitree_mujoco): holds the pelvis at its spawn pose while the robot starts in
DAMP. Released automatically `harness_release_delay` s after /g1/mode becomes STAND (fading out over
`harness_fade` s), or via /sim/harness false. /sim/reset re-spawns and re-attaches it.
"""
import json
import math
import os
import threading
import time

import numpy as np
import mujoco
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState, LaserScan, PointCloud2, PointField
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from unitree_hg.msg import LowCmd, LowState

from g1_interfaces.msg import Mode
from g1_mujoco_sim.lidar import Mid360, Scan2D
from g1_mujoco_sim.model_map import BODY_JOINTS, HAND_SLOTS, ModelMap
from g1_mujoco_sim.scene_objects import BaseGlide, SceneObjects, weld_to_world
from g1_mujoco_sim.head_camera import HeadCamera, _u8


def _stamp(t):
    # Round once in integer nanoseconds. The old form, int(t) plus round(frac) % 1e9, turned t = 6.9999999999 (the
    # float sum of 500 steps of 0.002 s) into 6.0 s: /clock went back 1 s every few sim seconds, which made every
    # tf2 buffer "detect a jump back in time" and clear itself (AMCL / Nav2), and robot_state_publisher warn.
    s, ns = divmod(int(round(t * 1e9)), 1000000000)
    return Time(sec=s, nanosec=ns)


def _repo_root():
    r = os.environ.get("HG1_ROOT")
    if r and os.path.isdir(r):
        return r
    here = os.path.realpath(__file__)
    for _ in range(8):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "g1-mujoco-sim")):
            return here
    return os.getcwd()


def quat_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


LIVOX_FIELDS = [PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name="tag", offset=16, datatype=PointField.UINT8, count=1),
                PointField(name="line", offset=17, datatype=PointField.UINT8, count=1),
                PointField(name="timestamp", offset=18, datatype=PointField.FLOAT64, count=1)]
LIVOX_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("tag", "u1"),
                        ("line", "u1"), ("timestamp", "<f8")])  # itemsize 26 = livox_ros_driver2


class SimNode(Node):
    def __init__(self):
        super().__init__("g1_mujoco_sim")
        root = _repo_root()
        p = self.declare_parameter
        p("world", "hospital")
        p("model", "")                      # scene MJCF; default g1-mujoco-sim/<world>_g1.xml
        p("spawn", "")                      # spawn name from worlds/spawns.json, or "x,y,yaw"; "" = world default
        p("viewer", False)
        p("harness", True)
        p("harness_release_delay", 2.5)
        p("harness_fade", 1.0)
        p("realtime", True)                 # pace to wall clock (false = as fast as possible)
        p("lidar", True)
        p("lidar_rate", 10.0)
        p("lidar_points", 10000)
        p("scan_beams", 720)
        p("lidar_range", 40.0)
        p("joint_state_rate", 100.0)
        p("odom_rate", 100.0)
        p("hand_state_rate", 50.0)
        p("status_rate", 1.0)
        p("odom_frame", "odom")
        p("base_frame", "pelvis")
        p("lidar_frame", "mid360_link")     # /livox/lidar frame (real mount: rolled by pi, z down)
        p("scan_frame", "laser_link")       # /scan frame = mid360_link rolled back by pi (z up); static TF published
        p("record_path", "")                # if set: save (t, qpos) at record_rate to this .npz on shutdown
        p("record_rate", 30.0)
        p("effort_limits", "model")         # model: MJCF limits | ultra: robot_spec.json effort_limit_ultra
        p("min_armature", 0.0)              # >0: raise body-joint armature below this (for models without any)
        p("fixed_base", False)              # weld the pelvis at the spawn pose (arm-skill development)
        p("fixed_base_height", 0.78)        # pelvis height for fixed_base (feet just on the floor)
        p("camera", True)                   # head RGB-D (rendered only while subscribed)
        p("camera_rate", 15.0)
        p("camera_width", 640)
        p("camera_height", 360)             # 16:9 like the Isaac head camera (same K: fovy 42 deg -> fx=fy=468.9)
        p("camera_frame", "d435_color_optical_frame")   # optical frame, static TF from d435_link
        p("camera_name", "head_d435")
        p("objects_rate", 10.0)             # /sim/objects (ground truth) Hz
        p("object_offset", [0.0, 0.0, 0.0]) # dx, dy, dyaw applied to the rack + instruments by /sim/reset_objects
        p("surgeon_grip", True)             # simulated surgeon grip (proximity + dwell -> weld)
        p("estop_stub", True)               # publish /g1/estop/heartbeat (simulated hardware e-stop, stubs.py)
        g = self.get_parameter
        world = g("world").value
        model_path = g("model").value or os.path.join(root, "g1-mujoco-sim", f"{world}_g1.xml")
        self.get_logger().info(f"loading {model_path}")
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        self.M = ModelMap(self.m)
        low = self.m.dof_armature[self.M.dadr] < g("min_armature").value
        if low.any():
            self.m.dof_armature[self.M.dadr[low]] = g("min_armature").value
            self.get_logger().warn(f"{int(low.sum())} body joints had armature < min_armature; raised")
        if g("effort_limits").value == "ultra":
            spec_path = os.path.join(root, "g1-brainco-description", "g1_29dof_brainco", "robot_spec.json")
            ultra = np.array(json.load(open(spec_path))["effort_limit_ultra"], float)
            self.M.tau_limit = ultra
            self.get_logger().warn(f"effort limits overridden with robot_spec effort_limit_ultra: {ultra.tolist()}")
            for i, a in enumerate(self.M.act):
                self.m.actuator_ctrlrange[a] = [-ultra[i], ultra[i]]
                self.m.jnt_actfrcrange[self.M.jid[i]] = [-ultra[i], ultra[i]]
        self.dt = float(self.m.opt.timestep)
        self.nb = len(BODY_JOINTS)

        spawns_file = os.path.join(os.path.dirname(model_path), "worlds", "spawns.json")
        spawns = json.load(open(spawns_file)).get(world, {}) if os.path.exists(spawns_file) else {}
        sp = g("spawn").value or spawns.get("default", "")
        if "," in sp:
            x, y, yaw = (float(v) for v in sp.split(","))
            self.spawn = {"x": x, "y": y, "yaw": yaw}
        else:
            self.spawn = spawns.get(sp, {"x": 0.0, "y": 0.0, "yaw": 0.0})
        self.get_logger().info(f"spawn '{sp}': {self.spawn}")
        # default standing pose used for the spawn (policy-neutral: legs slightly bent, arms down)
        self.spawn_q = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2 + [0.0] * 17)
        self.spawn_z = 0.79

        # low-level command state (written by the ROS thread, read by physics)
        self.lock = threading.Lock()
        z = np.zeros(self.nb)
        self.cmd = {"q": z.copy(), "dq": z.copy(), "tau": z.copy(), "kp": z.copy(), "kd": z.copy(),
                    "on": np.zeros(self.nb, bool)}
        self.lowcmd_count = 0
        self.hand_target = {s: np.zeros(6) for s in ("left", "right")}
        self.mode = None
        self.stand_since = None
        self.harness_on = bool(g("harness").value)
        self.harness_gain = 1.0 if self.harness_on else 0.0
        self.reset_requested = False

        # ROS I/O
        be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        rel = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_low = self.create_publisher(LowState, "/lowstate", rel)
        self.pub_clock = self.create_publisher(Clock, "/clock", rel)
        self.pub_js = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_odom = self.create_publisher(Odometry, "/odom", 10)
        self.pub_gt = self.create_publisher(Odometry, "/sim/ground_truth", 10)
        self.pub_cloud = self.create_publisher(PointCloud2, "/livox/lidar", 5)
        self.pub_scan = self.create_publisher(LaserScan, "/scan", 5)
        self.pub_status = self.create_publisher(String, "/sim/status", 10)
        self.pub_hand = {s: self.create_publisher(JointState, f"/g1/hand/{s}/state", 10) for s in ("left", "right")}
        self.tf = TransformBroadcaster(self)
        self.static_tf = StaticTransformBroadcaster(self)
        st = TransformStamped()
        st.header.frame_id = g("lidar_frame").value
        st.child_frame_id = g("scan_frame").value
        st.transform.rotation.x = 1.0  # rotation by pi about x
        st.transform.rotation.w = 0.0
        self.static_tf.sendTransform(st)
        self.R_flip = np.diag([1.0, -1.0, -1.0])
        self.create_subscription(LowCmd, "/lowcmd", self.on_lowcmd, be)
        self.create_subscription(Mode, "/g1/mode", self.on_mode, latched)
        for s in ("left", "right"):
            self.create_subscription(JointState, f"/g1/hand/{s}/cmd", lambda msg, s=s: self.on_hand(s, msg), 10)
        self.create_service(Trigger, "/sim/reset", self.on_reset)
        self.create_service(SetBool, "/sim/harness", self.on_harness)
        self.create_service(Trigger, "/sim/reset_objects", self.on_reset_objects)
        # robot-side stubs: hardware e-stop heartbeat (+ /sim/estop), indicator light and speech (log only)
        from g1_mujoco_sim.stubs import RobotStubs
        self.stubs = RobotStubs(self, enabled=bool(self.get_parameter("estop_stub").value))
        self.reset_objects_requested = False
        # ground-truth objects + simulated surgeon grip
        setup = os.path.join(os.path.dirname(model_path), "worlds", "theatre_setup.json")
        self.objects = SceneObjects(self.m, setup if os.path.exists(setup) else None)
        self.pub_objects = self.create_publisher(String, "/sim/objects", 10)
        self.glide = BaseGlide()
        self.glide_goal = None
        self.glide_remaining = None
        self.create_subscription(PoseStamped, "/sim/fixed_base_goal", self.on_glide_goal, 5)
        self.fixed_base = bool(g("fixed_base").value)
        self.fixed_eq = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_EQUALITY, "fixed_base")
        if self.fixed_base and self.fixed_eq < 0:
            self.get_logger().error("fixed_base requested but the model has no 'fixed_base' weld: rebuild g1-mujoco-sim")
            self.fixed_base = False
        if self.fixed_base:
            self.get_logger().warn("FIXED BASE: pelvis welded to the world (arm-skill development mode)")
        # head camera
        self.camera = None
        if g("camera").value:
            try:
                self.camera = HeadCamera(self, self.m, g("camera_name").value, g("camera_width").value,
                                         g("camera_height").value, g("camera_rate").value, model_path=model_path,
                                         frame_id=g("camera_frame").value)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"head camera disabled: {e}")

        self.lowstate = LowState()
        self.lidar = Mid360(self.m, g("lidar_points").value, g("lidar_rate").value, self.dt,
                            range_max=g("lidar_range").value) if g("lidar").value else None
        self.scan = Scan2D(self.m, g("scan_beams").value, range_max=min(30.0, g("lidar_range").value)) \
            if g("lidar").value else None
        if self.M.lidar_site < 0:
            self.get_logger().warn("model has no mid360_link site: lidar disabled")
            self.lidar = self.scan = None
        if not self.M.has_hands:
            self.get_logger().warn("model has no BrainCo hand actuators: hand state echoes the command")
        self.hand_names = {s: (self.M.hand_joint_names[s] if self.M.has_hands else
                               [f"{s}_hand_{k}" for k in HAND_SLOTS]) for s in ("left", "right")}
        self.js_names = list(BODY_JOINTS) + (self.hand_names["left"] + self.hand_names["right"]
                                             if self.M.has_hands else [])
        self.respawn()

    # ------------------------------------------------------------------ callbacks (ROS thread)
    def on_lowcmd(self, msg):
        mc = msg.motor_cmd
        n = self.nb
        q = np.fromiter((mc[i].q for i in range(n)), float, n)
        dq = np.fromiter((mc[i].dq for i in range(n)), float, n)
        tau = np.fromiter((mc[i].tau for i in range(n)), float, n)
        kp = np.fromiter((mc[i].kp for i in range(n)), float, n)
        kd = np.fromiter((mc[i].kd for i in range(n)), float, n)
        on = np.fromiter((mc[i].mode != 0 for i in range(n)), bool, n)
        with self.lock:
            self.cmd = {"q": q, "dq": dq, "tau": tau, "kp": kp, "kd": kd, "on": on}
            self.lowcmd_count += 1

    def on_mode(self, msg):
        if msg.mode == Mode.STAND and self.mode != Mode.STAND:
            self.stand_since = time.monotonic()
        if msg.mode == Mode.DAMP:
            self.stand_since = None
        self.mode = msg.mode

    def on_hand(self, side, msg):
        if len(msg.position) >= 6:
            self.hand_target[side] = np.clip(np.array(msg.position[:6], float), 0.0, 1.0)

    def on_reset(self, req, res):
        self.reset_requested = True
        res.success = True
        res.message = "respawning with harness"
        return res

    def on_glide_goal(self, msg):
        if not self.fixed_base:
            self.get_logger().warn("/sim/fixed_base_goal ignored: fixed_base is false")
            return
        o = msg.pose.orientation
        yaw = math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y * o.y + o.z * o.z))
        self.glide_goal = (msg.pose.position.x, msg.pose.position.y, yaw)

    def on_reset_objects(self, req, res):
        self.reset_objects_requested = True
        res.success = True
        off = list(self.get_parameter("object_offset").value)
        res.message = f"objects reset, rack offset {off}"
        return res

    def on_harness(self, req, res):
        self.harness_on = bool(req.data)
        if self.harness_on:
            self.harness_gain = 1.0
            self.harness_anchor = self.d.qpos[self.M.free_q:self.M.free_q + 3].copy()
            self.harness_anchor[2] = self.spawn_z
        res.success = True
        res.message = f"harness {'on' if self.harness_on else 'releasing'}"
        return res

    # ------------------------------------------------------------------ physics
    def respawn(self):
        mujoco.mj_resetData(self.m, self.d)
        fq = self.M.free_q
        yaw = self.spawn["yaw"]
        self.d.qpos[fq:fq + 3] = [self.spawn["x"], self.spawn["y"], self.spawn_z]
        self.d.qpos[fq + 3:fq + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.M.set_body_q(self.d, self.spawn_q)
        if getattr(self, "fixed_base", False):
            self.d.qpos[fq + 2] = self.get_parameter("fixed_base_height").value
        mujoco.mj_forward(self.m, self.d)
        if getattr(self, "fixed_base", False):
            weld_to_world(self.m, self.d, "fixed_base", "pelvis")
        if getattr(self, "objects", None) is not None:
            self.objects.reset(self.d, tuple((list(self.get_parameter("object_offset").value) + [0.0] * 3)[:3]))
        self.harness_anchor = np.array([self.spawn["x"], self.spawn["y"], self.spawn_z])
        self.harness_quat = self.d.qpos[fq + 3:fq + 7].copy()
        self.harness_on = bool(self.get_parameter("harness").value) and not getattr(self, "fixed_base", False)
        self.harness_gain = 1.0 if self.harness_on else 0.0
        self.stand_since = None if self.mode != Mode.STAND else time.monotonic()
        # odom frame = spawn pose (like a real odometry source starting at identity)
        self.odom_origin = self.d.qpos[fq:fq + 3].copy()
        self.odom_origin[2] = 0.0
        self.odom_yaw0 = yaw
        with self.lock:
            self.cmd["on"][:] = False

    def apply_harness(self):
        if self.harness_on and self.stand_since is not None:
            if time.monotonic() - self.stand_since > self.get_parameter("harness_release_delay").value:
                self.harness_on = False
        if not self.harness_on and self.harness_gain > 0:
            self.harness_gain = max(0.0, self.harness_gain - self.dt / max(1e-3, self.get_parameter("harness_fade").value))
        b = self.M.pelvis
        self.d.xfrc_applied[b] = 0
        if self.harness_gain <= 0:
            return
        g = self.harness_gain
        fq, fd = self.M.free_q, self.M.free_d
        pos = self.d.qpos[fq:fq + 3]
        vel = self.d.qvel[fd:fd + 3]
        kxy, kz, c = 1500.0, 4000.0, 300.0
        f = np.array([kxy, kxy, kz]) * (self.harness_anchor - pos) - c * vel
        # orientation hold: torque = -k * rotation error - c * angular velocity (world frame)
        q = self.d.qpos[fq + 3:fq + 7]
        qe = quat_mul(self.harness_quat, np.array([q[0], -q[1], -q[2], -q[3]]))
        if qe[0] < 0:
            qe = -qe
        err = 2 * qe[1:4]
        R = self.d.xmat[b].reshape(3, 3)
        wv = R @ self.d.qvel[fd + 3:fd + 6]
        tq = 300.0 * err - 30.0 * wv
        self.d.xfrc_applied[b, :3] = g * f
        self.d.xfrc_applied[b, 3:] = g * tq

    def apply_motors(self):
        M, d = self.M, self.d
        with self.lock:
            c = self.cmd
        q = d.qpos[M.qadr]
        dq = d.qvel[M.dadr]
        tau = c["tau"] + c["kp"] * (c["q"] - q) + c["kd"] * (c["dq"] - dq)
        tau = np.where(c["on"], np.clip(tau, -M.tau_limit, M.tau_limit), 0.0)
        d.ctrl[M.act] = tau
        self.tau_applied = tau
        if M.has_hands:
            for s in ("left", "right"):
                d.ctrl[M.hand_act[s]] = self.hand_target[s] * M.hand_upper[s]

    # ------------------------------------------------------------------ outputs (physics thread)
    def publish_lowstate(self, t):
        M, d, ls = self.M, self.d, self.lowstate
        fq, fd = M.free_q, M.free_d
        quat = d.qpos[fq + 3:fq + 7]
        gyro = d.qvel[fd + 3:fd + 6]  # free-joint angular velocity is in the body (pelvis) frame
        if M.acc_adr is not None:
            acc = d.sensordata[M.acc_adr:M.acc_adr + 3]
        else:
            R = d.xmat[M.pelvis].reshape(3, 3)
            acc = R.T @ np.array([0.0, 0.0, 9.81])
        imu = ls.imu_state
        imu.quaternion = [float(v) for v in quat]
        imu.gyroscope = [float(v) for v in gyro]
        imu.accelerometer = [float(v) for v in acc]
        imu.rpy = [float(v) for v in quat_to_rpy(quat)]
        q = d.qpos[M.qadr]
        dq = d.qvel[M.dadr]
        tau = self.tau_applied
        ms = ls.motor_state
        for i in range(self.nb):
            s = ms[i]
            s.q = float(q[i])
            s.dq = float(dq[i])
            s.tau_est = float(tau[i])
        ls.tick = int(round(t * 1000)) & 0xFFFFFFFF
        self.pub_low.publish(ls)

    def publish_joint_states(self, stamp):
        M, d = self.M, self.d
        js = JointState()
        js.header.stamp = stamp
        js.name = self.js_names
        pos = list(d.qpos[M.qadr])
        vel = list(d.qvel[M.dadr])
        eff = list(self.tau_applied)
        if M.has_hands:
            for s in ("left", "right"):
                qa = M.hand_qadr[s]
                pos += list(d.qpos[qa])
                vel += [0.0] * 6
                eff += [0.0] * 6
        js.position = [float(v) for v in pos]
        js.velocity = [float(v) for v in vel]
        js.effort = [float(v) for v in eff]
        self.pub_js.publish(js)

    def publish_hands(self, stamp):
        for s in ("left", "right"):
            js = JointState()
            js.header.stamp = stamp
            js.name = [f"{s}_hand_{k}" for k in HAND_SLOTS]
            if self.M.has_hands:
                u = self.d.qpos[self.M.hand_qadr[s]] / self.M.hand_upper[s]
            else:
                u = self.hand_target[s]
            js.position = [float(v) for v in np.clip(u, 0.0, 1.0)]
            self.pub_hand[s].publish(js)

    def publish_odom(self, stamp):
        M, d = self.M, self.d
        fq, fd = M.free_q, M.free_d
        pos = d.qpos[fq:fq + 3]
        quat = d.qpos[fq + 3:fq + 7]
        c, s = math.cos(-self.odom_yaw0), math.sin(-self.odom_yaw0)
        rel = pos - self.odom_origin
        op = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])
        oq = quat_mul(np.array([math.cos(-self.odom_yaw0 / 2), 0, 0, math.sin(-self.odom_yaw0 / 2)]), quat)
        R = d.xmat[M.pelvis].reshape(3, 3)
        v_body = R.T @ d.qvel[fd:fd + 3]
        w_body = d.qvel[fd + 3:fd + 6]
        for pub, frame, p_, q_ in ((self.pub_odom, self.get_parameter("odom_frame").value, op, oq),
                                   (self.pub_gt, "world", pos, quat)):
            o = Odometry()
            o.header.stamp = stamp
            o.header.frame_id = frame
            o.child_frame_id = self.get_parameter("base_frame").value
            o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = (float(v) for v in p_)
            o.pose.pose.orientation.w, o.pose.pose.orientation.x, o.pose.pose.orientation.y, \
                o.pose.pose.orientation.z = (float(v) for v in q_)
            o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = (float(v) for v in v_body)
            o.twist.twist.angular.x, o.twist.twist.angular.y, o.twist.twist.angular.z = (float(v) for v in w_body)
            pub.publish(o)
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.get_parameter("odom_frame").value
        t.child_frame_id = self.get_parameter("base_frame").value
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in op)
        t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = \
            (float(v) for v in oq)
        self.tf.sendTransform(t)

    def lidar_pose(self):
        sid = self.M.lidar_site
        return self.d.site_xpos[sid].copy(), self.d.site_xmat[sid].reshape(3, 3).copy()

    def publish_cloud(self, frame):
        n = len(frame["xyz"])
        arr = np.zeros(n, LIVOX_DTYPE)
        arr["x"], arr["y"], arr["z"] = frame["xyz"][:, 0], frame["xyz"][:, 1], frame["xyz"][:, 2]
        arr["intensity"] = 100.0
        arr["tag"] = 0x10
        arr["line"] = frame["line"]
        arr["timestamp"] = frame["t"] * 1e9
        msg = PointCloud2()
        msg.header.stamp = _stamp(frame["t0"])
        msg.header.frame_id = self.get_parameter("lidar_frame").value
        msg.height, msg.width = 1, n
        msg.fields = LIVOX_FIELDS
        msg.is_bigendian = False
        msg.point_step = LIVOX_DTYPE.itemsize
        msg.row_step = msg.point_step * n
        msg.is_dense = True
        msg.data = _u8(arr.tobytes())   # array('B'): skips rosidl's per-byte validation (~10 ms/frame)
        self.pub_cloud.publish(msg)

    def publish_scan(self, stamp, ranges):
        s = LaserScan()
        s.header.stamp = stamp
        s.header.frame_id = self.get_parameter("scan_frame").value
        s.angle_min = self.scan.angle_min
        s.angle_increment = self.scan.inc
        s.angle_max = self.scan.angle_min + self.scan.inc * (self.scan.beams - 1)
        s.time_increment = 0.0
        s.scan_time = 1.0 / self.get_parameter("lidar_rate").value
        s.range_min = self.scan.rmin
        s.range_max = self.scan.rmax
        s.ranges = ranges.tolist()
        self.pub_scan.publish(s)

    # ------------------------------------------------------------------ main loop
    def run(self, stop):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        viewer = None
        if g("viewer"):
            from mujoco import viewer as mjviewer
            viewer = mjviewer.launch_passive(self.m, self.d, show_left_ui=False, show_right_ui=False)
        realtime = g("realtime")
        dec_js = max(1, int(round(1.0 / g("joint_state_rate") / self.dt)))
        dec_odom = max(1, int(round(1.0 / g("odom_rate") / self.dt)))
        dec_hand = max(1, int(round(1.0 / g("hand_state_rate") / self.dt)))
        dec_status = max(1, int(round(1.0 / g("status_rate") / self.dt)))
        dec_view = max(1, int(round(1.0 / 60.0 / self.dt)))
        dec_obj = max(1, int(round(1.0 / g("objects_rate") / self.dt)))
        dec_grip = 10
        grip = bool(g("surgeon_grip")) and self.objects.has_surgeon
        if self.camera is not None:
            self.camera.start()
        self.tau_applied = np.zeros(self.nb)
        self.rec_t, self.rec_q = [], []
        dec_rec = max(1, int(round(1.0 / g("record_rate") / self.dt)))
        recording = bool(g("record_path"))
        step = 0
        wall0 = time.perf_counter()
        sim0 = self.d.time
        win_wall, win_sim, win_cmd = wall0, self.d.time, self.lowcmd_count
        busy = 0.0
        clock = Clock()
        late_steps = 0
        while not stop.is_set() and rclpy.ok():
            if self.reset_requested:
                self.reset_requested = False
                self.respawn()
                wall0, sim0 = time.perf_counter(), self.d.time
            if self.glide_goal is not None:
                gx, gy, gyaw = self.glide_goal
                self.glide_goal = None
                p0 = self.m.eq_data[self.fixed_eq, 0:3].copy()
                q0 = self.m.eq_data[self.fixed_eq, 6:10]
                yaw0 = -math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2))
                self.glide.start(self.d.time, p0, yaw0, (gx, gy), gyaw)
                self.get_logger().info(f"fixed-base glide to ({gx:.2f}, {gy:.2f}, {math.degrees(gyaw):.0f} deg)")
            if self.glide.active:
                self.glide_remaining = self.glide.step(self.m, self.fixed_eq, self.d.time)
            if self.reset_objects_requested:
                self.reset_objects_requested = False
                off = [float(v) for v in g("object_offset")] + [0.0] * 3
                self.objects.reset(self.d, tuple(off[:3]))
                self.get_logger().info(f"objects reset (offset {off[:3]})")
            t_start = time.perf_counter()
            self.apply_harness()
            self.apply_motors()
            mujoco.mj_step(self.m, self.d)
            step += 1
            t = self.d.time
            stamp = _stamp(t)
            clock.clock = stamp
            self.pub_clock.publish(clock)
            self.publish_lowstate(t)
            if self.lidar is not None:
                pos, R = self.lidar_pose()
                fr = self.lidar.step(self.d, pos, R, t)
                if fr is not None:
                    self.publish_cloud(fr)
                    self.publish_scan(stamp, self.scan.cast(self.d, pos, R @ self.R_flip))
            if step % dec_js == 0:
                self.publish_joint_states(stamp)
            if step % dec_odom == 0:
                self.publish_odom(stamp)
            if step % dec_hand == 0:
                self.publish_hands(stamp)
            if grip and step % dec_grip == 0:
                n = self.objects.update_grip(self.d, dec_grip * self.dt)
                if n:
                    self.get_logger().info(f"simulated surgeon grip: {n}")
            if step % dec_obj == 0 and self.objects.bodies:
                self.pub_objects.publish(String(data=self.objects.to_json(self.d)))
            if self.camera is not None:
                self.camera.maybe_capture(self.d, t)
            if recording and step % dec_rec == 0:
                self.rec_t.append(t)
                self.rec_q.append(self.d.qpos.copy())
            if viewer is not None and step % dec_view == 0:
                if not viewer.is_running():
                    stop.set()
                    break
                viewer.sync()
            busy += time.perf_counter() - t_start
            if step % dec_status == 0:
                now = time.perf_counter()
                rtf = (t - win_sim) / max(1e-9, now - win_wall)
                cmd_hz = (self.lowcmd_count - win_cmd) / max(1e-9, now - win_wall)
                st = {"sim_time": round(t, 3), "rtf": round(rtf, 3), "step_ms_mean": round(1e3 * busy / dec_status, 3),
                      "lowcmd_hz": round(cmd_hz, 1), "late_steps": late_steps,
                      "harness": round(self.harness_gain, 2), "world": g("world"),
                      "hands_modelled": self.M.has_hands, "fixed_base": self.fixed_base,
                      "camera_hz": self.camera.rate_measured() if self.camera is not None else None,
                      "camera_render_ms": self.camera.render_ms() if self.camera is not None else None,
                      "surgeon_holding": self.objects.held, "objects": len(self.objects.bodies),
                      "glide": {"active": self.glide.active,
                                "remaining": None if self.glide_remaining is None else round(self.glide_remaining, 3)},
                      "pelvis_xyz": [round(float(v), 3) for v in self.d.qpos[self.M.free_q:self.M.free_q + 3]]}
                self.pub_status.publish(String(data=json.dumps(st)))
                win_wall, win_sim, win_cmd, busy, late_steps = now, t, self.lowcmd_count, 0.0, 0
            if realtime:
                target = wall0 + (t - sim0)
                slack = target - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                elif slack < -0.1:  # fell far behind (e.g. suspended): re-anchor instead of racing
                    wall0, sim0 = time.perf_counter(), t
                    late_steps += 1
                else:
                    late_steps += 1
        if viewer is not None:
            viewer.close()
        if self.camera is not None:
            self.camera.stop()


def main():
    rclpy.init()
    node = SimNode()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    stop = threading.Event()
    def spin():
        try:
            ex.spin()
        except Exception:
            if rclpy.ok():
                raise
    th = threading.Thread(target=spin, daemon=True)
    th.start()
    try:
        node.run(stop)
    except KeyboardInterrupt:
        pass
    except Exception:  # publishers invalidated by SIGINT shutdown mid-step
        if rclpy.ok():
            raise
    finally:
        stop.set()
        path = node.get_parameter("record_path").value
        if path and getattr(node, "rec_q", None):
            np.savez_compressed(path, t=np.array(node.rec_t), qpos=np.array(node.rec_q),
                                model=node.get_parameter("model").value or "")
            print(f"recorded {len(node.rec_q)} frames -> {path}")
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
