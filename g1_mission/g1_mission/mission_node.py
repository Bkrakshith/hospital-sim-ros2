"""Mission executive node: runs the scrub_nurse behaviour tree (tree.py) on the robot (docs/interfaces.md 6).

In:  /g1/mission/command (String: start | abort | pause | resume | goto:<location> | reset | clear_queue)
     /g1/mission/request_instrument (g1_interfaces/RequestInstrument service)
     /g1/mode (Mode, transient-local), /joint_states, /g1/hand/*/state, /sim/ground_truth, /sim/objects,
     /g1/perception/detections, /g1/perception/surgeon_hand, [/g1/head_camera/image_raw when recording]
Out: /g1/mission/status (String JSON, 5 Hz + on change), /g1/auto/upper_body_target, /g1/auto/hand/*/cmd,
     /g1/cmd_vel_nav (FineAlign only), /g1/set_mode requests (source "mission"), Nav2 NavigateToPose goals,
     /initialpose (optional, from ground truth = CHEAT) for AMCL

Operator override: if the mode leaves AUTO while a mission runs (p -> STAND, x -> DAMP, fall -> DAMP), the tree is
stopped at once (skills stop, Nav2 goal cancelled) and the mission reports "aborted"; the manager blends the arms back.

Stop (mode HOLD, docs/interfaces.md 3.2): AUTO -> HOLD pauses the mission instead (skills frozen, Nav2 goal cancelled,
state "paused", paused_by_hold). "resume" then asks the manager for HOLD -> AUTO (source "mission") and the tree
continues where it stopped once AUTO is back (skill timelines shifted by the pause). If the robot leaves HOLD any other
way (resume to STAND = "cancel task", power off), the paused mission is aborted.
"""
import json
import math
import threading
import time

import numpy as np
import py_trees
import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from g1_interfaces.msg import Mode
from g1_interfaces.srv import RequestInstrument, SetMode
from g1_mission.config import load_catalog, load_locations, load_mission, resolve
from g1_mission.geometry import make_T, quat_to_mat
from g1_mission.interfaces import (DetectionObjects, GroundTruthObjects, GroundTruthStation, GroundTruthSurgeonHand,
                                   PerceptionSurgeonHand)
from g1_mission.kinematics import UpperBodyIK
from g1_mission.ros_io import AutoNav, GlideNav, NavClient, RosRobot
from g1_mission.skills.base import ArmMotion, SkillContext, drive_arm
from g1_mission.tree import AUTO, HOLD, MODE_NAMES, MissionCore, active_behaviours, build_goto, build_scrub_nurse, tree_status


class _Fut:
    """Adapter: rclpy service future -> done()/result() -> (accepted, message)."""

    def __init__(self, fut):
        self.fut = fut

    def done(self):
        return self.fut.done()

    def result(self):
        try:
            r = self.fut.result()
            return bool(r.accepted), r.message
        except Exception as e:  # noqa: BLE001
            return False, str(e)


class MissionNode(Node):
    def __init__(self):
        super().__init__("g1_mission")
        p = self.declare_parameter
        p("config", "")                       # mission.yaml path ('' = package config)
        p("object_source", "ground_truth")    # ground_truth (/sim/objects, CHEAT) | perception (/g1/perception/detections)
        p("surgeon_source", "ground_truth")   # ground_truth (/sim/objects) | perception (/g1/perception/surgeon_hand)
        p("base_source", "ground_truth")      # ground_truth (/sim/ground_truth, CHEAT) | tf (map -> pelvis)
        p("use_nav", True)                    # create the Nav2 NavigateToPose client
        p("initial_pose", "")                 # '' | ground_truth (CHEAT) | "x,y,yaw": publish /initialpose for AMCL
        p("start_on_request", True)           # an instrument request starts the mission when idle
        p("autostart", False)                 # start the mission as soon as the robot is in STAND
        p("record_dir", "")                   # LeRobot dataset directory ('' = recording off)
        p("record_fps", 15.0)
        p("repeat", True)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cfg = resolve(load_mission(g("config") or None))
        self.cfg["repeat"] = bool(g("repeat"))
        self.catalog = load_catalog()
        self.locations = load_locations()
        self.cb = ReentrantCallbackGroup()
        self.lock = threading.RLock()

        # tf (only needed for base_source=tf / perception frames)
        self.tf_buffer = None
        if g("base_source") == "tf" or g("object_source") == "perception" or g("initial_pose"):
            from tf2_ros import Buffer, TransformListener
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)

        record_dir = g("record_dir")
        self.robot = RosRobot(self, g("base_source"), self.tf_buffer, image=bool(record_dir))
        self.gt_objects = GroundTruthObjects()
        self.create_subscription(String, "/sim/objects", self.on_sim_objects, 5)
        if g("object_source") == "perception":
            self.objects = DetectionObjects(to_map=self._to_map)
            self.create_subscription(String, "/g1/perception/detections", lambda m: self.objects.update(m.data), 5)
        else:
            self.objects = self.gt_objects
        if g("surgeon_source") == "perception":
            self.surgeon = PerceptionSurgeonHand()
            self.create_subscription(PoseStamped, "/g1/perception/surgeon_hand", self.on_surgeon_hand, 5)
        else:
            self.surgeon = GroundTruthSurgeonHand(self.gt_objects)
        ik = UpperBodyIK(side=self.cfg.get("arm", "right"))
        self.ctx = SkillContext(robot=self.robot, objects=self.objects, surgeon=self.surgeon,
                                station=GroundTruthStation(self.robot.base_T), ik=ik, catalog=self.catalog,
                                locations=self.locations, params=self.cfg.get("skills", {}),
                                ready_pose=np.array(self.cfg.get("ready_pose", [0.0] * 17), float),
                                log=lambda s: self.get_logger().info(s))
        self.ctx.arm = ArmMotion(ik, side=self.cfg.get("arm", "right"))
        self.recorder = None
        if record_dir:
            from g1_mission.recorder import LeRobotRecorder
            self.recorder = LeRobotRecorder(record_dir, fps=int(g("record_fps")))
            self.get_logger().info(f"recording skill episodes (LeRobot v2.1) to {record_dir}")
        self.ctx.recorder = self.recorder
        # Nav2 when walking; in fixed-base mode the sim's base glide (CHEAT) stands in for walking between stances
        nav2 = NavClient(self) if g("use_nav") else None
        self.nav = AutoNav(nav2, GlideNav(self, self.robot))
        self.ctx.nav = self.nav

        self.mode = None
        self.mode_src = ""
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Mode, "/g1/mode", self.on_mode, latched)
        self.set_mode_cli = self.create_client(SetMode, "/g1/set_mode", callback_group=self.cb)
        self.core = MissionCore(self.ctx, self.cfg, nav=self.nav, mode_fn=lambda: self.mode,
                                request_mode=self.request_mode, recorder=self.recorder,
                                log=lambda s: self.get_logger().info(s))
        self.root = None
        self.kind = ""                      # scrub_nurse | goto:<loc>
        self.state = "idle"                 # idle | running | paused | succeeded | failed | aborted
        self.message = "idle: send start or request an instrument"
        self.pause_t = None
        self.paused_by_hold = False
        self._pending_checks = []
        self.auto_seen = False
        self.pub_status = self.create_publisher(String, "/g1/mission/status", 10)
        self.create_subscription(String, "/g1/mission/command", self.on_command, 10)
        self.create_service(RequestInstrument, "/g1/mission/request_instrument", self.on_request, callback_group=self.cb)
        self.dt = 1.0 / float(self.cfg.get("rate", 50.0))
        self.create_timer(self.dt, self.tick)
        self.create_timer(0.2, self.publish_status)
        self.init_pose_sent = 0
        self.pub_init = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 1)
        if g("initial_pose"):
            self.create_timer(2.0, self.send_initial_pose)
        self.last_status = ""
        self.get_logger().info(f"mission node up: {len(self.catalog)} instruments, {len(self.locations)} locations, "
                               f"objects={g('object_source')} surgeon={g('surgeon_source')} base={g('base_source')}")

    # ------------------------------------------------------------------ inputs
    def on_sim_objects(self, m):
        self.gt_objects.update(m.data)

    def on_surgeon_hand(self, m):
        T = make_T(quat_to_mat([m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y,
                                m.pose.orientation.z]), [m.pose.position.x, m.pose.position.y, m.pose.position.z])
        if m.header.frame_id not in ("map", "world", ""):
            T = self._to_map(m.header.frame_id, T)
        if T is not None:
            self.surgeon.update(T)

    def _to_map(self, frame, T):
        if self.tf_buffer is None:
            return None
        try:
            from rclpy.time import Time
            t = self.tf_buffer.lookup_transform("map", frame, Time())
            tr, r = t.transform.translation, t.transform.rotation
            return make_T(quat_to_mat([r.w, r.x, r.y, r.z]), [tr.x, tr.y, tr.z]) @ T
        except Exception:  # noqa: BLE001
            return None

    def on_mode(self, m):
        prev = self.mode
        self.mode, self.mode_src = m.mode, m.source
        if m.mode == AUTO:
            self.auto_seen = True
        with self.lock:
            if self.state in ("running", "paused") and self.auto_seen and prev == AUTO and m.mode == HOLD:
                if self.state == "running":
                    self.pause()
                self.paused_by_hold = True
                self.message = f"stopped (source {m.source or '?'}): paused, resume continues the task"
                self.get_logger().warn(f"mission paused by HOLD (source {m.source or '?'})")
            elif self.paused_by_hold and prev == HOLD and m.mode == AUTO:
                self.paused_by_hold = False
                self.resume()
                self.message = f"resumed after stop (source {m.source or '?'})"
                self.get_logger().info("mission resumed after HOLD")
            elif self.paused_by_hold and prev == HOLD and m.mode != HOLD:
                self.paused_by_hold = False
                self.abort(f"cancelled after stop: robot resumed to {MODE_NAMES.get(m.mode, m.mode)} "
                           f"(source {m.source or '?'})", release=False)
            elif self.state in ("running", "paused") and self.auto_seen and prev == AUTO and m.mode != AUTO:
                why = "e-stop" if m.mode == 0 else "operator override"
                self.abort(f"{why}: mode {MODE_NAMES.get(m.mode, m.mode)} (source {m.source or '?'})",
                           release=False)
        if self.get_parameter("autostart").value and m.mode == 1 and self.state == "idle":
            self.start("scrub_nurse")

    def request_mode(self, mode):
        if not self.set_mode_cli.service_is_ready():
            class _No:
                def done(self):
                    return True

                def result(self):
                    return False, "/g1/set_mode not available (is the mode manager running?)"
            return _No()
        req = SetMode.Request()
        req.mode, req.source = int(mode), "mission"
        return _Fut(self.set_mode_cli.call_async(req))

    def on_command(self, m):
        cmd = m.data.strip()
        with self.lock:
            if cmd == "start":
                self.start("scrub_nurse")
            elif cmd == "abort":
                self.abort("aborted by operator")
            elif cmd == "pause":
                self.pause()
            elif cmd == "resume":
                if self.paused_by_hold:
                    self.resume_after_hold()
                else:
                    self.resume()
            elif cmd.startswith("goto:"):
                loc = cmd[5:].strip()
                if loc not in self.locations:
                    self.message = f"unknown location '{loc}' (known: {', '.join(self.locations)})"
                else:
                    self.start(f"goto:{loc}")
            elif cmd == "reset":
                if self.state not in ("running", "paused"):
                    self.state, self.message, self.root = "idle", "idle", None
                    self.core.queue.clear()
            elif cmd == "clear_queue":
                self.core.queue.clear()
            else:
                self.message = f"unknown command '{cmd}'"
        self.publish_status()

    def on_request(self, req, res):
        name = req.instrument.strip()
        with self.lock:
            if name not in self.catalog:
                res.accepted = False
                res.message = f"unknown instrument '{name}'; available: {', '.join(self.catalog)}"
                return res
            self.core.queue.append(name)
            res.accepted = True
            res.message = f"{name} queued (position {len(self.core.queue)}) from {req.requester or '?'}"
            if self.state not in ("running", "paused"):
                if self.get_parameter("start_on_request").value:
                    self.start("scrub_nurse")
                    res.message += "; mission started"
                else:
                    res.message += "; mission not running (send start)"
            elif self.kind != "scrub_nurse":
                res.message += f"; current task {self.kind} runs first"
        self.get_logger().info(f"request: {res.message}")
        return res

    # ------------------------------------------------------------------ mission control
    def start(self, kind):
        if self.state in ("running", "paused"):
            self.message = f"already running {self.kind}"
            return
        self.ctx.arm.reset()
        self.ctx.board.clear()
        self.root = build_scrub_nurse(self.core) if kind == "scrub_nurse" else build_goto(self.core, kind[5:])
        self.root.setup_with_descendants()
        self.kind, self.state, self.message = kind, "running", f"{kind} started"
        self.auto_seen = self.mode == AUTO
        self.get_logger().info(self.message)

    def abort(self, why, release=True):
        if self.root is not None and self.state in ("running", "paused"):
            self.root.stop(py_trees.common.Status.INVALID)
        if self.nav is not None and self.nav.state in ("pending", "active"):
            self.nav.cancel()
        self.robot.command(cmd_vel=(0.0, 0.0, 0.0))
        if release and self.mode == AUTO:
            self.request_mode(1)
        self.state, self.message = "aborted", why
        self.pause_t = None
        self.paused_by_hold = False
        self.get_logger().warn(f"mission aborted: {why}")

    def pause(self):
        if self.state != "running":
            return
        self.state, self.pause_t = "paused", time.monotonic()
        for b in active_behaviours(self.root):
            if hasattr(b, "pause"):
                b.pause()
        self.robot.command(cmd_vel=(0.0, 0.0, 0.0))
        self.message = "paused (arms hold, Nav2 goal cancelled)"

    def resume_after_hold(self):
        """Ask the manager for HOLD -> AUTO; on_mode resumes the tree when AUTO arrives."""
        if self.mode == AUTO:
            self.paused_by_hold = False
            self.resume()
            return
        fut = self.request_mode(AUTO)
        self.message = "resuming after stop: requesting AUTO"

        def check():
            if fut.done():
                ok, msg = fut.result()
                if not ok:
                    self.message = f"cannot resume: {msg}"
                    self.get_logger().warn(self.message)
                return True
            return False
        self._pending_checks.append(check)

    def resume(self):
        if self.state != "paused":
            return
        dt = time.monotonic() - self.pause_t
        for b in active_behaviours(self.root):
            if hasattr(b, "resume"):
                b.resume(dt)
        self.state, self.pause_t, self.message = "running", None, "resumed"

    # ------------------------------------------------------------------ loop
    def tick(self):
        with self.lock:
            self._pending_checks = [c for c in self._pending_checks if not c()]
            self.robot.commanded = False
            if self.state == "running" and self.root is not None:
                if self.robot.upper_q() is None:
                    self.message = "waiting for /joint_states"
                else:
                    self.root.tick_once()
                    st = self.root.status
                    if st == py_trees.common.Status.SUCCESS:
                        self.state, self.message = "succeeded", f"{self.kind} done"
                    elif st == py_trees.common.Status.FAILURE:
                        tip = self.root.tip()
                        msg = self._failure_message()
                        self.state, self.message = "failed", msg or f"{tip.name if tip else self.kind} failed"
                        if self.mode == AUTO:
                            self.request_mode(1)
                        self.get_logger().error(f"mission failed: {self.message}")
            # keep the AUTO channels alive (the controller drops targets older than 0.5 s)
            if self.mode == AUTO and not self.robot.commanded and self.state in ("running", "paused") and \
                    self.robot.upper_q() is not None:
                drive_arm(self.ctx, self.dt) if self.state == "running" else self._hold()
            if self.recorder is not None and self.recorder.recording:
                self.recorder.frame(self.robot, image=self.robot.head_image())

    def _hold(self):
        last = self.robot.last
        if last["upper"] is not None:
            self.robot.command(upper=last["upper"], left_hand=last["left"], right_hand=last["right"])

    def _failure_message(self):
        for b in self.root.iterate():
            if b.status == py_trees.common.Status.FAILURE and b.feedback_message and not b.children:
                return f"{b.name}: {b.feedback_message}"
        return self.core.last_failure

    def status_dict(self):
        tree = tree_status(self.root) if self.root is not None else []
        tip = self.root.tip() if (self.root is not None and self.state in ("running", "paused")) else None
        step = tip.name if tip is not None else ""
        inst = self.ctx.board.get("instrument") if self.state in ("running", "paused") else None
        fa = None
        loc = self.locations.get(self.cfg.get("station", "theatre_instrument_table"))
        if loc is not None:
            e = self.ctx.station.error(loc)
            if e is not None:
                fa = [round(e[0], 3), round(e[1], 3), round(math.degrees(e[2]), 1)]
        return {"state": self.state, "step": step, "paused_by_hold": self.paused_by_hold, "instrument": inst, "tree": tree, "message": self.message,
                "kind": self.kind, "queue": list(self.core.queue), "instruments": list(self.catalog),
                "instrument_info": {k: {"desc": v.get("desc", ""), "standin": v.get("standin", False),
                                        "handle_z": v.get("handle_z", v.get("length", 0.17))}
                                    for k, v in self.catalog.items()},
                "locations": list(self.locations), "mode": MODE_NAMES.get(self.mode, None),
                "station_error": fa, "history": self.core.history[-10:],
                "counts": {"ok": sum(h["success"] for h in self.core.history),
                           "failed": sum(not h["success"] for h in self.core.history)},
                "recording": bool(self.recorder is not None),
                "sources": {"objects": self.get_parameter("object_source").value,
                            "surgeon": self.get_parameter("surgeon_source").value,
                            "base": self.get_parameter("base_source").value}}

    def publish_status(self):
        with self.lock:
            s = json.dumps(self.status_dict(), separators=(",", ":"))
        self.pub_status.publish(String(data=s))

    def send_initial_pose(self):
        """CHEAT when initial_pose=ground_truth: seed AMCL with the true pose until map -> pelvis exists."""
        if self.init_pose_sent >= 10:
            return
        if self.tf_buffer is not None:
            try:
                from rclpy.time import Time
                self.tf_buffer.lookup_transform("map", "pelvis", Time())
                if self.init_pose_sent > 0:
                    self.init_pose_sent = 10
                    return
            except Exception:  # noqa: BLE001
                pass
        v = self.get_parameter("initial_pose").value
        if v == "ground_truth":
            T = self.robot.gt
            if T is None:
                return
            x, y, yaw = T[0, 3], T[1, 3], math.atan2(T[1, 0], T[0, 0])
        else:
            x, y, yaw = (float(s) for s in v.split(","))
        m = PoseWithCovarianceStamped()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.pose.position.x, m.pose.pose.position.y = x, y
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        m.pose.covariance[0] = m.pose.covariance[7] = 0.05 ** 2
        m.pose.covariance[35] = math.radians(5) ** 2
        self.pub_init.publish(m)
        self.init_pose_sent += 1
        self.get_logger().info(f"initial pose for AMCL: {x:.2f} {y:.2f} {math.degrees(yaw):.0f} deg ({v})")


def main():
    rclpy.init()
    node = MissionNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: BLE001
        if rclpy.ok():
            raise
    finally:
        if node.recorder is not None and node.recorder.recording:
            node.recorder.stop(False, discard=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
