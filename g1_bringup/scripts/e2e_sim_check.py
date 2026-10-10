#!/usr/bin/env python3
"""End-to-end check of the sim stack through the real interfaces (run while sim.launch.py is up):

    source env.sh
    ros2 launch g1_bringup sim.launch.py &            # headless, hospital, placeholder policy
    python3 ros2_ws/src/hospital-sim-ros2/g1_bringup/scripts/e2e_sim_check.py [--walk-time 8] [--json out.json]

Sequence (keys go through /g1/teleop/key exactly like the keyboard node):
  r -> STAND (stand-up, harness release), l -> LOCO, hold w (key repeat 10 Hz) -> walk, release -> stop,
  p -> STAND, m -> MANIP + upper_body_target_raw / hand cmd_raw -> arms/hands move, n -> blend back,
  l then m -> LOCO->MANIP must be rejected, x -> DAMP (all motors kp = 0).
Prints a JSON report and exits non-zero if a check fails.
"""
import argparse
import json
import math
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from unitree_hg.msg import LowCmd, LowState

from g1_interfaces.msg import Mode

NAMES = {0: "DAMP", 1: "STAND", 2: "LOCO", 3: "MANIP", 4: "NAV"}


class Checker(Node):
    def __init__(self):
        super().__init__("g1_e2e_check")
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.key_pub = self.create_publisher(String, "/g1/teleop/key", 10)
        self.upper_pub = self.create_publisher(JointState, "/g1/teleop/upper_body_target_raw", 10)
        self.hand_pub = self.create_publisher(JointState, "/g1/teleop/hand/left/cmd_raw", 10)
        self.mode = None
        self.gt = None
        self.js = {}
        self.ctrl = {}
        self.sim = {}
        self.mgr = {}
        self.hand_state = None
        self.lowcmd = None
        self.n_lowstate = 0
        self.n_lowcmd = 0
        self.create_subscription(Mode, "/g1/mode", lambda m: setattr(self, "mode", m), latched)
        self.create_subscription(Odometry, "/sim/ground_truth", lambda m: setattr(self, "gt", m), 10)
        self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.create_subscription(String, "/g1/controller/status", lambda m: setattr(self, "ctrl", json.loads(m.data)), 10)
        self.create_subscription(String, "/sim/status", lambda m: setattr(self, "sim", json.loads(m.data)), 10)
        self.create_subscription(String, "/g1/teleop/manager_status", lambda m: setattr(self, "mgr", json.loads(m.data)), 10)
        self.create_subscription(JointState, "/g1/hand/left/state", lambda m: setattr(self, "hand_state", list(m.position)), 10)
        self.create_subscription(LowCmd, "/lowcmd", self.on_lowcmd, be)
        self.create_subscription(LowState, "/lowstate", self.on_lowstate, be)

    def on_js(self, m):
        self.js = dict(zip(m.name, m.position))

    def on_lowcmd(self, m):
        self.n_lowcmd += 1
        self.lowcmd = m

    def on_lowstate(self, m):
        self.n_lowstate += 1

    def spin_for(self, sec, every=None, fn=None):
        end = time.monotonic() + sec
        nxt = time.monotonic()
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.01)
            if every and time.monotonic() >= nxt:
                fn()
                nxt += every

    def key(self, k):
        self.key_pub.publish(String(data=k))
        self.spin_for(0.3)

    def pos(self):
        p = self.gt.pose.pose.position
        return p.x, p.y, p.z

    def yaw(self):
        q = self.gt.pose.pose.orientation
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def mode_name(self):
        return NAMES.get(self.mode.mode) if self.mode else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk-time", type=float, default=12.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rclpy.init()
    c = Checker()
    rep, fails = {}, []

    def check(name, ok, detail=""):
        rep.setdefault("checks", {})[name] = {"ok": bool(ok), "detail": detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails.append(name)

    t0 = time.monotonic()
    while (c.gt is None or not c.ctrl or c.mode is None) and time.monotonic() - t0 < 30:
        c.spin_for(0.2)
    check("stack_up", c.gt is not None and c.ctrl and c.mode is not None, f"mode={c.mode_name()}")
    if fails:
        return 1
    # rates
    n0s, n0c, t = c.n_lowstate, c.n_lowcmd, time.monotonic()
    c.spin_for(3.0)
    dt = time.monotonic() - t
    rep["rates_idle"] = {"lowstate_hz": round((c.n_lowstate - n0s) / dt, 1), "lowcmd_hz": round((c.n_lowcmd - n0c) / dt, 1),
                         "sim": c.sim}
    print("rates", rep["rates_idle"])
    rep["start_pose"] = c.pos()

    # 1. stand
    c.key("r")
    c.spin_for(6.0)
    z = c.pos()[2]
    check("stand", c.mode_name() == "STAND" and c.ctrl.get("phase") == "policy" and c.sim.get("harness", 1) == 0
          and z > 0.65, f"mode={c.mode_name()} phase={c.ctrl.get('phase')} harness={c.sim.get('harness')} z={z:.3f}")
    stand_xy = c.pos()[:2]
    c.spin_for(4.0)
    drift = math.dist(stand_xy, c.pos()[:2])
    rep["stand_drift_4s_m"] = round(drift, 3)
    check("stand_in_place", drift < 0.5 and c.pos()[2] > 0.65, f"drift {drift:.3f} m in 4 s")

    # 2. walk forward with w held (10 Hz key repeat)
    c.key("l")
    check("loco_mode", c.mode_name() == "LOCO", f"mode={c.mode_name()}")
    p_start, yaw0 = c.pos(), c.yaw()
    tw = time.monotonic()
    n0s, n0c = c.n_lowstate, c.n_lowcmd
    rtfs, loops = [], []

    def hold_w():
        c.key_pub.publish(String(data="w"))
        if c.sim:
            rtfs.append(c.sim.get("rtf", 0))
        if c.ctrl:
            loops.append((c.ctrl.get("loop_hz", 0), c.ctrl.get("policy_hz", 0)))
    c.spin_for(a.walk_time, every=0.1, fn=hold_w)
    walk_wall = time.monotonic() - tw
    rep["rates_walking"] = {"lowstate_hz": round((c.n_lowstate - n0s) / walk_wall, 1),
                            "lowcmd_hz": round((c.n_lowcmd - n0c) / walk_wall, 1),
                            "controller_loop_hz": loops[-1][0] if loops else None,
                            "policy_hz": loops[-1][1] if loops else None,
                            "sim_rtf_min": min(rtfs) if rtfs else None, "sim_rtf_last": rtfs[-1] if rtfs else None,
                            "sim_step_ms_mean": c.sim.get("step_ms_mean")}
    cmd_during = c.mgr.get("cmd_vel")
    # 3. release -> 0.5 s timeout -> stop
    c.spin_for(3.0)
    p_end = c.pos()
    dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
    fwd = dx * math.cos(yaw0) + dy * math.sin(yaw0)
    lat = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
    rep["walk"] = {"forward_m": round(fwd, 2), "lateral_m": round(lat, 2), "wall_s": round(walk_wall, 1),
                   "cmd_vel_at_end_of_hold": cmd_during, "z_end": round(p_end[2], 3),
                   "yaw_change_rad": round(c.yaw() - yaw0, 3)}
    print("walk", rep["walk"])
    check("walk_forward", fwd > 2.0 and p_end[2] > 0.6, f"{fwd:.2f} m forward, {lat:.2f} m lateral")
    p1 = c.pos()
    c.spin_for(2.0)
    creep = math.dist(p1[:2], c.pos()[:2])
    check("stops_after_key_timeout", creep < 0.4 and c.mgr.get("cmd_vel") == [0.0, 0.0, 0.0],
          f"moved {creep:.2f} m in 2 s after release; manager cmd {c.mgr.get('cmd_vel')}")

    # 4. pause, manip
    c.key("p")
    check("pause_to_stand", c.mode_name() == "STAND", f"mode={c.mode_name()}")
    c.key("m")
    check("manip_mode", c.mode_name() == "MANIP", f"mode={c.mode_name()}")
    # "Hold a tray" pose. The legs-only PLACEHOLDER policy tolerates this (drift ~0.03 m/s in MuJoCo); large
    # forward reaches with both arms (e.g. shoulder pitch -0.9, elbow 1.0) shift the CoM and it walks off / falls.
    names = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint",
             "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint"]
    vals = [-0.3, 0.3, 1.3, -0.3, -0.3, 1.3]
    before = [c.js.get(n, 0.0) for n in names]

    def send_upper():
        c.upper_pub.publish(JointState(name=names, position=vals))
        c.hand_pub.publish(JointState(position=[0.8] * 6))
    c.spin_for(3.0, every=1 / 30, fn=send_upper)
    after = [c.js.get(n, 0.0) for n in names]
    err = max(abs(x - v) for x, v in zip(after, vals))
    rep["manip"] = {"before": [round(v, 3) for v in before], "after": [round(v, 3) for v in after], "target": vals,
                    "max_err_rad": round(err, 3), "left_hand_state": [round(v, 2) for v in (c.hand_state or [])],
                    "z": round(c.pos()[2], 3)}
    print("manip", rep["manip"])
    check("arms_track_target", err < 0.15, f"max err {err:.3f} rad")
    check("hand_closes", c.hand_state is not None and min(c.hand_state) > 0.5, f"{c.hand_state}")
    check("balanced_in_manip", c.pos()[2] > 0.6, f"z {c.pos()[2]:.3f}")
    manip_xy = c.pos()[:2]
    c.spin_for(3.0, every=1 / 30, fn=send_upper)
    rep["manip"]["drift_3s_m"] = round(math.dist(manip_xy, c.pos()[:2]), 3)
    check("balanced_in_manip_6s", c.pos()[2] > 0.65, f"z {c.pos()[2]:.3f}, drift {rep['manip']['drift_3s_m']} m / 3 s")
    c.key("n")
    detail0 = c.mode.detail
    c.spin_for(1.8)
    back = [c.js.get(n, 0.0) for n in names]
    rep["manip_exit"] = {"detail_right_after_n": detail0, "detail_after_1_8s": c.mode.detail,
                         "arms_after": [round(v, 3) for v in back]}
    check("manip_blend_out", c.mode_name() == "STAND" and detail0.startswith("blending") and c.mode.detail == "ready"
          and max(abs(v) for v in back) < 0.15, json.dumps(rep["manip_exit"]))

    # 5. LOCO -> MANIP rejected
    c.key("l")
    c.key("m")
    c.spin_for(1.2)
    check("loco_to_manip_rejected", c.mode_name() == "LOCO", f"mode={c.mode_name()} msg={c.mgr.get('last_message')}")
    rep["loco_to_manip_message"] = c.mgr.get("last_message")

    # 6. e-stop
    c.key("x")
    c.spin_for(0.5)
    kp_max = max(c.lowcmd.motor_cmd[i].kp for i in range(29)) if c.lowcmd else None
    kd_min = min(c.lowcmd.motor_cmd[i].kd for i in range(29)) if c.lowcmd else None
    check("estop_damps", c.mode_name() == "DAMP" and c.ctrl.get("phase") == "damp" and kp_max == 0.0 and kd_min > 0,
          f"mode={c.mode_name()} phase={c.ctrl.get('phase')} lowcmd kp_max={kp_max} kd_min={kd_min}")
    c.spin_for(3.0)
    rep["after_estop_z"] = round(c.pos()[2], 3)
    rep["controller_status"] = c.ctrl
    rep["sim_status"] = c.sim
    rep["failures"] = fails
    print(json.dumps(rep, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
    c.destroy_node()
    rclpy.shutdown()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
