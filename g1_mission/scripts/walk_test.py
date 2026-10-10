#!/usr/bin/env python3
"""Walking (fixed_base:=false) mission check with the placeholder policy, through the full ROS stack.
Start the stack first:  ros2 launch g1_bringup mission.launch.py fixed_base:=false   (Nav2 + AMCL, spawn corridor_west)
    python ros2_ws/src/hospital-sim-ros2/g1_mission/scripts/walk_test.py [--arm scalpel_handle] [--json out.json]
1. r (STAND), wait for Nav2's navigate_to_pose server
2. goto:theatre_instrument_table  (EnsureAuto -> ToReady -> GoTo (Nav2) -> FineAlign -> ReleaseAuto)
3. optionally request an instrument (the arm part) and report what happens (the placeholder policy is not trained
   for arm reaches)
Logs every mission step change with sim time, ground-truth pose, controller fall flag.
"""
import argparse
import json
import math
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

from g1_interfaces.srv import RequestInstrument


class W(Node):
    def __init__(self):
        super().__init__("g1_walk_test")
        self.st, self.sim, self.ctrl, self.gt = None, {}, {}, None
        self.create_subscription(String, "/g1/mission/status", lambda m: setattr(self, "st", json.loads(m.data)), 10)
        self.create_subscription(String, "/sim/status", lambda m: setattr(self, "sim", json.loads(m.data)), 10)
        self.create_subscription(String, "/g1/controller/status", lambda m: setattr(self, "ctrl", json.loads(m.data)), 10)
        self.create_subscription(Odometry, "/sim/ground_truth", self._gt, 10)
        self.key = self.create_publisher(String, "/g1/teleop/key", 10)
        self.cmd = self.create_publisher(String, "/g1/mission/command", 10)
        self.req = self.create_client(RequestInstrument, "/g1/mission/request_instrument")
        from nav2_msgs.action import NavigateToPose
        self.nav = ActionClient(self, NavigateToPose, "/navigate_to_pose")

    def _gt(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        self.gt = (p.x, p.y, p.z, math.atan2(2 * (o.w * o.z + o.x * o.y), 1 - 2 * (o.y ** 2 + o.z ** 2)))

    def spin(self, t):
        t0 = time.time()
        while time.time() - t0 < t:
            rclpy.spin_once(self, timeout_sec=0.05)


def watch(n, log, until, timeout):
    last = None
    t0 = time.time()
    while time.time() - t0 < timeout:
        n.spin(0.25)
        st = n.st or {}
        key = (st.get("state"), st.get("step"))
        if key != last:
            e = {"wall": round(time.time() - t0, 1), "sim_t": n.sim.get("sim_time"), "state": st.get("state"),
                 "step": st.get("step"), "msg": st.get("message"), "mode": st.get("mode"),
                 "pose": [round(v, 3) for v in n.gt] if n.gt else None,
                 "fall": n.ctrl.get("fall_detected"), "station_error": st.get("station_error")}
            log.append(e)
            print(json.dumps(e), flush=True)
            last = key
        if until(st):
            return True
        if n.ctrl.get("fall_detected") or st.get("mode") == "DAMP":
            print("robot fell / DAMP", flush=True)
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="", help="instrument to request after reaching the table")
    ap.add_argument("--json", default="")
    ap.add_argument("--timeout", type=float, default=900.0)
    a = ap.parse_args()
    rclpy.init()
    n = W()
    n.spin(3.0)
    log = []
    print("waiting for Nav2 ...", flush=True)
    t0 = time.time()
    while not n.nav.server_is_ready() and time.time() - t0 < 120:
        n.spin(1.0)
    print("nav2 ready:", n.nav.server_is_ready(), flush=True)
    n.key.publish(String(data="r"))
    n.spin(8.0)                      # stand-up + harness release
    n.cmd.publish(String(data="goto:theatre_instrument_table"))
    ok = watch(n, log, lambda st: st.get("state") in ("succeeded", "failed", "aborted"), a.timeout)
    res = {"goto": (n.st or {}).get("state"), "goto_message": (n.st or {}).get("message"), "log": log,
           "final_pose": n.gt, "fall": n.ctrl.get("fall_detected")}
    print("GOTO RESULT", res["goto"], res["goto_message"], "pose", n.gt, flush=True)
    if a.arm and res["goto"] == "succeeded":
        n.cmd.publish(String(data="reset"))
        n.spin(1.0)
        f = n.req.call_async(RequestInstrument.Request(instrument=a.arm, requester="walk_test"))
        while not f.done():
            n.spin(0.1)
        print("request:", f.result().message, flush=True)
        log2 = []
        watch(n, log2, lambda st: bool(st.get("history")) or st.get("state") in ("failed", "aborted"), 240.0)
        res["arm"] = {"log": log2, "history": (n.st or {}).get("history"), "fall": n.ctrl.get("fall_detected"),
                      "mode": (n.st or {}).get("mode"), "final_pose": n.gt}
        print("ARM RESULT history", (n.st or {}).get("history"), "fall", n.ctrl.get("fall_detected"),
              "mode", (n.st or {}).get("mode"), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
