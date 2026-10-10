#!/usr/bin/env python3
"""Pick -> handover -> return trials through the full ROS stack (sim + controller + manager + mission_node).
Start the stack first (fixed base):  ros2 launch g1_bringup mission.launch.py
then:
    python ros2_ws/src/hospital-sim-ros2/g1_mission/scripts/run_trials.py --instruments scalpel_handle needle_driver --trials 10 \
        --offset 0.02 0.02 0.08 --json /tmp/trials.json
Per trial: set the sim parameter object_offset to a uniform random (dx, dy, dyaw) in +-offset, /sim/reset_objects,
request the instrument (/g1/mission/request_instrument) and wait for the mission history entry. The robot is put in
STAND (key r) first if needed; the mission is started by the first request.
"""
import argparse
import json
import sys
import time

import numpy as np
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from g1_interfaces.srv import RequestInstrument


class Runner(Node):
    def __init__(self):
        super().__init__("g1_mission_trials")
        self.status = None
        self.sim = None
        self.create_subscription(String, "/g1/mission/status", self._st, 10)
        self.create_subscription(String, "/sim/status", self._sim, 10)
        self.key = self.create_publisher(String, "/g1/teleop/key", 10)
        self.req = self.create_client(RequestInstrument, "/g1/mission/request_instrument")
        self.reset = self.create_client(Trigger, "/sim/reset_objects")
        self.setp = self.create_client(SetParameters, "/g1_mujoco_sim/set_parameters")

    def _st(self, m):
        self.status = json.loads(m.data)

    def _sim(self, m):
        self.sim = json.loads(m.data)

    def spin_for(self, t):
        t0 = time.time()
        while time.time() - t0 < t:
            rclpy.spin_once(self, timeout_sec=0.05)

    def call(self, cli, req, timeout=10.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {cli.srv_name} not available")
        f = cli.call_async(req)
        t0 = time.time()
        while not f.done() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        return f.result()

    def set_offset(self, off):
        p = Parameter(name="object_offset", value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                                                                 double_array_value=[float(v) for v in off]))
        return self.call(self.setp, SetParameters.Request(parameters=[p]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruments", nargs="+", default=["scalpel_handle", "needle_driver"])
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--offset", nargs=3, type=float, default=[0.02, 0.02, 0.08], help="dx dy (m) dyaw (rad) range")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=90.0, help="s wall per trial")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rclpy.init()
    n = Runner()
    rng = np.random.default_rng(a.seed)
    n.spin_for(2.0)
    if n.status is None:
        print("no /g1/mission/status: is mission_node running?")
        return 1
    if n.status.get("mode") in (None, "DAMP"):
        n.key.publish(String(data="r"))
        print("sent r (DAMP -> STAND), waiting for stand-up")
        n.spin_for(6.0)
    results = []
    t_all = time.time()
    for inst in a.instruments:
        for k in range(a.trials):
            # wait until the mission is idle again (recovery / walking back finished) before touching the world
            t_idle = time.time()
            while time.time() - t_idle < 120 and n.status and n.status["state"] == "running" \
                    and n.status.get("step") != "WaitForRequest":
                n.spin_for(0.5)
            off = rng.uniform(-1, 1, 3) * np.array(a.offset)
            n.set_offset(off)
            n.call(n.reset, Trigger.Request())
            n.spin_for(1.0)
            ok_before = n.status["counts"]["ok"] + n.status["counts"]["failed"] if n.status else 0
            r = n.call(n.req, RequestInstrument.Request(instrument=inst, requester="run_trials"))
            if not r.accepted:
                print("request rejected:", r.message)
                return 1
            t0 = time.time()
            res = None
            while time.time() - t0 < a.timeout:
                n.spin_for(0.2)
                st = n.status
                if st and st["counts"]["ok"] + st["counts"]["failed"] > ok_before:
                    res = st["history"][-1]
                    break
                if st and st["state"] in ("failed", "aborted"):
                    res = {"instrument": inst, "success": False, "message": f"mission {st['state']}: {st['message']}"}
                    break
            if res is None:
                res = {"instrument": inst, "success": False, "message": "timeout"}
            res = dict(res, offset=[round(float(v), 4) for v in off], wall=round(time.time() - t0, 1),
                       rtf=(n.sim or {}).get("rtf"))
            results.append(res)
            print(f"{inst:15s} {k:2d} offset {res['offset']} -> {'OK  ' if res['success'] else 'FAIL'} "
                  f"{res.get('message', '')} ({res['wall']} s wall, rtf {res['rtf']})", flush=True)
            if n.status and n.status["state"] in ("failed", "aborted"):
                print("mission stopped:", n.status["message"])
                break
    summary = {i: {"success": sum(r["success"] for r in results if r["instrument"] == i),
                   "trials": sum(r["instrument"] == i for r in results)} for i in a.instruments}
    print("summary", summary, f"total wall {time.time() - t_all:.0f} s")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summary, "offset_range": a.offset, "seed": a.seed, "results": results}, f, indent=1)
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
