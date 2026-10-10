#!/usr/bin/env python3
"""Offline (no ROS) trials of the real scrub_nurse tree (pick -> [carry] -> handover -> retract -> ready -> [back])
on the in-process MuJoCo model (g1_mission.offline, fixed base; walking = the fixed-base glide).

    python ros2_ws/src/hospital-sim-ros2/g1_mission/scripts/offline_trials.py --instruments scalpel_handle needle_driver --trials 10 \
        [--offset 0.02 0.02 0.08] [--seed 0] [--video out.mp4] [--json out.json]
Rack offsets: uniform in +-offset (dx m, dy m, dyaw rad) per trial, applied to the rack and its instruments.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from g1_mission.offline import OfflineRobot, make_context  # noqa: E402

class _Done:
    def done(self):
        return True

    def result(self):
        return True, "ok"


def make_tree(robot, ctx):
    """The real scrub_nurse tree (repeat forever) with the offline glide as navigation and AUTO granted."""
    from g1_mission.offline import OfflineNav
    from g1_mission.tree import AUTO, MissionCore, build_scrub_nurse
    cfg = dict(robot.mission_cfg, rate=50.0, repeat=True)
    core = MissionCore(ctx, cfg, nav=OfflineNav(robot), mode_fn=lambda: AUTO, request_mode=lambda m: _Done(),
                       log=lambda s: None)
    root = build_scrub_nurse(core)
    root.setup_with_descendants()
    return core, root


def trial(robot, ctx, core, root, inst, timeout=150.0):
    n0 = len(core.history)
    core.queue.append(inst)
    t0 = robot.now()
    while robot.now() - t0 < timeout:
        root.tick_once()
        robot.advance(0.02)
        if len(core.history) > n0:
            break
    if len(core.history) == n0:
        return {"instrument": inst, "success": False, "message": "timeout", "time": round(robot.now() - t0, 1)}
    h = dict(core.history[-1])
    held = robot.objects.snapshot(robot.d).get(inst, {}).get("held_by")
    h["surgeon_holds"] = held == "surgeon"
    h["success"] = bool(h["success"] and h["surgeon_holds"])
    h["time"] = round(robot.now() - t0, 1)
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruments", nargs="+", default=["scalpel_handle", "needle_driver"])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--offset", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default="")
    ap.add_argument("--camera", default="theatre_station")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    render = {"camera": a.camera, "size": (640, 480), "fps": 25} if a.video else None
    robot = OfflineRobot(render=render)
    ctx = make_context(robot)
    ctx.log = lambda s: print("   ", s)
    robot.advance(1.0)
    core, root = make_tree(robot, ctx)
    results = []
    t0 = time.time()
    for inst in a.instruments:
        for k in range(a.trials):
            off = tuple(rng.uniform(-1, 1, 3) * np.array(a.offset))
            robot.reset_objects(off)
            robot.advance(0.5)
            r = trial(robot, ctx, core, root, inst)
            r["offset"] = [round(float(v), 4) for v in off]
            results.append(r)
            print(f"{inst:15s} trial {k:2d} offset {r['offset']} -> {'OK ' if r['success'] else 'FAIL'} "
                  f"{'' if r['success'] else r.get('message', '')}  ({r['time']:.1f} s sim)", flush=True)
    summary = {}
    for inst in a.instruments:
        rs = [r for r in results if r["instrument"] == inst]
        summary[inst] = {"success": sum(r["success"] for r in rs), "trials": len(rs)}
    print("summary", summary, f"wall {time.time() - t0:.0f} s")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summary, "offset_range": a.offset, "results": results}, f, indent=1)
    if a.video and robot.frames:
        import imageio.v2 as iio
        w = iio.get_writer(a.video, fps=25, codec="libx264", quality=None,
                           ffmpeg_params=["-crf", "28", "-preset", "slow", "-pix_fmt", "yuv420p"])
        for fr in robot.frames:
            w.append_data(fr)
        w.close()
        print("video", a.video, len(robot.frames), "frames")


if __name__ == "__main__":
    main()
