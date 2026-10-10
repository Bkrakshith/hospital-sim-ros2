#!/usr/bin/env python3
"""Can each instrument be lifted straight out of the tool rack? Translates every instrument vertically in 5 mm
steps (no physics) and reports the worst penetration with the rack. A pick lifts straight up, so any
interference > ~0.5 mm means the instrument jams (the rack holes are sized to the section at guide height).

    python ros2_ws/src/hospital-sim-ros2/g1_mission/scripts/check_extraction.py [--scene g1-mujoco-sim/hospital_g1.xml] [--lift 0.10]
"""
import argparse
import os

import mujoco
import numpy as np

# master repo root: $HG1_ROOT (env.sh), else from this file (ros2_ws/src/hospital-sim-ros2/g1_mission/scripts/)
ROOT = os.environ.get("HG1_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=os.path.join(ROOT, "g1-mujoco-sim", "hospital_g1.xml"))
    ap.add_argument("--lift", type=float, default=0.10)
    a = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(a.scene)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rack = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tool_rack")
    bad = 0
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE or m.jnt_bodyid[j] in (1,):
            continue
        b = m.jnt_bodyid[j]
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if name in ("pelvis", "tool_rack"):
            continue
        qa = m.jnt_qposadr[j]
        q0 = d.qpos[qa:qa + 7].copy()
        worst, at = 0.0, None
        for dz in np.arange(0.0, a.lift + 1e-9, 0.005):
            d.qpos[qa:qa + 7] = q0
            d.qpos[qa + 2] += dz
            mujoco.mj_forward(m, d)
            for c in d.contact[:d.ncon]:
                if {m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]} == {b, rack} and c.dist < worst:
                    worst, at = c.dist, dz
        d.qpos[qa:qa + 7] = q0
        mujoco.mj_forward(m, d)
        ok = worst > -0.0005
        bad += not ok
        print(f"{name:16s} {'OK  ' if ok else 'JAMS'} worst penetration {worst * 1000:5.1f} mm"
              + ("" if ok else f" at lift {at * 100:.1f} cm"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
