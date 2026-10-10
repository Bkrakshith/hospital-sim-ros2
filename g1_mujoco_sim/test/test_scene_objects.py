"""SceneObjects (ground truth, reset with rack offsets, simulated surgeon grip) and the fixed-base weld, on the
generated hospital scene (g1-mujoco-sim/hospital_g1.xml). Pure MuJoCo, no ROS."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
mujoco = pytest.importorskip("mujoco")
from g1_mujoco_sim.scene_objects import SceneObjects, weld_to_world  # noqa: E402

# master repo root: $HG1_ROOT (env.sh), else from this file (ros2_ws/src/hospital-sim-ros2/g1_mujoco_sim/test/)
ROOT = os.environ.get("HG1_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
SCENE = os.path.join(ROOT, "g1-mujoco-sim", "hospital_g1.xml")
SETUP = os.path.join(ROOT, "g1-mujoco-sim", "worlds", "theatre_setup.json")
pytestmark = pytest.mark.skipif(not os.path.exists(SETUP), reason="run g1-mujoco-sim/build_world.py first")


@pytest.fixture
def sim():
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d, SceneObjects(m, SETUP)


def test_snapshot_lists_every_free_object(sim):
    m, d, so = sim
    snap = json.loads(so.to_json(d))
    setup = json.load(open(SETUP))
    for name in setup["instruments"]:
        assert snap[name]["type"] == "instrument" and len(snap[name]["pose"]) == 7 and snap[name]["held_by"] is None
    assert snap["tool_rack"]["type"] == "rack"          # free (stand-ins) or static (hospital-sim-env/objects)
    assert snap["surgeon_hand"]["type"] == "surgeon_hand"
    assert np.allclose(snap["surgeon_hand"]["pose"][:3], setup["surgeon"]["hand"], atol=1e-4)


def test_reset_offset_moves_rack_and_instruments_rigidly(sim):
    m, d, so = sim
    p0 = {n: d.xpos[b].copy() for n, (b, _, _, _) in so.bodies.items()}
    c = so._rack_centre0()
    so.reset(d, (0.02, -0.01, 0.1))
    if so.static_rack >= 0:                              # static rack: moved through body_pos
        mujoco.mj_kinematics(m, d)
        assert np.allclose(d.xpos[so.static_rack][:2], c[:2] + [0.02, -0.01], atol=1e-9)
    R = np.array([[math.cos(0.1), -math.sin(0.1)], [math.sin(0.1), math.cos(0.1)]])
    for n, (b, _, _, _) in so.bodies.items():
        exp = c[:2] + R @ (p0[n][:2] - c[:2]) + [0.02, -0.01]
        assert np.allclose(d.xpos[b][:2], exp, atol=1e-9), n
    so.reset(d)
    for n, (b, _, _, _) in so.bodies.items():
        assert np.allclose(d.xpos[b], p0[n], atol=1e-9)


def test_simulated_grip_welds_the_instrument_to_the_hand(sim):
    m, d, so = sim
    name = "scalpel_handle"
    b, qa, da, _ = so.bodies[name]
    # hold the instrument still (as a robot hand would) with its handle end at the surgeon's hand
    hand = d.site_xpos[so.hand_site].copy()
    hz = so.handle_z.get(name) or 0.16
    hold = np.r_[hand - np.array([0.0, 0.0, hz]), 1.0, 0.0, 0.0, 0.0]
    got = None
    for _ in range(600):
        if got is None:
            d.qpos[qa:qa + 7], d.qvel[da:da + 6] = hold, 0.0
        mujoco.mj_step(m, d)
        got = so.update_grip(d, m.opt.timestep) or got
    assert got == name and so.held == name and json.loads(so.to_json(d))[name]["held_by"] == "surgeon"
    p = d.xpos[b].copy()
    for _ in range(500):
        mujoco.mj_step(m, d)
    assert np.linalg.norm(d.xpos[b] - p) < 0.01          # held by the weld, does not fall
    so.reset(d)
    assert so.held is None and not d.eq_active[so.grip_eq[name]]


def _grip(m, d, so, name):
    """Hold `name` still with its handle end at the surgeon's hand until the simulated grip welds it."""
    b, qa, da, _ = so.bodies[name]
    hand = d.site_xpos[so.hand_site].copy()
    hold = np.r_[hand - np.array([0.0, 0.0, so.handle_z.get(name) or 0.16]), 1.0, 0.0, 0.0, 0.0]
    got = None
    for _ in range(600):
        if got is None:
            d.qpos[qa:qa + 7], d.qvel[da:da + 6] = hold, 0.0
        mujoco.mj_step(m, d)
        got = so.update_grip(d, m.opt.timestep) or got
    assert got == name and so.held == name


def test_held_instrument_is_put_down_when_the_next_one_is_offered(sim):
    """Sessions without /sim/reset_objects (the app): the instrument the surgeon still holds sits welded at the grip
    point along the handover axis, so the next instrument (angled_tweezers after scissors, measured) jammed against
    it 5-9 cm short of the 4.5 cm grip radius and the handover timed out. The surgeon puts the held instrument back
    (its rack slot) once another instrument that is out of the rack comes within put_down_radius of the hand."""
    m, d, so = sim
    so.reset(d)
    _grip(m, d, so, "scissors")
    sb, sqa, _, _ = so.bodies["scissors"]
    home = so.q0["scissors"][:3].copy()
    # every other instrument rests in its rack slot: nothing is offered, the surgeon keeps the scissors (even with
    # a radius that covers the rack)
    so.put_down_radius = 10.0
    for _ in range(50):
        mujoco.mj_step(m, d)
        so.update_grip(d, m.opt.timestep)
    assert so.held == "scissors" and d.eq_active[so.grip_eq["scissors"]]
    so.put_down_radius = SceneObjects.PUT_DOWN_RADIUS
    # the robot brings the tweezers: handle end 0.25 m from the surgeon's hand, out of the rack
    tb, tqa, tda, _ = so.bodies["angled_tweezers"]
    hand = d.site_xpos[so.hand_site].copy()
    hz = so.handle_z["angled_tweezers"]
    d.qpos[tqa:tqa + 7] = np.r_[hand + np.array([0.25, 0.0, 0.0]) - np.array([0.0, 0.0, hz]), 1.0, 0.0, 0.0, 0.0]
    d.qvel[tda:tda + 6] = 0.0
    mujoco.mj_forward(m, d)
    so.update_grip(d, m.opt.timestep)
    assert so.held is None and not d.eq_active[so.grip_eq["scissors"]]
    assert np.allclose(d.qpos[sqa:sqa + 3], home, atol=1e-9)           # back in its slot
    assert json.loads(so.to_json(d))["surgeon_hand"]["holding"] is None


def test_fixed_base_weld_holds_the_pelvis(sim):
    m, d, so = sim
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] == 1][0]
    qa = m.jnt_qposadr[free]
    st = json.load(open(SETUP))["station"]
    d.qpos[qa:qa + 3] = [st["x"], st["y"], 0.9]                  # in the air: gravity must not move it
    d.qpos[qa + 3:qa + 7] = [math.cos(st["yaw"] / 2), 0, 0, math.sin(st["yaw"] / 2)]
    mujoco.mj_forward(m, d)
    p0 = d.qpos[qa:qa + 3].copy()
    weld_to_world(m, d, "fixed_base", "pelvis")
    for _ in range(1000):
        mujoco.mj_step(m, d)
    assert np.linalg.norm(d.qpos[qa:qa + 3] - p0) < 0.005


def test_stamp_is_monotonic_across_whole_seconds():
    """/clock must never go back: 500 x 0.002 s sums to 0.99999999999 s, which the old stamp helper turned into
    0.0 s (every tf2 buffer then cleared itself)."""
    pytest.importorskip("builtin_interfaces")
    from g1_mujoco_sim.sim_node import _stamp
    t, last = 0.0, -1
    for _ in range(20000):
        t += 0.002
        st = _stamp(t)
        ns = st.sec * 1000000000 + st.nanosec
        assert ns > last and 0 <= st.nanosec < 1000000000
        last = ns
