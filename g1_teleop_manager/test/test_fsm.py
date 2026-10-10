"""Pure FSM tests (no ROS graph): every transition of docs/interfaces.md 3.2 plus key/velocity rules."""
import itertools
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from g1_teleop_manager.fsm import (DAMP, STAND, LOCO, MANIP, NAV, AUTO, LEGAL, ModeFSM,  # noqa: E402
                                   VelocityIntegrator, blend, clamp_twist)

ALL = [DAMP, STAND, LOCO, MANIP, NAV, AUTO]
EXPECTED_LEGAL = {(DAMP, STAND), (STAND, LOCO), (STAND, MANIP), (STAND, NAV), (STAND, DAMP),
                  (LOCO, STAND), (MANIP, STAND), (NAV, STAND),
                  (LOCO, DAMP), (MANIP, DAMP), (NAV, DAMP),
                  (STAND, AUTO), (AUTO, STAND), (AUTO, DAMP)}


def fsm_in(mode):
    f = ModeFSM(blend_time=1.0)
    path = {DAMP: [], STAND: [STAND], LOCO: [STAND, LOCO], MANIP: [STAND, MANIP], NAV: [STAND, NAV],
            AUTO: [STAND, AUTO]}[mode]
    for m in path:
        assert f.request(m, "mission" if m == AUTO else "test", 0.0)[0]
    assert f.mode == mode
    return f


@pytest.mark.parametrize("a,b", [(a, b) for a, b in itertools.product(ALL, ALL) if a != b])
def test_transition_matrix(a, b):
    f = fsm_in(a)
    ok, msg = f.request(b, "mission" if b == AUTO else "test", 10.0)
    assert ok == ((a, b) in EXPECTED_LEGAL), msg
    assert f.mode == (b if ok else a)


def test_legal_table_matches_contract():
    got = {(a, b) for a, bs in LEGAL.items() for b in bs} | {(a, DAMP) for a in ALL if a != DAMP}
    assert got == EXPECTED_LEGAL


def test_loco_to_manip_rejected_with_hint():
    f = fsm_in(LOCO)
    ok, msg = f.request(MANIP, "keyboard", 1.0)
    assert not ok and "pause" in msg
    assert f.mode == LOCO


@pytest.mark.parametrize("m", ALL)
def test_same_mode_noop(m):
    f = fsm_in(m)
    ok, _ = f.request(m, "test", 5.0)
    assert ok and f.mode == m


@pytest.mark.parametrize("m", ALL)
def test_estop_from_anywhere(m):
    f = fsm_in(m)
    ok, _ = f.request(DAMP, "keyboard", 5.0)
    assert ok and f.mode == DAMP and f.detail == "e-stop"


def test_safety_damp_detail():
    f = fsm_in(LOCO)
    f.request(DAMP, "safety", 1.0)
    assert f.mode == DAMP and f.source == "safety" and f.detail == "fall detected"


def test_manip_exit_blends_then_ready():
    f = fsm_in(MANIP)
    ok, _ = f.request(STAND, "keyboard", 100.0, upper_target=[0.5] * 17)
    assert ok and f.mode == STAND and f.detail.startswith("blending")
    assert f.blending(100.5)
    # during blend: LOCO/NAV/MANIP rejected, DAMP accepted
    for m in (LOCO, NAV, MANIP):
        assert not f.request(m, "keyboard", 100.5)[0]
    assert f.tick(101.01) is True and f.detail == "ready" and not f.blending(101.01)
    assert f.request(LOCO, "keyboard", 101.1)[0]


def test_damp_during_blend():
    f = fsm_in(MANIP)
    f.request(STAND, "keyboard", 0.0)
    assert f.request(DAMP, "keyboard", 0.2)[0]
    assert f.mode == DAMP and not f.blending(0.3)


def test_velocity_integration_limits_and_timeout():
    v = VelocityIntegrator({"vx": [-0.5, 0.8], "vy": [-0.4, 0.4], "wz": [-1.0, 1.0]},
                           {"vx": 0.1, "vy": 0.1, "wz": 0.2}, timeout=0.5)
    t = 0.0
    for _ in range(12):
        v.key("vx_up", t)
        t += 0.05
    assert v.get(t) == (0.8, 0.0, 0.0)
    for _ in range(3):
        v.key("wz_down", t)
    v.key("vy_up", t)
    assert v.get(t) == (0.8, 0.1, -0.6)
    v.key("zero", t)
    assert v.get(t) == (0.0, 0.0, 0.0)
    v.key("vx_down", t)
    for _ in range(10):
        v.key("vx_down", t)
    assert v.get(t)[0] == -0.5
    assert v.get(t + 0.51) == (0.0, 0.0, 0.0)  # timeout


def test_clamp_and_blend():
    lim = {"vx": [-0.5, 0.8], "vy": [-0.4, 0.4], "wz": [-1.0, 1.0]}
    assert clamp_twist((2.0, -1.0, 0.3), lim) == (0.8, -0.4, 0.3)
    assert blend([1.0], [0.0], 0.0) == [1.0] and blend([1.0], [0.0], 1.0) == [0.0]
    assert blend([1.0], [0.0], 0.5)[0] == pytest.approx(0.5)
