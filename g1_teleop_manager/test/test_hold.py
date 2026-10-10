"""HOLD ("Stop", docs/ui/README.md decision 1): transitions, arm freeze semantics, resume paths. Pure FSM."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from g1_teleop_manager.fsm import (DAMP, STAND, LOCO, MANIP, NAV, AUTO, HOLD, FORWARD, KEY_TO_MODE,  # noqa: E402
                                   LEGAL, ModeFSM, forwards)

ORIG = [DAMP, STAND, LOCO, MANIP, NAV, AUTO]
DEFAULT = [0.0] * 17


def fsm_in(mode):
    f = ModeFSM(blend_time=1.0, hold_blend_time=2.5, default_upper=DEFAULT)
    path = {DAMP: [], STAND: [STAND], LOCO: [STAND, LOCO], MANIP: [STAND, MANIP], NAV: [STAND, NAV],
            AUTO: [STAND, AUTO]}[mode]
    for m in path:
        assert f.request(m, "mission" if m == AUTO else "test", 0.0)[0]
    return f


@pytest.mark.parametrize("m", [STAND, LOCO, MANIP, NAV, AUTO])
def test_stop_from_every_active_mode(m):
    f = fsm_in(m)
    ok, msg = f.request(HOLD, "ui:ipad-1", 5.0)
    assert ok and f.mode == HOLD and f.hold_from == m and f.source == "ui:ipad-1", msg
    assert not f.blending(5.0)


def test_stop_in_damp_is_noop():
    f = fsm_in(DAMP)
    ok, msg = f.request(HOLD, "ui:x", 1.0)
    assert ok and f.mode == DAMP and "powered off" in msg


def test_stop_twice_is_noop():
    f = fsm_in(LOCO)
    f.request(HOLD, "keyboard", 1.0)
    ok, msg = f.request(HOLD, "keyboard", 2.0)
    assert ok and f.mode == HOLD and f.hold_from == LOCO


def test_hold_not_in_legal_table():
    # the original 6-mode table is unchanged; HOLD rules live in the FSM
    assert HOLD not in LEGAL and all(HOLD not in v for v in LEGAL.values())


@pytest.mark.parametrize("target", [LOCO, MANIP, NAV])
def test_leaving_hold_only_by_resume(target):
    f = fsm_in(LOCO)
    f.request(HOLD, "ui:a", 1.0)
    ok, msg = f.request(target, "keyboard", 2.0)
    assert not ok and f.mode == HOLD and "resume" in msg


def test_hold_to_stand_blends_slowly_from_frozen_pose():
    f = fsm_in(MANIP)
    f.request(HOLD, "ui:a", 1.0)
    frozen = [0.4] * 17
    ok, _ = f.request(STAND, "ui:a", 3.0, upper_target=frozen)
    assert ok and f.mode == STAND and f.blend_from == frozen and f.cur_blend_time >= 2.0
    assert f.blending(4.9) and not f.request(LOCO, "keyboard", 4.9)[0]
    assert f.tick(5.51) and f.detail == "ready"


def test_hold_to_stand_no_blend_when_arms_at_default():
    f = fsm_in(NAV)
    f.request(HOLD, "ui:a", 1.0)
    ok, _ = f.request(STAND, "ui:a", 2.0, upper_target=[0.01] * 17)
    assert ok and not f.blending(2.0) and f.detail == "ready"
    assert f.request(NAV, "ui:a", 2.1)[0]


def test_hold_blend_time_at_least_2s():
    assert ModeFSM(hold_blend_time=0.5).hold_blend_time == 2.0


def test_hold_to_auto_only_when_held_from_auto():
    f = fsm_in(AUTO)
    f.request(HOLD, "ui:a", 1.0)
    assert not f.request(AUTO, "keyboard", 2.0)[0]           # wrong source
    ok, msg = f.request(AUTO, "mission", 2.0)
    assert ok and f.mode == AUTO and f.detail == "mission resumed", msg
    g = fsm_in(NAV)
    g.request(HOLD, "ui:a", 1.0)
    ok, msg = g.request(AUTO, "mission", 2.0)
    assert not ok and g.mode == HOLD and "stopped task" in msg


def test_ui_with_identity_counts_as_ui_for_auto():
    f = fsm_in(STAND)
    assert f.request(AUTO, "ui:ipad-7", 1.0)[0]


def test_hold_to_damp():
    f = fsm_in(AUTO)
    f.request(HOLD, "ui:a", 1.0)
    ok, _ = f.request(DAMP, "ui:a", 2.0)
    assert ok and f.mode == DAMP and f.hold_from is None


def test_stop_during_manip_blend_freezes_arms():
    f = fsm_in(MANIP)
    f.request(STAND, "keyboard", 10.0, upper_target=[0.5] * 17)
    assert f.blending(10.3)
    ok, _ = f.request(HOLD, "ui:a", 10.3)
    assert ok and f.mode == HOLD and not f.blending(10.3) and f.blend_from is None


def test_hold_forwards_nothing():
    for ch in FORWARD:
        assert not forwards(HOLD, ch), ch


def test_keys_hold_and_resume():
    assert KEY_TO_MODE["hold"] == (HOLD, None)
    mode, only_from = KEY_TO_MODE["ready"]
    assert mode == STAND and HOLD in only_from and DAMP in only_from
    assert HOLD not in KEY_TO_MODE["pause"][1]              # p must not resume a stop


@pytest.mark.parametrize("m", ORIG)
def test_damp_still_from_anywhere(m):
    f = fsm_in(m)
    assert f.request(DAMP, "ui:x", 3.0)[0] and f.mode == DAMP


# ---------------------------------------------------------------- arms hold inside MANIP, gripper override
def test_arms_hold_only_in_manip_and_cleared_on_mode_change():
    f = fsm_in(STAND)
    assert not f.set_arms_hold(True)[0]
    f = fsm_in(MANIP)
    ok, _ = f.set_arms_hold(True)
    assert ok and f.arms_hold and f.mode == MANIP and f.detail == "arms held"
    assert f.set_arms_hold(False)[0] and not f.arms_hold
    f.set_arms_hold(True)
    f.request(HOLD, "ui:a", 1.0)
    assert not f.arms_hold
    g = fsm_in(MANIP)
    g.set_arms_hold(True)
    g.request(STAND, "keyboard", 2.0)
    assert not g.arms_hold


def test_hand_override_allowed_only_in_hold_or_held_manip():
    for m in (DAMP, STAND, LOCO, NAV, AUTO):
        assert not fsm_in(m).hand_override_allowed(), m
    f = fsm_in(MANIP)
    assert not f.hand_override_allowed()
    f.set_arms_hold(True)
    assert f.hand_override_allowed()
    h = fsm_in(AUTO)
    h.request(HOLD, "ui:a", 1.0)
    assert h.hand_override_allowed()
