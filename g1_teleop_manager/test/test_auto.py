"""AUTO mode (docs/interfaces.md 3.2 + 6): transitions, operator override, and the input forwarding table."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from g1_teleop_manager.fsm import (DAMP, STAND, LOCO, MANIP, NAV, AUTO, AUTO_SOURCES, FORWARD,  # noqa: E402
                                   KEY_TO_MODE, ModeFSM, forwards)

ALL = [DAMP, STAND, LOCO, MANIP, NAV, AUTO]


def standing():
    f = ModeFSM(blend_time=1.0)
    assert f.request(STAND, "keyboard", 0.0)[0]
    return f


@pytest.mark.parametrize("src", sorted(AUTO_SOURCES))
def test_stand_to_auto_from_mission_or_ui(src):
    f = standing()
    ok, msg = f.request(AUTO, src, 1.0)
    assert ok and f.mode == AUTO and f.source == src, msg


@pytest.mark.parametrize("src", ["keyboard", "nav", "test", "safety", ""])
def test_stand_to_auto_rejected_from_other_sources(src):
    f = standing()
    ok, msg = f.request(AUTO, src, 1.0)
    assert not ok and f.mode == STAND and "AUTO" in msg


@pytest.mark.parametrize("start", [DAMP, LOCO, MANIP, NAV])
def test_auto_only_from_stand(start):
    f = ModeFSM(blend_time=1.0)
    for m in {DAMP: [], LOCO: [STAND, LOCO], MANIP: [STAND, MANIP], NAV: [STAND, NAV]}[start]:
        f.request(m, "test", 0.0)
    ok, _ = f.request(AUTO, "mission", 5.0)
    assert not ok and f.mode == start


@pytest.mark.parametrize("target", [LOCO, MANIP, NAV])
def test_auto_cannot_switch_to_other_active_modes(target):
    f = standing()
    f.request(AUTO, "mission", 1.0)
    ok, _ = f.request(target, "keyboard", 2.0)
    assert not ok and f.mode == AUTO


def test_auto_to_stand_by_set_mode_blends_arms():
    f = standing()
    f.request(AUTO, "mission", 1.0)
    ok, _ = f.request(STAND, "mission", 2.0, upper_target=[0.3] * 17)
    assert ok and f.mode == STAND and f.detail.startswith("blending")
    assert f.blend_from == [0.3] * 17
    assert not f.request(AUTO, "mission", 2.5)[0]        # still blending
    assert f.tick(3.01) and f.detail == "ready"
    assert f.request(AUTO, "mission", 3.1)[0]           # mission may re-enter after the blend


def test_p_key_is_operator_override_in_auto():
    action = "pause"
    mode, only_from = KEY_TO_MODE[action]
    assert mode == STAND and AUTO in only_from
    f = standing()
    f.request(AUTO, "ui", 1.0)
    ok, _ = f.request(mode, "keyboard", 2.0)
    assert ok and f.mode == STAND and f.source == "keyboard"


@pytest.mark.parametrize("src", ["keyboard", "safety", "ui"])
def test_auto_to_damp_any_source(src):
    f = standing()
    f.request(AUTO, "mission", 1.0)
    ok, _ = f.request(DAMP, src, 2.0)
    assert ok and f.mode == DAMP


def test_forward_table_auto_inputs_only_in_auto():
    for ch in ("upper_auto", "hand_auto"):
        assert FORWARD[ch] == {AUTO}
        for m in ALL:
            assert forwards(m, ch) == (m == AUTO)


def test_forward_table_teleop_inputs_ignored_in_auto():
    for ch in ("key_vel", "cmd_vel_teleop", "upper_teleop", "hand_teleop"):
        assert not forwards(AUTO, ch), ch
    assert forwards(AUTO, "cmd_vel_nav")      # Nav2 drives the legs in AUTO (contract section 6)


def test_forward_table_unchanged_for_teleop_modes():
    assert forwards(LOCO, "key_vel") and forwards(LOCO, "cmd_vel_teleop")
    assert forwards(MANIP, "upper_teleop") and forwards(MANIP, "hand_teleop")
    assert forwards(NAV, "cmd_vel_nav")
    for ch in FORWARD:
        assert not forwards(DAMP, ch) and not forwards(STAND, ch), ch
