"""Control lease, watchdog, hardware e-stop interlock, motion gate and speed zones (safety.py). Pure python."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from g1_teleop_manager.fsm import DAMP, STAND, LOCO, MANIP, NAV, AUTO, HOLD  # noqa: E402
from g1_teleop_manager.safety import (EstopMonitor, Lease, Watchdog, clamp_speed, gate, zone_at,  # noqa: E402
                                      point_in_polygon)


def ok_estop(now=0.0):
    e = EstopMonitor(timeout=0.5)
    e.update(True, now)
    return e


# ---------------------------------------------------------------- lease
def test_lease_acquire_release():
    L = Lease(ttl=30)
    assert L.acquire("ui:a", "operator", "iPad A", "", 0.0)[0]
    assert L.acquire("ui:a", "operator", "iPad A", "", 1.0) == (True, "already yours")
    ok, msg = L.acquire("ui:b", "operator", "iPad B", "", 2.0)
    assert not ok and "iPad A" in msg
    assert not L.release("ui:b", 3.0)[0]
    assert L.release("ui:a", 3.0)[0] and L.free(3.0)


def test_lease_steal_needs_reason_and_reports_previous():
    L = Lease()
    L.acquire("ui:a", "operator", "iPad A", "", 0.0)
    ok, msg, prev = L.steal("ui:b", "technician", "iPad B", "  ", 1.0)
    assert not ok and L.holder == "ui:a"
    ok, msg, prev = L.steal("ui:b", "technician", "iPad B", "A left", 1.0)
    assert ok and L.holder == "ui:b" and prev == ("ui:a", "iPad A") and "A left" in msg
    assert any(k == "steal" for k, _ in L.events)


def test_lease_expires_without_renew():
    L = Lease(ttl=5)
    L.acquire("ui:a", "operator", "A", "", 0.0)
    assert L.renew("ui:a", 4.0) and not L.renew("ui:b", 4.0)
    assert L.held_by("ui:a", 8.9)
    assert L.free(9.1) and any(k == "expired" for k, _ in L.events)


# ---------------------------------------------------------------- watchdog
def test_watchdog_trips_only_in_manual_modes_for_ui_holders():
    L, W = Lease(), Watchdog(timeout=1.0)
    L.acquire("ui:a", "operator", "A", "", 0.0)
    W.beat("ui:a", 0.0)
    assert not W.tripped(LOCO, L, 0.9)
    assert W.tripped(LOCO, L, 1.1) and W.tripped(MANIP, L, 1.1)
    for m in (STAND, NAV, AUTO, HOLD, DAMP):
        assert not W.tripped(m, L, 5.0)
    W.beat("ui:a", 1.05)
    assert not W.tripped(LOCO, L, 1.5)


def test_watchdog_ignores_keyboard_and_free_lease():
    L, W = Lease(), Watchdog(timeout=1.0)
    assert not W.tripped(LOCO, L, 100.0)
    L.acquire("keyboard", "local", "keyboard", "", 0.0, implicit=True)
    assert not W.tripped(LOCO, L, 100.0)


def test_watchdog_grace_from_acquire_time():
    L, W = Lease(), Watchdog(timeout=1.0)
    L.acquire("ui:a", "operator", "A", "", 10.0)
    assert not W.tripped(LOCO, L, 10.5) and W.tripped(LOCO, L, 11.1)


# ---------------------------------------------------------------- e-stop
def test_estop_missing_locks_motion():
    e = EstopMonitor(timeout=0.5)
    assert not e.ok(0.0) and "not seen" in e.reason(0.0)
    e.update(True, 1.0)
    assert e.ok(1.2) and not e.ok(1.6) and "lost" in e.reason(1.6)


def test_estop_press_release_counts():
    e = ok_estop()
    assert e.update(False, 0.1) is True and not e.ok(0.1) and "pressed" in e.reason(0.1)
    assert e.update(False, 0.2) is False
    e.update(True, 0.3)
    assert e.ok(0.3) and e.press_count == 1


def test_estop_dev_override():
    e = EstopMonitor(required=False)
    assert e.ok(0.0) and e.as_dict(0.0)["dev_override"]


# ---------------------------------------------------------------- gate
@pytest.mark.parametrize("target", [LOCO, MANIP, NAV, AUTO])
def test_gate_locks_motion_without_estop(target):
    ok, msg = gate(target, "mission", STAND, Lease(), EstopMonitor(), 0.0)
    assert not ok and "motion locked" in msg


@pytest.mark.parametrize("target", [HOLD, DAMP, STAND])
def test_gate_stop_power_off_stand_always_pass(target):
    L = Lease()
    L.acquire("ui:a", "operator", "A", "", 0.0)
    assert gate(target, "ui:b", LOCO, L, EstopMonitor(), 0.0)[0]


@pytest.mark.parametrize("target", [LOCO, MANIP, NAV])
def test_gate_non_holder_rejected(target):
    L = Lease()
    L.acquire("ui:a", "operator", "iPad A", "", 0.0)
    ok, msg = gate(target, "ui:b", STAND, L, ok_estop(), 0.0)
    assert not ok and "iPad A" in msg
    assert gate(target, "ui:a", STAND, L, ok_estop(), 0.0)[0]
    ok, _ = gate(target, "keyboard", STAND, L, ok_estop(), 0.0)
    assert not ok


def test_gate_mission_not_blocked_by_lease():
    L = Lease()
    L.acquire("ui:a", "operator", "A", "", 0.0)
    assert gate(AUTO, "mission", STAND, L, ok_estop(), 0.0)[0]


def test_gate_resume_blocked_while_estop_pressed():
    e = ok_estop()
    e.update(False, 0.1)
    ok, msg = gate(STAND, "ui:a", HOLD, Lease(), e, 0.1)
    assert not ok and "release" in msg
    e.update(True, 0.2)
    assert gate(STAND, "ui:a", HOLD, Lease(), e, 0.2)[0]


# ---------------------------------------------------------------- zones
SQ = [(0, 0), (2, 0), (2, 2), (0, 2)]


def test_point_in_polygon():
    assert point_in_polygon(1, 1, SQ) and not point_in_polygon(3, 1, SQ) and not point_in_polygon(1, -0.1, SQ)


def test_zone_first_match_and_clamp():
    zones = [{"name": "theatre", "max_speed": 0.3, "polygon": SQ},
             {"name": "big", "max_speed": 0.6, "polygon": [(-5, -5), (5, -5), (5, 5), (-5, 5)]}]
    assert zone_at(1, 1, zones)["name"] == "theatre"
    assert zone_at(4, 4, zones)["name"] == "big"
    assert zone_at(9, 9, zones) is None
    vx, vy, wz = clamp_speed((0.8, 0.6, 0.5), 0.5)
    assert (vx, vy) == pytest.approx((0.4, 0.3)) and wz == 0.5
    assert clamp_speed((0.2, 0.0, 0.1), 0.5) == (0.2, 0.0, 0.1)
    assert clamp_speed((0.2, 0.1, 0.1), None) == (0.2, 0.1, 0.1)
