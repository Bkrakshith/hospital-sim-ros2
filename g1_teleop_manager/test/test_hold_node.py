"""Manager node: HOLD routing, lease service, UI heartbeat watchdog, e-stop interlock, key identity, zones.
Callbacks are called directly, publishers captured; an isolated DDS domain (no graph traffic reaches a stack)."""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
rclpy = pytest.importorskip("rclpy")
pytest.importorskip("g1_interfaces.msg")
ControlLease = pytest.importorskip("g1_interfaces.srv").ControlLease
from sensor_msgs.msg import JointState  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from std_msgs.msg import Bool, String  # noqa: E402
from g1_teleop_manager.fsm import DAMP, STAND, LOCO, MANIP, NAV, AUTO, HOLD  # noqa: E402


class Cap:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


@pytest.fixture(scope="module")
def ros():
    rclpy.init(domain_id=int(os.environ.get("G1_TEST_DOMAIN_ID", "172")))
    yield
    rclpy.shutdown()


@pytest.fixture
def node(ros):
    from g1_teleop_manager.manager_node import ManagerNode
    n = ManagerNode()
    n.pub_upper, n.pub_cmd, n.pub_mode, n.pub_status, n.pub_lease = Cap(), Cap(), Cap(), Cap(), Cap()
    n.pub_hand = {"left": Cap(), "right": Cap()}
    n.estop_beat = lambda ok=True: n.on_estop(Bool(data=ok))
    n.estop_beat()
    yield n
    n.destroy_node()


def lease(n, op, holder, reason="", role="operator"):
    req = ControlLease.Request(op=op, holder=holder, role=role, display=holder.upper(), reason=reason)
    return n.on_lease(req, ControlLease.Response())


def js(vals, names=None):
    m = JointState()
    m.position = [float(v) for v in vals]
    m.name = names or []
    return m


def twist(vx):
    t = Twist()
    t.linear.x = vx
    return t


def key_ui(n, k, holder):
    n.on_key_ui(String(data=json.dumps({"key": k, "holder": holder})))


def test_o_key_stops_from_loco_and_zeroes_velocity(node):
    node.request(STAND, "keyboard")
    node.request(LOCO, "keyboard")
    for _ in range(3):
        node.on_key(String(data="w"))
    node.tick()
    assert node.pub_cmd.msgs[-1].linear.x > 0
    node.on_key(String(data="o"))
    assert node.fsm.mode == HOLD and node.fsm.hold_from == LOCO
    node.on_twist_teleop(twist(0.5))
    node.on_key(String(data="w"))
    node.tick()
    assert node.pub_cmd.msgs[-1].linear.x == 0.0


def test_hold_in_auto_drops_mission_arm_and_hand_targets(node):
    node.request(STAND, "keyboard")
    node.request(AUTO, "mission")
    node.on_upper_auto(js([0.3] * 17))
    node.request(HOLD, "ui:ipad-1")
    n_up, n_hand = len(node.pub_upper.msgs), len(node.pub_hand["right"].msgs)
    node.on_upper_auto(js([0.9] * 17))
    node.on_hand_auto("right", js([1.0] * 6))
    node.on_twist_nav(twist(0.3))
    for _ in range(5):
        node.tick()
    assert len(node.pub_upper.msgs) == n_up          # arms frozen: nothing streamed (controller holds)
    assert len(node.pub_hand["right"].msgs) == n_hand  # hands keep their last command
    assert node.pub_cmd.msgs[-1].linear.x == 0.0
    assert node.hold_info["from"] == "AUTO"


def test_resume_from_hold_blends_from_measured_pose(node):
    node.request(STAND, "keyboard")
    node.request(MANIP, "keyboard")
    node.request(HOLD, "keyboard")
    node.on_joint_states(js([0.5] * 17, [n for n in __import__(
        "g1_teleop_manager.manager_node", fromlist=["UPPER_NAMES"]).UPPER_NAMES]))
    ok, _ = node.request(STAND, "ui:ipad-1")
    assert ok and node.fsm.blend_from == pytest.approx([0.5] * 17) and node.fsm.cur_blend_time >= 2.0
    node.tick()
    first = list(node.pub_upper.msgs[-1].position)
    assert first == pytest.approx([0.5] * 17, abs=0.01)   # starts where the arms are, no jump


def test_watchdog_trips_hold_when_holder_heartbeat_stops(node):
    node.request(STAND, "keyboard")
    assert lease(node, "acquire", "ui:ipad-1").ok
    node.on_ui_heartbeat(String(data=json.dumps({"holder": "ui:ipad-1", "seq": 1})))
    assert node.request(LOCO, "ui:ipad-1")[0]
    node.tick()
    assert node.fsm.mode == LOCO
    node.watchdog.last["ui:ipad-1"] = time.monotonic() - 1.2     # 1.2 s without heartbeat
    node.lease.since -= 5.0                                       # (the lease is older than that)
    node.tick()
    assert node.fsm.mode == HOLD and node.fsm.source == "watchdog"


def test_watchdog_quiet_with_heartbeat_and_in_nav(node):
    node.request(STAND, "keyboard")
    lease(node, "acquire", "ui:ipad-1")
    node.request(LOCO, "ui:ipad-1")
    for _ in range(3):
        node.on_ui_heartbeat(String(data=json.dumps({"holder": "ui:ipad-1"})))
        node.tick()
    assert node.fsm.mode == LOCO
    node.request(STAND, "ui:ipad-1")
    node.request(NAV, "ui:ipad-1")
    node.watchdog.last["ui:ipad-1"] = time.monotonic() - 5
    node.lease.since -= 10.0
    node.tick()
    assert node.fsm.mode == NAV


def test_non_holder_motion_rejected(node):
    node.request(STAND, "keyboard")
    lease(node, "acquire", "ui:ipad-1")
    ok, msg = node.request(LOCO, "ui:ipad-2")
    assert not ok and node.fsm.mode == STAND and "IPAD-1" in msg.upper()
    node.on_key(String(data="l"))                  # terminal keyboard is not the holder either
    assert node.fsm.mode == STAND
    assert node.request(HOLD, "ui:ipad-2")[0]      # but anyone can stop


def test_ui_velocity_keys_only_from_holder(node):
    node.request(STAND, "keyboard")
    lease(node, "acquire", "ui:ipad-1")
    node.on_ui_heartbeat(String(data="ui:ipad-1"))
    key_ui(node, "l", "ui:ipad-1")
    assert node.fsm.mode == LOCO
    key_ui(node, "w", "ui:ipad-2")
    node.on_key(String(data="w"))
    node.tick()
    assert node.pub_cmd.msgs[-1].linear.x == 0.0
    key_ui(node, "w", "ui:ipad-1")
    node.tick()
    assert node.pub_cmd.msgs[-1].linear.x > 0


def test_steal_in_manual_mode_holds(node):
    node.request(STAND, "keyboard")
    lease(node, "acquire", "ui:ipad-1")
    node.on_ui_heartbeat(String(data="ui:ipad-1"))
    node.request(LOCO, "ui:ipad-1")
    r = lease(node, "steal", "ui:ipad-2", reason="")
    assert not r.ok
    r = lease(node, "steal", "ui:ipad-2", reason="operator left")
    assert r.ok and r.holder == "ui:ipad-2" and node.fsm.mode == HOLD and node.fsm.source == "lease"


def test_implicit_keyboard_lease_released_after_manual(node):
    node.request(STAND, "keyboard")
    node.on_key(String(data="l"))
    assert node.fsm.mode == LOCO and node.lease.holder == "keyboard" and node.lease.implicit
    node.on_key(String(data="p"))
    assert node.fsm.mode == STAND and node.lease.holder == ""


def test_estop_missing_locks_and_press_holds(node):
    node.request(STAND, "keyboard")
    node.estop.last = time.monotonic() - 2.0         # heartbeat lost
    ok, msg = node.request(LOCO, "keyboard")
    assert not ok and "e-stop" in msg
    node.estop_beat(True)
    assert node.request(NAV, "keyboard")[0]
    node.estop_beat(False)                           # pressed
    assert node.fsm.mode == HOLD and node.fsm.source == "estop_hw"
    ok, msg = node.request(STAND, "ui:ipad-1")
    assert not ok and "release" in msg
    node.estop_beat(True)
    assert node.request(STAND, "ui:ipad-1")[0]


def test_estop_heartbeat_lost_while_moving_holds(node):
    node.request(STAND, "keyboard")
    node.request(NAV, "keyboard")
    node.estop.last = time.monotonic() - 1.0
    node.tick()
    assert node.fsm.mode == HOLD and node.fsm.source == "estop_hw"


def test_zone_clamps_nav_speed(node):
    node.request(STAND, "keyboard")
    node.request(NAV, "keyboard")
    node.zone = {"name": "theatre", "max_speed": 0.2}
    node.on_twist_nav(twist(0.35))
    node.tick()
    assert node.pub_cmd.msgs[-1].linear.x == pytest.approx(0.2)


def test_status_has_safety_fields(node):
    node.heartbeat()
    st = json.loads(node.pub_status.msgs[-1].data)
    for k in ("hold", "lease", "estop", "watchdog", "motion_locked", "zone"):
        assert k in st
    assert st["estop"]["ok"] and not st["estop"]["dev_override"]


def test_damp_to_stand_allowed_without_estop(node):
    node.estop.last = None
    assert node.request(STAND, "keyboard")[0]
    assert node.request(HOLD, "keyboard")[0] and node.request(DAMP, "keyboard")[0]


def test_arms_hold_stops_teleop_forwarding_and_tells_controller(node):
    node.pub_arms_hold = Cap()
    node.request(STAND, "keyboard")
    lease(node, "acquire", "ui:ipad-1")
    node.on_ui_heartbeat(String(data="ui:ipad-1"))
    node.request(MANIP, "ui:ipad-1")
    node.on_upper_raw(js([0.2] * 17))
    n = len(node.pub_upper.msgs)
    node.on_arms_hold(String(data=json.dumps({"on": True, "holder": "ui:ipad-2"})))   # not the holder
    assert not node.fsm.arms_hold
    node.on_arms_hold(String(data=json.dumps({"on": True, "holder": "ui:ipad-1"})))
    assert node.fsm.arms_hold and node.fsm.mode == MANIP and node.pub_arms_hold.msgs[-1].data is True
    node.on_upper_raw(js([0.6] * 17))
    node.on_hand_raw("left", js([1.0] * 6))
    assert len(node.pub_upper.msgs) == n and node.pub_hand["left"].msgs == []
    node.on_arms_hold(String(data=json.dumps({"on": False, "holder": "ui:ipad-1"})))
    node.on_upper_raw(js([0.6] * 17))
    assert len(node.pub_upper.msgs) == n + 1 and node.pub_arms_hold.msgs[-1].data is False


def test_hand_override_rules(node):
    node.pub_arms_hold = Cap()
    node.request(STAND, "keyboard")
    ho = lambda side, c, who: node.on_hand_override(String(data=json.dumps(  # noqa: E731
        {"side": side, "close": c, "holder": who})))
    ho("right", 0.0, "ui:ipad-1")
    assert node.pub_hand["right"].msgs == [] and "refused" in node.last_msg          # STAND: refused
    node.request(HOLD, "ui:ipad-1")
    ho("right", 0.0, "ui:ipad-1")
    assert list(node.pub_hand["right"].msgs[-1].position) == [0.0] * 6
    assert node.last_override["side"] == "right"
    lease(node, "acquire", "ui:ipad-2")
    ho("left", 1.0, "ui:ipad-1")                                                     # not the lease holder
    assert node.pub_hand["left"].msgs == []
    ho("left", 1.5, "ui:ipad-2")
    assert list(node.pub_hand["left"].msgs[-1].position) == [1.0] * 6               # clamped
