"""Manager node routing in AUTO vs other modes (callbacks called directly, publishers captured; no graph traffic).
Skipped when rclpy / g1_interfaces are not importable (e.g. pure-python runs without the overlay)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
rclpy = pytest.importorskip("rclpy")
pytest.importorskip("g1_interfaces.msg")
from sensor_msgs.msg import JointState  # noqa: E402
from geometry_msgs.msg import Twist  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from g1_teleop_manager.fsm import STAND, LOCO, MANIP, NAV, AUTO  # noqa: E402


class Cap:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


@pytest.fixture(scope="module")
def ros():
    # an isolated DDS domain: the node publishes /g1/mode on construction, which must never reach a running stack
    rclpy.init(domain_id=int(os.environ.get("G1_TEST_DOMAIN_ID", "171")))
    yield
    rclpy.shutdown()


@pytest.fixture
def node(ros):
    from g1_teleop_manager.manager_node import ManagerNode
    n = ManagerNode()
    n.estop.required = False     # no hardware e-stop heartbeat in unit tests (see test_hold_node.py for the lock)
    n.pub_upper, n.pub_cmd = Cap(), Cap()
    n.pub_hand = {"left": Cap(), "right": Cap()}
    n.pub_mode = Cap()
    yield n
    n.destroy_node()


def goto(n, mode, source="mission"):
    n.request(STAND, "keyboard")
    if mode != STAND:
        assert n.request(mode, source if mode == AUTO else "keyboard")[0]


def js(vals, names=None):
    m = JointState()
    m.position = [float(v) for v in vals]
    if names:
        m.name = names
    return m


def twist(vx):
    t = Twist()
    t.linear.x = vx
    return t


@pytest.mark.parametrize("mode", [STAND, LOCO, MANIP, NAV])
def test_auto_inputs_ignored_outside_auto(node, mode):
    goto(node, mode)
    node.on_upper_auto(js([0.2] * 17))
    node.on_hand_auto("left", js([1.0] * 6))
    assert node.pub_upper.msgs == [] and node.pub_hand["left"].msgs == []


def test_auto_inputs_forwarded_in_auto(node):
    goto(node, AUTO)
    node.on_upper_auto(js([0.2] * 17))
    node.on_hand_auto("right", js([0.5] * 6))
    assert len(node.pub_upper.msgs) == 1 and list(node.pub_upper.msgs[0].position) == pytest.approx([0.2] * 17)
    assert list(node.pub_hand["right"].msgs[0].position) == pytest.approx([0.5] * 6)


def test_teleop_inputs_ignored_in_auto(node):
    goto(node, AUTO)
    node.on_upper_raw(js([0.4] * 17))
    node.on_hand_raw("left", js([1.0] * 6))
    node.on_twist_teleop(twist(0.5))
    for k in ("w", "w", "a"):
        node.on_key(String(data=k))
    node.tick()
    assert node.pub_upper.msgs == [] and node.pub_hand["left"].msgs == []
    assert node.pub_cmd.msgs[-1].linear.x == 0.0 and node.pub_cmd.msgs[-1].linear.y == 0.0


def test_nav_cmd_vel_forwarded_in_auto(node):
    goto(node, AUTO)
    node.on_twist_nav(twist(0.3))
    node.tick()
    assert node.pub_cmd.msgs[-1].linear.x == pytest.approx(0.3)


def test_p_key_leaves_auto_and_blends(node):
    goto(node, AUTO)
    node.on_upper_auto(js([0.2] * 17))
    node.on_key(String(data="p"))
    assert node.fsm.mode == STAND and node.fsm.source == "keyboard"
    assert node.fsm.detail.startswith("blending") and node.fsm.blend_from == pytest.approx([0.2] * 17)
    node.on_upper_auto(js([0.9] * 17))    # mission still streaming: ignored now
    n = len(node.pub_upper.msgs)
    node.tick()                          # blend stream towards default
    assert len(node.pub_upper.msgs) == n + 1
    assert all(abs(v) <= 0.2 + 1e-9 for v in node.pub_upper.msgs[-1].position)


def test_x_key_estop_from_auto(node):
    goto(node, AUTO)
    node.on_key(String(data="x"))
    assert node.fsm.mode == 0
