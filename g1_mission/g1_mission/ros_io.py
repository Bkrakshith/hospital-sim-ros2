"""ROS implementations of the skill interfaces (interfaces.py) for the mission node."""
import math
import time

import numpy as np
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

from g1_mission.geometry import make_T, quat_to_mat
from g1_mission.interfaces import RobotInterface
from g1_mission.kinematics import UPPER_NAMES

HAND_ORDER = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]


class RosRobot(RobotInterface):
    """Proprioception from /joint_states + /g1/hand/*/state, base pose from /sim/ground_truth (CHEAT; 'tf' = map ->
    pelvis from AMCL), commands on the AUTO channels: /g1/auto/upper_body_target, /g1/auto/hand/{side}/cmd and
    /g1/cmd_vel_nav (FineAlign only, while Nav2 is idle)."""

    def __init__(self, node, base_source="ground_truth", tf_buffer=None, image=False):
        self.node = node
        self.base_source = base_source
        self.tf = tf_buffer
        self.q = None
        self.hands = {"left": None, "right": None}
        self.gt = None
        self.img = None
        self.img_t = 0.0
        self.commanded = False
        self.last = {"upper": None, "left": None, "right": None}
        self.idx = {n: i for i, n in enumerate(UPPER_NAMES)}
        node.create_subscription(JointState, "/joint_states", self._on_js, 20)
        for s in ("left", "right"):
            node.create_subscription(JointState, f"/g1/hand/{s}/state", lambda m, s=s: self._on_hand(s, m), 10)
        node.create_subscription(Odometry, "/sim/ground_truth", self._on_gt, 10)
        self.pub_upper = node.create_publisher(JointState, "/g1/auto/upper_body_target", 10)
        self.pub_hand = {s: node.create_publisher(JointState, f"/g1/auto/hand/{s}/cmd", 10) for s in ("left", "right")}
        self.pub_vel = node.create_publisher(Twist, "/g1/cmd_vel_nav", 10)
        self.image_sub = None
        if image:
            self.enable_image(True)

    def enable_image(self, on):
        if on and self.image_sub is None:
            self.image_sub = self.node.create_subscription(Image, "/g1/head_camera/image_raw", self._on_img, 2)
        elif not on and self.image_sub is not None:
            self.node.destroy_subscription(self.image_sub)
            self.image_sub, self.img = None, None

    # ------------------------------------------------------------------ callbacks
    def _on_js(self, m):
        q = np.full(17, np.nan) if self.q is None else self.q.copy()
        for n, v in zip(m.name, m.position):
            i = self.idx.get(n)
            if i is not None:
                q[i] = v
        if not np.isnan(q).any():
            self.q = q

    def _on_hand(self, side, m):
        if len(m.position) >= 6:
            self.hands[side] = np.array(m.position[:6], float)

    def _on_gt(self, m):
        p, o = m.pose.pose.position, m.pose.pose.orientation
        self.gt = make_T(quat_to_mat([o.w, o.x, o.y, o.z]), [p.x, p.y, p.z])

    def _on_img(self, m):
        if m.encoding in ("rgb8", "bgr8") and m.height and m.width:
            a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            self.img = a if m.encoding == "rgb8" else a[:, :, ::-1]
            self.img_t = time.time()

    # ------------------------------------------------------------------ RobotInterface
    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def upper_q(self):
        return None if self.q is None else self.q.copy()

    def hand_q(self, side):
        h = self.hands[side]
        return None if h is None else h.copy()

    def base_T(self):
        if self.base_source == "tf" and self.tf is not None:
            try:
                from rclpy.time import Time
                t = self.tf.lookup_transform("map", "pelvis", Time())
                tr, r = t.transform.translation, t.transform.rotation
                return make_T(quat_to_mat([r.w, r.x, r.y, r.z]), [tr.x, tr.y, tr.z])
            except Exception:  # noqa: BLE001
                return None
        return None if self.gt is None else self.gt.copy()

    def head_image(self):
        return self.img

    def command(self, upper=None, left_hand=None, right_hand=None, cmd_vel=None):
        stamp = self.node.get_clock().now().to_msg()
        if upper is not None:
            js = JointState()
            js.header.stamp = stamp
            js.name = list(UPPER_NAMES)
            js.position = [float(v) for v in upper]
            self.pub_upper.publish(js)
            self.last["upper"] = list(js.position)
            self.commanded = True
        for side, h in (("left", left_hand), ("right", right_hand)):
            if h is not None:
                js = JointState()
                js.header.stamp = stamp
                js.name = [f"{side}_hand_{k}" for k in HAND_ORDER]
                js.position = [float(min(1.0, max(0.0, v))) for v in h]
                self.pub_hand[side].publish(js)
                self.last[side] = list(js.position)
        if cmd_vel is not None:
            t = Twist()
            t.linear.x, t.linear.y, t.angular.z = (float(v) for v in cmd_vel)
            self.pub_vel.publish(t)


class NavClient:
    """nav2_msgs/NavigateToPose on /navigate_to_pose (map frame)."""

    def __init__(self, node, action="/navigate_to_pose"):
        from nav2_msgs.action import NavigateToPose
        self.node = node
        self.Goal = NavigateToPose.Goal
        self.client = ActionClient(node, NavigateToPose, action)
        self.state = "idle"          # idle | pending | active | succeeded | failed | canceled | rejected
        self.handle = None
        self.feedback = None
        self.goal_pose = None
        self.msg = ""

    def available(self):
        return self.client.server_is_ready()

    def go(self, x, y, yaw):
        self.goal_pose = (x, y, yaw)
        g = self.Goal()
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.node.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        g.pose = ps
        self.state, self.feedback, self.handle, self.msg = "pending", None, None, ""
        fut = self.client.send_goal_async(g, feedback_callback=self._fb)
        fut.add_done_callback(self._accepted)

    def _fb(self, fb):
        self.feedback = fb.feedback

    def _accepted(self, fut):
        h = fut.result()
        if not h.accepted:
            self.state, self.msg = "rejected", "Nav2 rejected the goal"
            return
        if self.state == "canceled":          # cancelled while pending
            h.cancel_goal_async()
            return
        self.handle, self.state = h, "active"
        h.get_result_async().add_done_callback(self._done)

    def _done(self, fut):
        st = fut.result().status
        if st == GoalStatus.STATUS_SUCCEEDED:
            self.state = "succeeded"
        elif st == GoalStatus.STATUS_CANCELED:
            self.state = "canceled" if self.state != "idle" else "idle"
        else:
            self.state, self.msg = "failed", f"Nav2 goal ended with status {st}"

    def cancel(self):
        if self.handle is not None and self.state == "active":
            self.handle.cancel_goal_async()
        self.state = "canceled"

    def distance_remaining(self):
        return None if self.feedback is None else float(self.feedback.distance_remaining)


class GlideNav:
    """FIXED-BASE CHEAT with NavClient's interface: /sim/fixed_base_goal makes the sim glide the welded pelvis to the
    goal (straight line, no obstacle checks). Succeeds when the sim reports the glide finished and the pelvis ground
    truth is within 3 cm / 0.05 rad of the goal."""

    def __init__(self, node, robot):
        import json as _json
        self._json = _json
        self.node, self.robot = node, robot
        self.pub = node.create_publisher(PoseStamped, "/sim/fixed_base_goal", 5)
        self.sim = {}
        node.create_subscription(String, "/sim/status", self._on_status, 5)
        self.state, self.msg, self.goal_pose, self.t_sent = "idle", "", None, 0.0

    def _on_status(self, m):
        try:
            self.sim = self._json.loads(m.data)
        except ValueError:
            pass

    def fixed_base(self):
        return bool(self.sim.get("fixed_base"))

    def available(self):
        return self.fixed_base()

    def go(self, x, y, yaw):
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.pose.position.x, ps.pose.position.y = float(x), float(y)
        ps.pose.orientation.z, ps.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.pub.publish(ps)
        T = self.robot.base_T()
        dist = 0.0 if T is None else math.hypot(x - T[0, 3], y - T[1, 3])
        self.budget = 6.0 + 2.0 * max(dist / 0.3, math.pi / 0.6)     # generous: sim glide 0.3 m/s, 0.6 rad/s
        self.goal_pose, self.state, self.t_sent, self.msg = (x, y, yaw), "active", time.time(), ""

    def cancel(self):
        self.state = "canceled"

    def distance_remaining(self):
        """Arrival from the pelvis ground truth (the sim's 1 Hz /sim/status is too coarse to time a glide)."""
        T = self.robot.base_T()
        if T is None or self.goal_pose is None:
            return None
        d = math.hypot(self.goal_pose[0] - T[0, 3], self.goal_pose[1] - T[1, 3])
        yaw = math.atan2(T[1, 0], T[0, 0])
        dyaw = abs((self.goal_pose[2] - yaw + math.pi) % (2 * math.pi) - math.pi)
        if self.state == "active":
            if d < 0.03 and dyaw < 0.05 and time.time() - self.t_sent > 0.5:
                self.state = "succeeded"
            elif time.time() - self.t_sent > self.budget:
                self.state = "failed"
                self.msg = f"glide ended {d * 100:.1f} cm / {math.degrees(dyaw):.1f} deg from the goal"
        return d


class AutoNav:
    """Nav2 normally; the fixed-base glide when the sim reports fixed_base (arm-skill development)."""

    def __init__(self, nav2, glide):
        self.nav2, self.glide = nav2, glide
        self.cur = None

    def _pick(self):
        return self.glide if (self.glide is not None and self.glide.fixed_base()) else self.nav2

    def available(self):
        n = self._pick()
        return n is not None and n.available()

    def go(self, x, y, yaw):
        self.cur = self._pick()
        self.cur.go(x, y, yaw)

    def exact(self):
        """True when the goal is reached exactly (glide); Nav2 stops within its goal tolerance."""
        return self._pick() is self.glide

    @property
    def state(self):
        if self.cur is None:
            return "idle"
        if self.cur is self.glide:
            self.glide.distance_remaining()          # evaluates arrival
        return self.cur.state

    @property
    def msg(self):
        return self.cur.msg if self.cur is not None else ""

    def cancel(self):
        if self.cur is not None:
            self.cur.cancel()

    def distance_remaining(self):
        return self.cur.distance_remaining() if self.cur is not None else None
