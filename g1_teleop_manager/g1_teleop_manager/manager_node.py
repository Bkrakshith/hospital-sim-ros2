"""G1 teleop manager: mode FSM + command arbiter (docs/interfaces.md 3.2-3.5). The only node that decides who drives
the robot, and the authority for the control lease, the UI heartbeat watchdog and the hardware e-stop interlock.

In:  /g1/teleop/key (String, identity "keyboard"), /g1/teleop/key_ui (String JSON {"key","holder"}),
     /g1/cmd_vel_teleop (Twist), /g1/cmd_vel_nav (Twist),
     /g1/teleop/upper_body_target_raw (JointState), /g1/teleop/hand/{left,right}/cmd_raw (JointState),
     /g1/teleop/tracking_status (String JSON, informational), service /g1/set_mode (SetMode),
     /g1/auto/upper_body_target (JointState), /g1/auto/hand/{left,right}/cmd (JointState)  [mission/skills]
     /joint_states (measured arm pose: blend start when a stop is resumed), /g1/estop/heartbeat (Bool, 10 Hz),
     /g1/ui/heartbeat (String JSON {"holder", "seq"}), service /g1/control/lease (ControlLease), TF map->pelvis (zones)
     /g1/teleop/arms_hold (String JSON {"on", "holder"}), /g1/teleop/hand_override (String JSON {"side", "close",
     "holder"})
Out: /g1/mode (Mode, transient-local), /g1/cmd_vel (Twist, 50 Hz; zero unless LOCO / NAV / AUTO; zone-capped),
     /g1/upper_body_target (JointState; only in MANIP / AUTO, plus the blend to default after leaving them / HOLD),
     /g1/hand/{left,right}/cmd (JointState; only in MANIP / AUTO), /g1/teleop/manager_status (String JSON, 1 Hz and on
     change), /g1/control/lease_state (String JSON, transient-local), /g1/arms_hold (Bool, transient-local: the
     controller freezes the arms in MANIP while true)
Which input drives which output in which mode: fsm.FORWARD. Interlocks: safety.gate.
"""
import json
import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from g1_interfaces.msg import Mode
from g1_interfaces.srv import SetMode
try:
    from g1_interfaces.srv import ControlLease
except ImportError:  # g1_interfaces built before the lease existed
    ControlLease = None
from g1_teleop_manager.fsm import (DAMP, STAND, LOCO, MANIP, NAV, AUTO, HOLD, NAMES, KEY_TO_MODE, MANUAL,  # noqa: F401
                                   MOTION, ModeFSM, VelocityIntegrator, blend, clamp_twist, forwards)
from g1_teleop_manager.safety import EstopMonitor, Lease, Watchdog, clamp_speed, gate, identity, zone_at

UPPER_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
UPPER_IDX = {n: i for i, n in enumerate(UPPER_NAMES)}
UPPER_IDX.update({n[:-6]: i for i, n in enumerate(UPPER_NAMES)})


def config_file(name):
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("g1_teleop_manager"), "config", name)
    except Exception:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "config", name)


def default_keys_file():
    return config_file("keys.yaml")


class ManagerNode(Node):
    def __init__(self, **kw):
        super().__init__("g1_teleop_manager", **kw)
        self.declare_parameter("keys_file", "")
        self.declare_parameter("zones_file", "")
        self.declare_parameter("upper_body_default", [0.0] * 17)
        # DEV ONLY: run without a hardware e-stop heartbeat (motion is otherwise locked). Shown in manager_status and
        # on every UI as "dev_override". Never set on the real robot.
        self.declare_parameter("DEV_ONLY_ignore_estop_heartbeat", False)
        path = self.get_parameter("keys_file").value or default_keys_file()
        with open(path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg
        self.keymap = {str(k): v for k, v in cfg["keys"].items()}
        self.limits = cfg["limits"]
        self.default_upper = list(self.get_parameter("upper_body_default").value)
        self.fsm = ModeFSM(cfg.get("manip_blend_time", 1.0), cfg.get("hold_blend_time", 2.5), self.default_upper)
        self.vel = VelocityIntegrator(cfg["limits"], cfg["step"], cfg.get("key_timeout", 0.5))
        sc = cfg.get("safety", {})
        self.lease = Lease(ttl=sc.get("lease_ttl", 30.0))
        self.watchdog = Watchdog(timeout=sc.get("heartbeat_timeout", 1.0))
        dev = bool(self.get_parameter("DEV_ONLY_ignore_estop_heartbeat").value)
        self.estop = EstopMonitor(timeout=sc.get("estop_timeout", 0.5),
                                  required=bool(sc.get("estop_required", True)) and not dev)
        if not self.estop.required:
            self.get_logger().warning("DEV OVERRIDE: hardware e-stop heartbeat NOT required for motion")
        self.zones_cfg = self._load_zones()
        self.zone = None
        self.pose_map = None
        self.twist_teleop = None
        self.twist_teleop_t = -1e9
        self.twist_nav = None
        self.twist_nav_t = -1e9
        self.last_upper = None
        self.measured_upper = None
        self.measured_t = -1e9
        self.last_key = ""
        self.last_msg = ""
        self.tracking = None
        self.cmd = (0.0, 0.0, 0.0)
        self.hold_info = None          # {since, source, from, reason}
        self.events = []               # recent safety events for the status

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_mode = self.create_publisher(Mode, "/g1/mode", latched)
        self.pub_cmd = self.create_publisher(Twist, "/g1/cmd_vel", 10)
        self.pub_upper = self.create_publisher(JointState, "/g1/upper_body_target", 10)
        self.pub_hand = {s: self.create_publisher(JointState, f"/g1/hand/{s}/cmd", 10) for s in ("left", "right")}
        self.pub_status = self.create_publisher(String, "/g1/teleop/manager_status", 10)
        self.pub_lease = self.create_publisher(String, "/g1/control/lease_state", latched)
        self.pub_arms_hold = self.create_publisher(Bool, "/g1/arms_hold", latched)
        self.create_subscription(String, "/g1/teleop/arms_hold", self.on_arms_hold, 10)
        self.create_subscription(String, "/g1/teleop/hand_override", self.on_hand_override, 10)
        self.last_override = None
        self.create_subscription(String, "/g1/teleop/key", self.on_key, 50)
        self.create_subscription(String, "/g1/teleop/key_ui", self.on_key_ui, 50)
        self.create_subscription(Twist, "/g1/cmd_vel_teleop", self.on_twist_teleop, 10)
        self.create_subscription(Twist, "/g1/cmd_vel_nav", self.on_twist_nav, 10)
        self.create_subscription(JointState, "/g1/teleop/upper_body_target_raw", self.on_upper_raw, 10)
        for s in ("left", "right"):
            self.create_subscription(JointState, f"/g1/teleop/hand/{s}/cmd_raw",
                                     lambda m, s=s: self.on_hand_raw(s, m), 10)
        self.create_subscription(String, "/g1/teleop/tracking_status", self.on_tracking, 10)
        self.create_subscription(JointState, "/g1/auto/upper_body_target", self.on_upper_auto, 10)
        for s in ("left", "right"):
            self.create_subscription(JointState, f"/g1/auto/hand/{s}/cmd",
                                     lambda m, s=s: self.on_hand_auto(s, m), 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, 10)
        self.create_subscription(Bool, "/g1/estop/heartbeat", self.on_estop, 20)
        self.create_subscription(String, "/g1/ui/heartbeat", self.on_ui_heartbeat, 50)
        self.auto_t = -1e9
        self.create_service(SetMode, "/g1/set_mode", self.on_set_mode)
        if ControlLease is not None:
            self.create_service(ControlLease, "/g1/control/lease", self.on_lease)
        self.tf_buffer = None
        if self.zones_cfg["zones"]:
            try:
                import tf2_ros
                self.tf_buffer = tf2_ros.Buffer()
                self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
                self.create_timer(0.2, self.update_zone)
            except Exception as e:  # noqa: BLE001
                self.get_logger().warning(f"zones disabled (tf2_ros: {e})")
        self.create_timer(1.0 / cfg.get("publish_rate", 50.0), self.tick)
        self.create_timer(1.0, self.heartbeat)
        self.publish_mode()
        self.publish_lease()
        self._arms_hold_pub = None
        self.publish_arms_hold()
        self.get_logger().info(f"teleop manager up (keys {path}); mode DAMP; "
                               f"{len(self.zones_cfg['zones'])} speed zones")

    def _load_zones(self):
        path = self.get_parameter("zones_file").value or config_file("zones.yaml")
        try:
            with open(path) as f:
                z = yaml.safe_load(f) or {}
        except OSError:
            z = {}
        return {"frame": z.get("frame", "map"), "manual_cap": z.get("manual_cap"),
                "zones": [dict(name=q["name"], max_speed=float(q["max_speed"]),
                               polygon=[tuple(map(float, p)) for p in q["polygon"]]) for q in z.get("zones", [])]}

    # ------------------------------------------------------------------ mode requests
    def request(self, mode, source, detail=None):
        now = time.monotonic()
        prev = self.fsm.mode
        ok, msg = gate(mode, source, prev, self.lease, self.estop, now)
        if ok:
            start = self.measured_upper if (prev == HOLD and self.measured_upper is not None
                                            and now - self.measured_t < 1.0) else self.last_upper
            ok, msg = self.fsm.request(mode, source, now, upper_target=start, detail=detail)
        self.last_msg = msg
        if ok:
            self.get_logger().info(f"[{source}] {msg}")
        else:
            self.get_logger().warning(f"[{source}] rejected: {msg}")
        if ok:
            cur = self.fsm.mode
            if cur != LOCO:
                self.vel.zero()
            if cur == HOLD and prev != HOLD:
                self.twist_teleop = None
                self.hold_info = {"since": time.time(), "source": source, "from": NAMES[self.fsm.hold_from],
                                  "reason": detail or ""}
                self.event("hold", f"stopped by {source} (was {NAMES[prev]})" + (f": {detail}" if detail else ""))
            elif cur != HOLD:
                self.hold_info = None
            # implicit lease: entering manual motion with the lease free; dropped when manual motion ends
            if cur in MANUAL and prev not in MANUAL and self.lease.free(now):
                who = identity(source)
                self.lease.acquire(who, "local" if not who.startswith("ui:") else "", who, "", now, implicit=True)
                self.flush_lease_events()
            if cur not in MANUAL and prev in MANUAL and self.lease.holder and self.lease.implicit:
                self.lease.clear()
                self.publish_lease()
            if self.fsm.changed:
                self.publish_mode()
                self.heartbeat()
            self.publish_arms_hold()
        return ok, msg

    def publish_arms_hold(self):
        if self._arms_hold_pub != self.fsm.arms_hold:
            self._arms_hold_pub = self.fsm.arms_hold
            self.pub_arms_hold.publish(Bool(data=self.fsm.arms_hold))

    def _json_holder(self, msg):
        try:
            d = json.loads(msg.data)
            return d, str(d.get("holder") or "ui")
        except (ValueError, AttributeError):
            return None, None

    def on_arms_hold(self, msg):
        d, who = self._json_holder(msg)
        if d is None:
            return
        now = time.monotonic()
        if self.lease.held_by_other(who, now):
            self.last_msg = f"arms hold refused: control is held by {self.lease.display or self.lease.holder}"
            self.heartbeat()
            return
        ok, m = self.fsm.set_arms_hold(bool(d.get("on")))
        self.last_msg = m
        if ok:
            self.get_logger().info(f"[{who}] {m}")
            self.event("arms_hold", f"{m} by {who}")
            if self.fsm.changed:
                self.publish_mode()
            self.publish_arms_hold()
        self.heartbeat()

    def on_hand_override(self, msg):
        d, who = self._json_holder(msg)
        if d is None:
            return
        now = time.monotonic()
        side = d.get("side")
        if side not in ("left", "right"):
            return
        if self.lease.held_by_other(who, now):
            self.last_msg = f"gripper override refused: control is held by {self.lease.display or self.lease.holder}"
        elif not self.fsm.hand_override_allowed():
            self.last_msg = (f"gripper override refused in {NAMES[self.fsm.mode]}: only while stopped (HOLD) or with "
                             f"the arms held in MANIP")
        else:
            close = min(1.0, max(0.0, float(d.get("close", 0.0))))
            out = JointState()
            out.header.stamp = self.get_clock().now().to_msg()
            out.position = [close] * 6
            self.pub_hand[side].publish(out)
            self.last_override = {"side": side, "close": close, "by": who, "t": round(time.time(), 2)}
            self.last_msg = f"gripper override: {side} hand {'closed' if close > 0.5 else 'opened'} ({close:.2f})"
            self.event("hand_override", f"{self.last_msg} by {who}")
            self.get_logger().info(f"[{who}] {self.last_msg}")
        self.heartbeat()

    def on_set_mode(self, req, res):
        res.accepted, res.message = self.request(req.mode, req.source or "service")
        return res

    def event(self, kind, text):
        self.events = (self.events + [{"t": round(time.time(), 2), "kind": kind, "text": text}])[-20:]

    # ------------------------------------------------------------------ lease
    def on_lease(self, req, res):
        now = time.monotonic()
        op = (req.op or "query").strip().lower()
        who = req.holder.strip()
        if op == "acquire":
            ok, msg = self.lease.acquire(who, req.role, req.display, req.reason, now)
            if ok:
                self.watchdog.beat(who, now)
        elif op == "release":
            ok, msg = self.lease.release(who, now)
        elif op == "steal":
            prev_mode = self.fsm.mode
            ok, msg, prev = self.lease.steal(who, req.role, req.display, req.reason, now)
            if ok:
                self.watchdog.beat(who, now)
                if prev is not None and prev_mode in MANUAL:
                    self.request(HOLD, "lease", detail=f"control taken by {req.display or who}")
        elif op == "renew":
            ok = self.lease.renew(who, now)
            if ok:
                self.watchdog.beat(who, now)
            msg = "renewed" if ok else "not the holder"
        elif op == "query":
            self.lease.expire(now)
            ok, msg = True, ""
        else:
            ok, msg = False, f"unknown op '{req.op}' (acquire | release | steal | renew | query)"
        self.flush_lease_events()
        res.ok, res.message = ok, msg
        res.holder, res.role, res.display = self.lease.holder, self.lease.role, self.lease.display
        res.expires_in = float(self.lease.expires_in(now))
        if op != "query":
            self.get_logger().info(f"lease {op} by {who or '?'}: {'ok' if ok else 'refused'} {msg}")
        return res

    def flush_lease_events(self):
        if self.lease.events:
            for kind, text in self.lease.events:
                self.event("lease_" + kind, text)
            self.lease.events = []
            self.publish_lease()

    def publish_lease(self):
        d = self.lease.as_dict(time.monotonic())
        self.pub_lease.publish(String(data=json.dumps(d)))

    def on_ui_heartbeat(self, msg):
        try:
            d = json.loads(msg.data)
            who = str(d.get("holder", ""))
        except (ValueError, AttributeError):
            who = msg.data.strip()
        if not who:
            return
        now = time.monotonic()
        self.watchdog.beat(who, now)
        self.lease.renew(who, now)

    # ------------------------------------------------------------------ keys
    def on_key(self, msg):
        self.handle_key(msg.data, "keyboard")

    def on_key_ui(self, msg):
        try:
            d = json.loads(msg.data)
            self.handle_key(str(d.get("key", "")), str(d.get("holder") or "ui"))
        except (ValueError, AttributeError):
            pass

    def handle_key(self, raw, who):
        k = raw.strip().lower() if raw.strip() else "space"
        if k == " ":
            k = "space"
        self.last_key = k
        action = self.keymap.get(k)
        if action is None:
            return
        now = time.monotonic()
        if action in KEY_TO_MODE:
            mode, only_from = KEY_TO_MODE[action]
            if only_from is not None and self.fsm.mode not in only_from:
                self.last_msg = f"key '{k}' ({action}) ignored in {NAMES[self.fsm.mode]}"
                self.get_logger().warn(self.last_msg)
                return
            self.request(mode, who)
        elif forwards(self.fsm.mode, "key_vel"):
            if self.lease.held_by_other(who, now):
                self.last_msg = f"key '{k}' ignored: control is held by {self.lease.display or self.lease.holder}"
                return
            self.vel.key(action, now)
        else:
            self.last_msg = f"key '{k}' ({action}) ignored in {NAMES[self.fsm.mode]} (velocity keys need LOCO)"

    # ------------------------------------------------------------------ streams
    def on_twist_teleop(self, msg):
        self.twist_teleop = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.twist_teleop_t = time.monotonic()

    def on_twist_nav(self, msg):
        self.twist_nav = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.twist_nav_t = time.monotonic()

    def on_upper_raw(self, msg):
        if forwards(self.fsm.mode, "upper_teleop") and not self.fsm.arms_hold:
            self.forward_upper(msg)

    def on_upper_auto(self, msg):
        self.auto_t = time.monotonic()
        if forwards(self.fsm.mode, "upper_auto"):
            self.forward_upper(msg)

    def forward_upper(self, msg):
        tgt = list(self.last_upper or self.default_upper)
        if msg.name:
            for n, v in zip(msg.name, msg.position):
                i = UPPER_IDX.get(n)
                if i is not None:
                    tgt[i] = float(v)
        elif len(msg.position) == 17:
            tgt = [float(v) for v in msg.position]
        else:
            return
        self.last_upper = tgt
        self.publish_upper(tgt)

    def on_hand_raw(self, side, msg):
        if forwards(self.fsm.mode, "hand_teleop") and not self.fsm.arms_hold:
            self.forward_hand(side, msg)

    def on_hand_auto(self, side, msg):
        if forwards(self.fsm.mode, "hand_auto"):
            self.forward_hand(side, msg)

    def forward_hand(self, side, msg):
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = msg.name
        out.position = [min(1.0, max(0.0, float(v))) for v in msg.position]
        self.pub_hand[side].publish(out)

    def on_tracking(self, msg):
        try:
            self.tracking = json.loads(msg.data)
        except ValueError:
            pass

    def on_joint_states(self, msg):
        if not msg.name:
            return
        cur = list(self.measured_upper or self.default_upper)
        hit = 0
        for n, v in zip(msg.name, msg.position):
            i = UPPER_IDX.get(n)
            if i is not None:
                cur[i] = float(v)
                hit += 1
        if hit:
            self.measured_upper = cur
            self.measured_t = time.monotonic()

    def on_estop(self, msg):
        now = time.monotonic()
        if self.estop.update(bool(msg.data), now):
            self.event("estop", "hardware e-stop pressed")
            if self.estop.required and self.fsm.mode not in (DAMP, HOLD):
                self.request(HOLD, "estop_hw", detail="hardware e-stop pressed")

    def update_zone(self):
        if self.tf_buffer is None:
            return
        try:
            from rclpy.time import Time
            t = self.tf_buffer.lookup_transform(self.zones_cfg["frame"], "pelvis", Time())
            self.pose_map = (t.transform.translation.x, t.transform.translation.y)
        except Exception:  # noqa: BLE001
            self.pose_map = None
        self.zone = zone_at(*self.pose_map, self.zones_cfg["zones"]) if self.pose_map else None

    # ------------------------------------------------------------------ outputs
    def publish_upper(self, tgt):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = UPPER_NAMES
        js.position = [float(v) for v in tgt]
        self.pub_upper.publish(js)

    def publish_mode(self):
        m = Mode()
        m.mode = self.fsm.mode
        m.source = self.fsm.source
        m.detail = self.fsm.detail
        self.pub_mode.publish(m)
        self.fsm.changed = False

    def safety_tick(self, now):
        mode = self.fsm.mode
        if self.lease.expire(now):
            self.flush_lease_events()
        if self.watchdog.tripped(mode, self.lease, now):
            age = self.watchdog.age(self.lease.holder, now)
            self.event("watchdog", f"heartbeat of {self.lease.display or self.lease.holder} lost "
                                   f"({'never' if age is None else f'{age:.1f} s'}) in {NAMES[mode]}")
            self.request(HOLD, "watchdog", detail="manual control lost: controller heartbeat missing")
        elif self.estop.required and mode in MOTION and not self.estop.ok(now):
            self.request(HOLD, "estop_hw", detail=self.estop.reason(now))

    def tick(self):
        now = time.monotonic()
        self.safety_tick(now)
        if self.fsm.tick(now):
            self.last_upper = None
            self.publish_mode()
        # blend-out stream after MANIP / AUTO / HOLD -> STAND
        if self.fsm.blending(now):
            start = self.fsm.blend_from or self.default_upper
            a = (now - self.fsm.blend_start) / self.fsm.cur_blend_time
            self.publish_upper(blend(start, self.default_upper, a))
        mode = self.fsm.mode
        v = (0.0, 0.0, 0.0)
        if forwards(mode, "key_vel"):
            v = self.vel.get(now)
            if self.twist_teleop is not None and now - self.twist_teleop_t < self.cfg.get("cmd_vel_teleop_timeout", 0.5) \
                    and (self.vel.last is None or self.twist_teleop_t > self.vel.last):
                v = self.twist_teleop
            v = clamp_speed(v, self.zones_cfg.get("manual_cap"))
        elif forwards(mode, "cmd_vel_nav"):
            if self.twist_nav is not None and now - self.twist_nav_t < self.cfg.get("cmd_vel_nav_timeout", 0.5):
                v = self.twist_nav
        if self.zone is not None:
            v = clamp_speed(v, self.zone["max_speed"])
        v = clamp_twist(v, self.limits)
        t = Twist()
        t.linear.x, t.linear.y, t.angular.z = (float(x) for x in v)
        self.pub_cmd.publish(t)
        self.cmd = v

    def status_dict(self):
        now = time.monotonic()
        locked = [] if self.estop.ok(now) else [self.estop.reason(now)]
        hb_age = self.watchdog.age(self.lease.holder, now) if self.lease.holder else None
        return {"mode": NAMES[self.fsm.mode], "source": self.fsm.source, "detail": self.fsm.detail,
                "cmd_vel": [round(x, 3) for x in self.cmd], "last_key": self.last_key,
                "last_message": self.last_msg, "tracking": self.tracking,
                "auto_input_age": round(min(99.0, now - self.auto_t), 2),
                "hold": self.hold_info, "blending": self.fsm.blending(now), "arms_hold": self.fsm.arms_hold,
                "hand_override": self.last_override, "hand_override_allowed": self.fsm.hand_override_allowed(),
                "lease": self.lease.as_dict(now),
                "watchdog": {"timeout": self.watchdog.timeout,
                             "holder_heartbeat_age": None if hb_age is None else round(hb_age, 2)},
                "estop": self.estop.as_dict(now),
                "motion_locked": locked,
                "zone": None if self.zone is None else {"name": self.zone["name"], "cap": self.zone["max_speed"]},
                "manual_cap": self.zones_cfg.get("manual_cap"),
                "events": self.events[-5:]}

    def heartbeat(self):
        self.pub_status.publish(String(data=json.dumps(self.status_dict())))


def main():
    rclpy.init()
    node = ManagerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # context torn down by SIGINT while spinning
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
