"""G1 policy runner node: /lowstate -> /lowcmd (docs/interfaces.md 3.1, 3.2, 4).

Driven by /lowstate (500 Hz from the sim or the real robot): every message produces one /lowcmd; the policy
runs every control_dt / sim_dt messages. Phases:
  damp     mode DAMP (or no mode yet): kp = 0, kd = damp_kd on all motors
  standup  entering any active mode from damp: PD-interpolate from the current pose to default_qpos over
           standup_time, then start the policy (recurrent state reset)
  HOLD     ("Stop", mode 6): the policy keeps balancing with cmd_vel 0; the upper body freezes where it IS the moment
           HOLD arrives: new set-point = measured q + the static PD offset (gravity sag, and the mission's sag
           compensation), for joints whose command is moving (the others keep their command), estimated per joint as a 0.3 s low-pass of q_des - q - (kd/kp) dq, time constant 0.3 s while
           that joint and its command are quasi-static (|dq|, |dq_des/dt| < 0.15 rad/s), 1.5 s while moving (the
           inertial tracking error of continuous teleop motion averages out instead of leaking into the set-point). Freezing the old
           set-point would let a moving arm travel on by its tracking lag (~kd v / kp plus inertia; 5 cm at a wrist
           was measured); freezing at the measured q would let it sag. /g1/upper_body_target is ignored until the
           mode leaves HOLD. The upper-body kd is raised by hold_kd_scale in HOLD so a moving arm brakes quickly.
           The same freeze applies in MANIP while /g1/arms_hold (Bool, manager) is true ("hold arms" inside MANIP).
  policy   STAND / LOCO / NAV / MANIP: policy with cmd_vel from /g1/cmd_vel (forced 0 unless LOCO/NAV/AUTO) and
           upper-body targets from /g1/upper_body_target (manager publishes it in MANIP/AUTO only, else default); both rate-limited
Fall detection (tilt > fall_tilt for fall_time while standing/policy) -> own damp immediately and request
DAMP via /g1/set_mode (source "safety").
"""
import json
import math
import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, String
from unitree_hg.msg import LowCmd, LowState

from g1_interfaces.msg import Mode
from g1_interfaces.srv import SetMode
from g1_controller.policy import PolicyRunner, UPPER_BODY, projected_gravity
from g1_controller.crc import lowcmd_crc

MODE_NAMES = {0: "DAMP", 1: "STAND", 2: "LOCO", 3: "MANIP", 4: "NAV", 5: "AUTO", 6: "HOLD"}
HOLD = 6   # g1_interfaces/Mode.HOLD (literal so an older g1_interfaces build still imports)


def resolve_policy(name):
    """Absolute dir, else $HG1_ROOT/g1-whole-body-rl/exports/<name>, else this package's policies/<name>."""
    if os.path.isabs(name) and os.path.isdir(name):
        return name
    root = os.environ.get("HG1_ROOT", "")
    cand = os.path.join(root, "g1-whole-body-rl", "exports", name)
    if root and os.path.exists(os.path.join(cand, "policy.json")):
        return cand
    try:
        from ament_index_python.packages import get_package_share_directory
        cand = os.path.join(get_package_share_directory("g1_controller"), "policies", name)
        if os.path.exists(os.path.join(cand, "policy.json")):
            return cand
    except Exception:
        pass
    here = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "policies", name)
    if os.path.exists(os.path.join(here, "policy.json")):
        return os.path.abspath(here)
    raise FileNotFoundError(f"policy '{name}' not found (abs path, g1-whole-body-rl/exports/, g1_controller/policies/)")


class ControllerNode(Node):
    def __init__(self):
        super().__init__("g1_controller")
        p = self.declare_parameter
        p("policy", "unitree_g1_12dof_placeholder")
        p("damp_kd", 3.0)
        p("standup_time", 2.0)
        p("cmd_acc_limit", [1.0, 1.0, 2.0])     # m/s^2, m/s^2, rad/s^2
        p("cmd_timeout", 0.5)
        p("upper_rate_limit", 1.5)              # rad/s per joint
        p("upper_target_timeout", 0.5)
        p("fall_tilt", 0.9)                     # rad from upright
        p("fall_time", 0.2)
        p("compute_crc", True)
        p("status_rate", 2.0)
        p("base_height", 0.0)                   # commanded pelvis height (m); <= 0 -> sidecar home height
        p("hold_kd_scale", 2.5)                 # HOLD: upper-body damping x this (brakes a moving arm; 1 = off)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.policy_dir = resolve_policy(g("policy"))
        self.pol = PolicyRunner(self.policy_dir)
        sc = self.pol.sc
        self.sc = sc
        self.decimation = max(1, int(round(sc.control_dt / sc.sim_dt)))
        self.get_logger().info(
            f"policy {sc.name} ({'PLACEHOLDER' if sc.placeholder else 'trained'}) from {self.policy_dir}; "
            f"obs {self.pol.obs_dim}, control_dt {sc.control_dt}, decimation {self.decimation}")
        lim = sc.raw.get("cmd_vel_limits", {"vx": [-0.5, 0.8], "vy": [-0.4, 0.4], "wz": [-1.0, 1.0]})
        self.cmd_lo = np.array([lim["vx"][0], lim["vy"][0], lim["wz"][0]])
        self.cmd_hi = np.array([lim["vx"][1], lim["vy"][1], lim["wz"][1]])

        self.mode = None
        self.phase = "damp"
        self.fall = False
        self.fall_since = None
        self.count = 0
        self.targets = sc.default_qpos.copy()
        self.cmd_raw = np.zeros(3)
        self.cmd_raw_t = 0.0
        self.cmd = np.zeros(3)
        self.upper_raw = None
        self.upper_raw_t = 0.0
        self.upper = sc.default_qpos[UPPER_BODY].copy()
        self.hold_upper = None         # frozen upper-body command while in HOLD
        self.q_upper = None            # latest measured upper-body q
        self.sag = np.zeros(len(UPPER_BODY))   # low-passed static PD offset q_des - q - kd/kp dq (upper body)
        self._tu_vel = np.zeros(len(UPPER_BODY))   # upper-body command speed (per policy step)
        self._tu_vel_lp = np.zeros(len(UPPER_BODY))
        kpu, kdu = sc.kp[UPPER_BODY].astype(float), sc.kd[UPPER_BODY].astype(float)
        self.lag_gain = np.where(kpu > 0, kdu / np.maximum(kpu, 1e-6), 0.0)
        bh = float(self.get_parameter("base_height").value)
        self.base_height = bh if bh > 0 else sc.home_pelvis_height
        self.last_lowstate_wall = None
        self.rate_win = [time.monotonic(), 0, 0]
        self.loop_hz = 0.0
        self.policy_hz = 0.0
        self.policy_ms = 0.0
        self.mode_machine = 0

        be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        rel = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_cmd = self.create_publisher(LowCmd, "/lowcmd", rel)
        self.pub_status = self.create_publisher(String, "/g1/controller/status", 10)
        self.create_subscription(LowState, "/lowstate", self.on_lowstate, be)
        self.create_subscription(Mode, "/g1/mode", self.on_mode, latched)
        self.create_subscription(Twist, "/g1/cmd_vel", self.on_cmd_vel, 10)
        self.create_subscription(JointState, "/g1/upper_body_target", self.on_upper, 10)
        self.arms_hold = False
        self.create_subscription(Bool, "/g1/arms_hold", self.on_arms_hold, latched)
        # Optional: only used when the sidecar enables command_details.base_height (else the obs term is 0).
        self.create_subscription(Float32, "/g1/cmd_base_height", self.on_base_height, 10)
        self.set_mode_cli = self.create_client(SetMode, "/g1/set_mode")
        self.create_timer(1.0 / g("status_rate"), self.publish_status)

        self.msg = LowCmd()
        self.msg.mode_pr = 0
        self.kp = sc.kp.astype(float)
        self.kd = sc.kd.astype(float)
        self.name_to_idx = {n: i for i, n in enumerate(sc.joint_names)}

    # ------------------------------------------------------------------ inputs
    def frozen(self, mode=None, arms_hold=None):
        mode = self.mode if mode is None else mode
        arms_hold = self.arms_hold if arms_hold is None else arms_hold
        return mode == HOLD or (mode == Mode.MANIP and arms_hold)

    def _freeze(self, why):
        if self.q_upper is not None and self.phase == "policy":
            # joints whose command is moving stop where they are now (measured + static offset); joints whose
            # command is still (e.g. a wrist swung by the shoulder) keep that command, which is where they settle
            moving = np.abs(self._tu_vel_lp) > 0.05
            self.hold_upper = np.where(moving, self.q_upper + self.sag, self.targets[UPPER_BODY])
        else:
            self.hold_upper = self.upper.copy()
        self.upper = self.hold_upper.copy()
        self.get_logger().info(f"{why}: arms frozen")

    def _unfreeze(self):
        if self.hold_upper is not None:
            # keep the frozen pose as the target until the manager's blend / mission / teleop stream takes over
            self.upper_raw = self.hold_upper.copy()
            self.upper_raw_t = time.monotonic()
            self.hold_upper = None

    def on_arms_hold(self, msg):
        was = self.frozen()
        self.arms_hold = bool(msg.data)
        now = self.frozen()
        if now and not was:
            self._freeze("MANIP arms hold")
        elif was and not now:
            self._unfreeze()

    def on_mode(self, msg):
        prev = self.mode
        was = self.frozen(prev) if prev is not None else False
        self.mode = msg.mode
        now = self.frozen()
        if now and not was:
            self._freeze(f"{MODE_NAMES.get(msg.mode)} (source {msg.source})")
        elif was and not now:
            self._unfreeze()
        if msg.mode == Mode.DAMP:
            self.phase = "damp"
            self.hold_upper = None
            self.fall = False  # the fall has been acknowledged (manager is in DAMP); 'r' may stand up again
            if prev is not None and prev != Mode.DAMP:
                self.get_logger().info(f"DAMP (source {msg.source})")

    def on_cmd_vel(self, msg):
        self.cmd_raw = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self.cmd_raw_t = time.monotonic()

    def on_base_height(self, msg):
        self.base_height = float(msg.data)

    def on_upper(self, msg):
        tgt = self.upper.copy() if self.upper_raw is None else self.upper_raw.copy()
        if msg.name:
            for n, v in zip(msg.name, msg.position):
                i = self.name_to_idx.get(n)
                if i is None and not n.endswith("_joint"):
                    i = self.name_to_idx.get(n + "_joint")
                if i is not None and i >= 12:
                    tgt[i - 12] = v
        elif len(msg.position) == 17:
            tgt = np.array(msg.position, float)
        self.upper_raw = tgt
        self.upper_raw_t = time.monotonic()

    # ------------------------------------------------------------------ main loop (per lowstate)
    def on_lowstate(self, ls):
        now = time.monotonic()
        self.last_lowstate_wall = now
        self.mode_machine = ls.mode_machine
        ms = ls.motor_state
        q = np.fromiter((ms[i].q for i in range(29)), float, 29)
        dq = np.fromiter((ms[i].dq for i in range(29)), float, 29)
        quat = np.array(ls.imu_state.quaternion, float)
        gyro = np.array(ls.imu_state.gyroscope, float)
        active = self.mode is not None and self.mode != Mode.DAMP and not self.fall
        if not active:
            self.phase = "damp"
        elif self.phase == "damp":
            self.phase = "standup"
            self.standup_t = 0.0
            self.standup_q0 = q.copy()
            self.upper = self.sc.default_qpos[UPPER_BODY].copy()
            self.cmd = np.zeros(3)

        if self.phase in ("standup", "policy"):
            self.check_fall(quat, now)
        if self.phase == "policy":
            qu, tu, dqu = q[UPPER_BODY], self.targets[UPPER_BODY], dq[UPPER_BODY]
            still = (np.abs(dqu) < 0.15) & (np.abs(self._tu_vel) < 0.15)
            off = tu - qu - self.lag_gain * dqu
            rate = np.where(still, self.sc.sim_dt / 0.3, self.sc.sim_dt / 1.5)   # moving: slow average (inertia ~0)
            self.sag += rate * (off - self.sag)
            self.q_upper = qu

        if self.phase == "damp":
            self.send(np.zeros(29), np.zeros(29), np.full(29, self.get_parameter("damp_kd").value), q)
        elif self.phase == "standup":
            T = self.get_parameter("standup_time").value
            self.standup_t += self.sc.sim_dt
            a = min(1.0, self.standup_t / T)
            a = a * a * (3 - 2 * a)
            tgt = self.standup_q0 + (self.sc.default_qpos - self.standup_q0) * a
            self.send(tgt, self.kp, self.kd, q)
            if self.standup_t >= T:
                self.phase = "policy"
                self.pol.reset()
                self.count = 0
                self.get_logger().info("stand-up done: policy running")
        else:
            if self.count % self.decimation == 0:
                t0 = time.perf_counter()
                self.update_commands(now)
                prev_tu = self.targets[UPPER_BODY].copy()
                self.targets = self.pol.step(q, dq, quat, gyro, self.cmd, self.upper, self.base_height)
                self._tu_vel = (self.targets[UPPER_BODY] - prev_tu) / self.sc.control_dt
                self._tu_vel_lp += 0.3 * (np.abs(self._tu_vel) - self._tu_vel_lp)    # ~60 ms smoothing
                self.policy_ms = 0.9 * self.policy_ms + 0.1 * 1e3 * (time.perf_counter() - t0)
                self.rate_win[2] += 1
            self.count += 1
            kd = self.kd
            if self.frozen() and self.hold_upper is not None:
                kd = self.kd.copy()
                kd[UPPER_BODY] *= float(self.get_parameter("hold_kd_scale").value)
            self.send(self.targets, self.kp, kd, q)
        self.rate_win[1] += 1
        if now - self.rate_win[0] >= 1.0:
            dtw = now - self.rate_win[0]
            self.loop_hz = self.rate_win[1] / dtw
            self.policy_hz = self.rate_win[2] / dtw
            self.rate_win = [now, 0, 0]

    def update_commands(self, now):
        dt = self.sc.control_dt
        moving = self.mode in (Mode.LOCO, Mode.NAV, Mode.AUTO)  # AUTO: Nav2 via the manager (contract 6)
        raw = self.cmd_raw if (moving and now - self.cmd_raw_t < self.get_parameter("cmd_timeout").value) \
            else np.zeros(3)
        raw = np.clip(raw, self.cmd_lo, self.cmd_hi)
        acc = np.array(self.get_parameter("cmd_acc_limit").value, float) * dt
        self.cmd = self.cmd + np.clip(raw - self.cmd, -acc, acc)
        default_upper = self.sc.default_qpos[UPPER_BODY]
        # The manager is the gate: it publishes /g1/upper_body_target only in MANIP and during the ~1 s
        # blend back to default when leaving MANIP. Absent/stale target -> default upper body.
        if self.frozen() and self.hold_upper is not None:
            self.upper = self.hold_upper.copy()      # frozen: no motion at all
            return
        use = (self.upper_raw is not None
               and now - self.upper_raw_t < self.get_parameter("upper_target_timeout").value)
        tgt = self.upper_raw if use else default_upper
        step = self.get_parameter("upper_rate_limit").value * dt
        self.upper = self.upper + np.clip(tgt - self.upper, -step, step)

    def check_fall(self, quat, now):
        gz = projected_gravity(quat)[2]
        tilt = math.acos(max(-1.0, min(1.0, -gz)))
        if tilt > self.get_parameter("fall_tilt").value:
            if self.fall_since is None:
                self.fall_since = now
            elif now - self.fall_since > self.get_parameter("fall_time").value and not self.fall:
                self.fall = True
                self.phase = "damp"
                self.get_logger().error(f"FALL detected (tilt {tilt:.2f} rad): damping, requesting DAMP")
                if self.set_mode_cli.service_is_ready():
                    self.set_mode_cli.call_async(SetMode.Request(mode=Mode.DAMP, source="safety"))
        else:
            self.fall_since = None

    def send(self, q_des, kp, kd, q_now):
        m = self.msg
        m.mode_machine = self.mode_machine
        mc = m.motor_cmd
        for i in range(29):
            c = mc[i]
            c.mode = 1
            c.q = float(q_des[i])
            c.dq = 0.0
            c.tau = 0.0
            c.kp = float(kp[i])
            c.kd = float(kd[i])
        if self.get_parameter("compute_crc").value:
            m.crc = lowcmd_crc(m)
        self.pub_cmd.publish(m)

    def publish_status(self):
        now = time.monotonic()
        age = None if self.last_lowstate_wall is None else round(1e3 * (now - self.last_lowstate_wall), 1)
        st = {
            "policy": self.sc.name,
            "placeholder": self.sc.placeholder,
            "note": ("PLACEHOLDER: Unitree 12-DoF legs-only locomotion policy; waist/arms PD-held. "
                     "Real ULTRA-style policy will come from g1-whole-body-rl/exports/.") if self.sc.placeholder else "",
            "policy_dir": self.policy_dir,
            "mode": MODE_NAMES.get(self.mode, "NONE") if self.mode is not None else "NONE",
            "phase": self.phase,
            "fall_detected": self.fall,
            "loop_hz": round(self.loop_hz, 1),
            "policy_hz": round(self.policy_hz, 1),
            "policy_ms": round(self.policy_ms, 3),
            "lowstate_age_ms": age,
            "cmd_vel": [round(float(v), 3) for v in self.cmd],
            "base_height_cmd": round(self.base_height, 3) if self.sc.base_height_enabled else None,
        }
        self.pub_status.publish(String(data=json.dumps(st)))


def main():
    rclpy.init()
    node = ControllerNode()
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
