"""Skill interface shared by scripted and learned skills.

A skill is ticked at the mission rate (default 50 Hz): `start(ctx, **goal)` once, then `tick(ctx)` until it returns
SUCCESS or FAILURE; `stop(ctx)` on abort/pause. Skills only act through `ctx.robot.command(...)`, i.e. the same
channels teleop and a learned policy use (/g1/auto/upper_body_target, /g1/auto/hand/*/cmd, /g1/cmd_vel_nav), so a
scripted demo, a teleop demo and a policy rollout are the same kind of data (see recorder.py).
"""
import enum
import math
from dataclasses import dataclass, field

import numpy as np

from g1_mission.geometry import interp_T, rot_log, smoothstep
from g1_mission.kinematics import ARM, WAIST


class Status(enum.Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


HAND_OPEN = np.zeros(6)
# thumb, thumb_aux, index, middle, ring, pinky (contract order). Pre-shape = thumb rotated into opposition.
HAND_PRESHAPE = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
HAND_POWER = np.array([0.85, 1.0, 1.0, 1.0, 1.0, 1.0])


@dataclass
class SkillContext:
    robot: object                      # interfaces.RobotInterface
    objects: object = None             # interfaces.ObjectSource
    surgeon: object = None             # interfaces.SurgeonHandSource
    station: object = None             # interfaces.StationEstimator
    ik: object = None                  # kinematics.UpperBodyIK (right arm)
    nav: object = None                 # navigation client (tree/nav.py) or None
    catalog: dict = field(default_factory=dict)     # instrument metadata (config/instruments.yaml)
    locations: dict = field(default_factory=dict)   # named map poses (config/locations.yaml)
    params: dict = field(default_factory=dict)      # skill parameters (config/mission.yaml "skills")
    ready_pose: np.ndarray = field(default_factory=lambda: np.zeros(17))
    recorder: object = None
    log: object = print
    board: dict = field(default_factory=dict)       # blackboard between skills (instrument, grasp transform, ...)
    arm: object = None                 # ArmMotion shared by all arm skills (continuity of targets)


class Skill:
    """Base class. Subclasses set `name` and implement start/tick; `phase` and `message` feed the mission status."""
    name = "skill"

    def __init__(self, **params):
        self.params = params
        self.phase = ""
        self.message = ""
        self.t0 = 0.0

    def p(self, key, default):
        return self.params.get(key, default)

    def start(self, ctx, **goal):
        self.t0 = ctx.robot.now()
        self.phase = "start"
        self.message = ""

    def tick(self, ctx):
        return Status.SUCCESS

    def stop(self, ctx):
        """Abort/pause: freeze arm targets where they are (the manager takes over on a mode change)."""
        if ctx.arm is not None:
            ctx.arm.hold()

    def fail(self, msg):
        self.message = msg
        return Status.FAILURE

    def shift_time(self, dt):
        """After a pause of dt seconds: move every time reference forward so the skill resumes where it was."""
        self.t0 += dt
        if getattr(self, "phase_t", None) is not None:
            self.phase_t += dt
        for k in ("ok_since",):
            if getattr(self, k, None) is not None:
                setattr(self, k, getattr(self, k) + dt)


class ArmMotion:
    """Joint-target generator for the upper body: Cartesian (TCP pose, IK per tick) or joint-space segments with
    smoothstep timing, plus a slow integral term per arm joint that removes the steady-state sag of the controller's
    PD arm (placeholder kp 20-100) under gravity and payload. Hands are ramped linearly."""

    def __init__(self, ik, ki=3.0, i_max=0.15, side="right"):
        self.ik = ik
        self.side = side
        self.ki, self.i_max = ki, i_max
        self.q_des = None                  # 17, trajectory
        self.integ = np.zeros(17)
        self.seg = None
        self.hand = {"left": HAND_OPEN.copy(), "right": HAND_OPEN.copy()}
        self.hand_seg = {}
        self.ik_err = (0.0, 0.0)

    def init(self, q_meas):
        if self.q_des is None:
            self.q_des = np.array(q_meas, float).copy()

    def reset(self):
        self.q_des, self.seg, self.integ = None, None, np.zeros(17)

    # ------------------------------------------------------------------ segments
    def move_cartesian(self, T_goal, now, speed=0.12, ang_speed=0.8, min_time=0.6, q_rest=None, rot_mask=None):
        T0 = self.ik.fk(self.q_des)
        dist = float(np.linalg.norm(T_goal[:3, 3] - T0[:3, 3]))
        ang = float(np.linalg.norm(rot_log(T0[:3, :3].T @ T_goal[:3, :3])))
        dur = max(min_time, dist / speed, ang / ang_speed)
        self.seg = {"kind": "cart", "T0": T0, "T1": T_goal, "t0": now, "dur": dur, "rest": q_rest,
                    "mask": rot_mask}
        return dur

    def move_joints(self, q_goal, now, max_vel=0.8, min_time=0.8):
        q0 = self.q_des.copy()
        dur = max(min_time, float(np.max(np.abs(np.asarray(q_goal) - q0))) / max_vel * 1.5)
        self.seg = {"kind": "joint", "q0": q0, "q1": np.array(q_goal, float), "t0": now, "dur": dur}
        return dur

    def hold(self):
        self.seg = None

    def shift_time(self, dt):
        if self.seg is not None:
            self.seg["t0"] += dt
        self.hand_seg = {k: (h0, h1, t0 + dt, dur) for k, (h0, h1, t0, dur) in self.hand_seg.items()}

    def move_hand(self, side, target, now, dur=0.8):
        self.hand_seg[side] = (self.hand[side].copy(), np.array(target, float), now, max(1e-3, dur))

    # ------------------------------------------------------------------ per tick
    def step(self, now, q_meas, dt):
        """Advance; returns (upper command 17, {side: hand 6}, segment progress 0..1)."""
        self.init(q_meas)
        prog = 1.0
        s = self.seg
        if s is not None:
            prog = min(1.0, (now - s["t0"]) / s["dur"])
            a = smoothstep(prog)
            if s["kind"] == "joint":
                self.q_des = s["q0"] + a * (s["q1"] - s["q0"])
            else:
                T = interp_T(s["T0"], s["T1"], a)
                q, pe, re = self.ik.solve(T, self.q_des, rest=s["rest"] if s["rest"] is not None else self.q_des,
                                          iters=30, rot_mask=s["mask"])
                self.ik_err = (pe, re)
                self.q_des = q
        q_meas = np.asarray(q_meas, float)
        err = self.q_des - q_meas
        idx = WAIST + ARM[self.side]
        for i in idx:
            if abs(err[i]) < 0.25:
                self.integ[i] = float(np.clip(self.integ[i] + self.ki * dt * err[i], -self.i_max, self.i_max))
        hands = {}
        for side, (h0, h1, t0, dur) in list(self.hand_seg.items()):
            a = min(1.0, (now - t0) / dur)
            self.hand[side] = h0 + a * (h1 - h0)
            if a >= 1.0:
                del self.hand_seg[side]
        for side in ("left", "right"):
            hands[side] = np.clip(self.hand[side], 0.0, 1.0)
        return self.q_des + self.integ, hands, prog

    def tracking_error(self, q_meas):
        idx = WAIST + ARM[self.side]
        return float(np.max(np.abs(self.q_des[idx] - np.asarray(q_meas)[idx])))

    def done(self, now):
        s = self.seg
        return s is None or now - s["t0"] >= s["dur"]

    def hands_done(self):
        return not self.hand_seg


def angle_between(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    return math.acos(max(-1.0, min(1.0, c)))


def drive_arm(ctx, dt):
    """One tick of the shared ArmMotion -> ctx.robot.command. Returns segment progress (0..1) or None if the robot
    has no joint state yet."""
    q = ctx.robot.upper_q()
    if q is None:
        return None
    up, hands, prog = ctx.arm.step(ctx.robot.now(), q, dt)
    ctx.robot.command(upper=up, left_hand=hands["left"], right_hand=hands["right"])
    if ctx.recorder is not None:
        ctx.recorder.action(upper=up, left_hand=hands["left"], right_hand=hands["right"])
    return prog
