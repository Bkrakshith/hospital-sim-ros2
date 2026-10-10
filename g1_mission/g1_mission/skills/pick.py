"""ScriptedPick: take an upright instrument out of the tool rack with a power grasp of the right hand.

Grasp: "handshake" orientation. TCP (right_grasp) z = instrument axis (index finger towards the handle), TCP x =
horizontal approach direction (shoulder -> instrument), palm (+y) facing the instrument. Sequence:
  preshape  thumb rotated into opposition (thumb_aux = 1), fingers open
  above     grasp orientation, `above` m over and `above_back` m behind the pre-grasp point (the hand turns
            away from the rack, so the fingertips cannot sweep through the instruments)
  pregrasp  TCP `pre_dist` short of the grasp point along the approach direction (Cartesian IK)
  approach  slide forward until the instrument sits in the palm pocket (the rotated thumb is the stop)
  close     power grasp (HAND_POWER), then settle
  straighten  closing the fingers tilts the instrument in the hand; rotate the hand about the grasp point so the
            observed instrument axis is vertical again (the rack's holes are vertical: a tilted instrument jams)
  lift      straight up `lift` m (out of the slot)
  retreat   back towards the robot `retreat` m (clears the rack row before any sideways move)
  verify    instrument moved with the hand (object pose within `verify_dist` of the TCP, lifted > lift/2)
Targets are recomputed from the latest instrument estimate until the approach starts (closed loop on the
object source; ground truth now, perception later).
"""
import math

import numpy as np

from g1_mission.geometry import inv, make_T, rot_axis
from g1_mission.skills.base import HAND_OPEN, HAND_POWER, HAND_PRESHAPE, Skill, Status, drive_arm


def grasp_pose(ctx, T_obj_world, grasp_z, approach_axis="y"):
    """TCP target (pelvis frame) for a power grasp of an instrument at T_obj_world. Returns (T_tcp_pelvis,
    approach unit vector in the pelvis frame).
    Approach = the instrument axis `approach_axis` that points at the robot in the rack (catalogue key; hospital-sim-env/objects
    assets: x = the wide axis, rings/jaws towards the robot; theatre.py stand-ins: y), sign chosen towards the robot,
    made horizontal and perpendicular to the instrument axis. Approaching square to the rack row keeps the fingers and the back of the hand clear of the neighbours."""
    T_pw = inv(ctx.robot.base_T())
    T_obj = T_pw @ T_obj_world
    g = T_obj[:3, :3] @ np.array([0, 0, grasp_z]) + T_obj[:3, 3]
    z = T_obj[:3, 2] / np.linalg.norm(T_obj[:3, 2])
    sh = ctx.ik.fk(ctx.ready_pose, f"{ctx.arm.side}_shoulder_pitch_link")[:3, 3]
    a0 = g - sh
    a0[2] = 0.0
    ay = T_obj[:3, {"x": 0, "y": 1}.get(approach_axis, 1)].copy()
    ay[2] = 0.0
    if np.linalg.norm(ay) < 0.3:          # instrument lying flat: fall back to shoulder -> object
        ay = a0.copy()
    a = ay if np.dot(ay, a0) >= 0 else -ay
    a = a - np.dot(a, z) * z
    a /= np.linalg.norm(a)
    y = np.cross(z, a)
    return make_T(np.c_[a, y, z], g), a


class ScriptedPick(Skill):
    name = "pick"

    def start(self, ctx, instrument=None, **goal):
        super().start(ctx)
        self.inst = instrument or ctx.board.get("instrument")
        meta = ctx.catalog.get(self.inst, {})
        self.grasp_z = float(meta.get("grasp_z", 0.09))
        self.axis = meta.get("approach_axis", "y")
        # per-instrument grasp parameters (config/grasp_overrides.yaml) over the skill defaults
        self.hand_close = np.array(meta.get("hand_close", HAND_POWER), float)
        self.straighten = bool(meta.get("straighten", True))
        self.pre_dist = self.p("pre_dist", 0.10)
        self.lift = float(meta.get("lift", self.p("lift", 0.12)))
        self.retreat = self.p("retreat", 0.08)
        self.dt = 1.0 / self.p("rate", 50.0)
        est = ctx.objects.get(self.inst) if ctx.objects else None
        if est is None:
            self.phase = "failed"
            self.message = f"no pose for {self.inst}"
            return
        self.z0 = float(est.T[2, 3])
        ctx.arm.init(ctx.robot.upper_q())
        ctx.arm.move_hand(ctx.arm.side, HAND_PRESHAPE, ctx.robot.now(), 0.6)
        self._goto("above")

    def _targets(self, ctx):
        est = ctx.objects.get(self.inst)
        T_g, a = grasp_pose(ctx, est.T, self.grasp_z, self.axis)
        T_pre = T_g.copy()
        T_pre[:3, 3] -= self.pre_dist * a
        return T_g, T_pre, a

    def _goto(self, phase):
        self.phase = phase
        self.phase_t = None

    def tick(self, ctx):
        if self.phase == "failed":
            return Status.FAILURE
        now = ctx.robot.now()
        arm = ctx.arm
        if self.phase_t is None:
            self.phase_t = now
            if self.phase == "above":
                T_g, T_pre, a = self._targets(ctx)
                T_up = T_pre.copy()
                T_up[:3, 3] -= self.p("above_back", 0.05) * a
                T_up[2, 3] += self.p("above", 0.12)
                arm.move_cartesian(T_up, now, speed=0.15, q_rest=ctx.ready_pose)
            elif self.phase == "pregrasp":
                T_g, T_pre, a = self._targets(ctx)
                self.T_g, self.a = T_g, a
                arm.move_cartesian(T_pre, now, speed=0.15, q_rest=ctx.ready_pose)
            elif self.phase == "approach":
                T_g, _, a = self._targets(ctx)
                self.T_g, self.a = T_g, a
                arm.move_cartesian(T_g, now, speed=self.p("approach_speed", 0.06))
            elif self.phase == "close":
                arm.hold()
                arm.move_hand(arm.side, self.hand_close, now, self.p("close_time", 1.0))
            elif self.phase == "straighten":
                T = ctx.ik.fk(arm.q_des)
                est = ctx.objects.get(self.inst) if ctx.objects else None
                if est is not None and self.straighten:
                    T_obj = inv(ctx.robot.base_T()) @ est.T
                    a = T_obj[:3, 2] / np.linalg.norm(T_obj[:3, 2])
                    up = np.array([0.0, 0.0, 1.0])
                    ang = math.acos(max(-1.0, min(1.0, float(a @ up))))
                    if self.p("straighten_min_deg", 2.0) < math.degrees(ang) < 35.0:
                        axis = np.cross(a, up)
                        D = rot_axis(axis, ang)
                        g = T_obj[:3, :3] @ np.array([0, 0, self.grasp_z]) + T_obj[:3, 3]
                        T2 = T.copy()
                        T2[:3, :3] = D @ T[:3, :3]
                        T2[:3, 3] = D @ (T[:3, 3] - g) + g
                        T = T2
                        self.message = f"straighten {math.degrees(ang):.1f} deg"
                arm.move_cartesian(T, now, speed=0.05, ang_speed=0.4, min_time=0.4)
                self.T_g = T
            elif self.phase == "lift":
                # straight up out of the slot (the rack's holes are vertical)
                T = self.T_g.copy()
                T[2, 3] += self.lift
                arm.move_cartesian(T, now, speed=self.p("lift_speed", 0.06))
                self.T_lift = T
            elif self.phase == "retreat":
                T = self.T_lift.copy()
                T[:3, 3] -= self.retreat * self.a
                arm.move_cartesian(T, now, speed=0.08)
        prog = drive_arm(ctx, self.dt)
        if prog is None:
            return Status.RUNNING
        pe, _ = arm.ik_err
        if arm.done(now) and pe > self.p("ik_tol", 0.02):
            return self.fail(f"{self.phase}: target out of reach (IK error {pe * 100:.1f} cm)")
        el = now - self.phase_t
        if self.phase == "above":
            if arm.done(now) and (arm.tracking_error(ctx.robot.upper_q()) < 0.08 or el > arm.seg["dur"] + 1.5):
                self._goto("pregrasp")
        elif self.phase == "pregrasp":
            if arm.done(now) and (arm.tracking_error(ctx.robot.upper_q()) < 0.05 or el > arm.seg["dur"] + 2.0):
                self._goto("approach")
        elif self.phase == "approach":
            if arm.done(now) and (arm.tracking_error(ctx.robot.upper_q()) < 0.04 or el > arm.seg["dur"] + 1.5):
                self._goto("close")
        elif self.phase == "close":
            if el > self.p("close_time", 1.0) + self.p("settle_time", 0.6):
                self._goto("straighten")
        elif self.phase == "straighten":
            if arm.done(now) and el > arm.seg["dur"] + 0.3:
                self._goto("lift")
        elif self.phase == "lift":
            if arm.done(now) and el > arm.seg["dur"] + 0.3:
                self._goto("retreat")
        elif self.phase == "retreat":
            if arm.done(now) and el > arm.seg["dur"] + 0.3:
                return self._verify(ctx)
        return Status.RUNNING

    def _verify(self, ctx):
        est = ctx.objects.get(self.inst) if ctx.objects else None
        T_tcp = ctx.robot.base_T() @ ctx.ik.fk(ctx.robot.upper_q())
        if est is None:
            self.message = "picked (not verified: no object estimate)"
            ctx.board["T_tcp_obj"] = None
            return Status.SUCCESS
        g = est.T[:3, :3] @ np.array([0, 0, self.grasp_z]) + est.T[:3, 3]
        d = float(np.linalg.norm(g - T_tcp[:3, 3]))
        dz = float(est.T[2, 3] - self.z0)
        # the instrument must have come up with the hand; it may have slid along its axis in the grasp (the
        # measured transform below is what the handover uses), so the distance bound is loose
        if d > self.p("verify_dist", 0.09) or dz < self.lift / 2:
            ctx.arm.move_hand(ctx.arm.side, HAND_OPEN, ctx.robot.now(), 0.5)
            return self.fail(f"grasp failed: {self.inst} {d * 100:.1f} cm from the hand, lifted {dz * 100:.1f} cm")
        ctx.board["T_tcp_obj"] = inv(T_tcp) @ est.T        # measured grasp (used by the handover)
        self.message = f"picked {self.inst} (grasp offset {d * 100:.1f} cm)"
        return Status.SUCCESS
