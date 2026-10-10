"""Retract (back the empty hand away from the surgeon along the handover axis) and ReturnToReady (joint-space move
to the ready pose, hands open)."""
import numpy as np

from g1_mission.skills.base import HAND_OPEN, Skill, Status, drive_arm


class Retract(Skill):
    name = "retract"

    def start(self, ctx, **goal):
        super().start(ctx)
        self.dt = 1.0 / self.p("rate", 50.0)
        ctx.arm.init(ctx.robot.upper_q())
        T = ctx.ik.fk(ctx.arm.q_des)
        # 1) away from the instrument: against the palm normal (TCP +y faces the grasped object)
        T1 = T.copy()
        T1[:3, 3] -= self.p("clear", 0.06) * T[:3, 1]
        # 2) then back along the handover axis (towards the body)
        d = ctx.board.get("handover_dir")
        if d is None:
            d = T[:3, 3] * np.array([1.0, 1.0, 0.0])
            d /= (np.linalg.norm(d) + 1e-9)
        T2 = T1.copy()
        T2[:3, 3] -= self.p("distance", 0.10) * np.asarray(d)
        T2[2, 3] += self.p("rise", 0.02)
        self.T2 = T2
        ctx.arm.move_cartesian(T1, self.t0, speed=0.08)
        ctx.arm.move_hand(ctx.arm.side, HAND_OPEN, self.t0, 0.3)
        self.phase = "clear"

    def tick(self, ctx):
        if drive_arm(ctx, self.dt) is None:
            return Status.RUNNING
        now = ctx.robot.now()
        if not (ctx.arm.done(now) and now - ctx.arm.seg["t0"] > ctx.arm.seg["dur"] + 0.2):
            return Status.RUNNING
        if self.phase == "clear":
            self.phase = "back"
            ctx.arm.move_cartesian(self.T2, now, speed=0.10)
            return Status.RUNNING
        inst = ctx.board.get("instrument")
        est = ctx.objects.get(inst) if (ctx.objects and inst) else None
        if est is not None:
            gz = float(ctx.catalog.get(inst, {}).get("grasp_z", 0.09))
            g = est.T[:3, :3] @ np.array([0, 0, gz]) + est.T[:3, 3]
            T_tcp = ctx.robot.base_T() @ ctx.ik.fk(ctx.robot.upper_q())
            dist = float(np.linalg.norm(g - T_tcp[:3, 3]))
            if dist < self.p("still_attached_dist", 0.06):
                return self.fail(f"{inst} is still in the hand")
        return Status.SUCCESS


class ReturnToReady(Skill):
    """Joint-space move to the ready pose with the hands open. If the hand is low (at rack height, e.g. after a
    failed grasp) it first rises `lift` m and backs off towards the body, so the joint move cannot sweep the rack."""
    name = "return_to_ready"

    def start(self, ctx, **goal):
        super().start(ctx)
        self.dt = 1.0 / self.p("rate", 50.0)
        ctx.arm.init(ctx.robot.upper_q())
        for side in ("left", "right"):
            ctx.arm.move_hand(side, HAND_OPEN, self.t0, 0.5)
        T = ctx.ik.fk(ctx.arm.q_des)
        T_ready = ctx.ik.fk(ctx.ready_pose)
        if T[2, 3] < T_ready[2, 3] + self.p("low_margin", 0.08) and np.linalg.norm(T[:3, 3] - T_ready[:3, 3]) > 0.08:
            T1 = T.copy()
            T1[2, 3] += self.p("lift", 0.08)
            T1[0, 3] -= self.p("back", 0.05)
            ctx.arm.move_cartesian(T1, self.t0, speed=0.10)
            self.phase = "lift"
        else:
            ctx.arm.move_joints(ctx.ready_pose, self.t0, max_vel=self.p("joint_speed", 0.7))
            self.phase = "to_ready"

    def tick(self, ctx):
        if drive_arm(ctx, self.dt) is None:
            return Status.RUNNING
        now = ctx.robot.now()
        if not ctx.arm.done(now):
            return Status.RUNNING
        if self.phase == "lift":
            self.phase = "to_ready"
            ctx.arm.move_joints(ctx.ready_pose, now, max_vel=self.p("joint_speed", 0.7))
            return Status.RUNNING
        if ctx.arm.tracking_error(ctx.robot.upper_q()) < 0.08 or now - ctx.arm.seg["t0"] > ctx.arm.seg["dur"] + 2.0:
            return Status.SUCCESS
        return Status.RUNNING


class Carry(Skill):
    """Bring the held instrument close to the body before walking (it must not sweep into people or furniture):
    TCP to `carry_pos` (pelvis frame) keeping the grasp orientation, instrument upright."""
    name = "carry"

    def start(self, ctx, **goal):
        super().start(ctx)
        self.dt = 1.0 / self.p("rate", 50.0)
        ctx.arm.init(ctx.robot.upper_q())
        T = ctx.ik.fk(ctx.arm.q_des)
        T[:3, 3] = self.p("carry_pos", [0.22, -0.14, 0.20])
        ctx.arm.move_cartesian(T, self.t0, speed=0.10, q_rest=ctx.ready_pose)
        self.phase = "carry"

    def tick(self, ctx):
        if drive_arm(ctx, self.dt) is None:
            return Status.RUNNING
        now = ctx.robot.now()
        if ctx.arm.done(now) and now - ctx.arm.seg["t0"] > ctx.arm.seg["dur"] + 0.3:
            return Status.SUCCESS
        return Status.RUNNING
