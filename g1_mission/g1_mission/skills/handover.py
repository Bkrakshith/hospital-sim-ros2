"""ScriptedHandover: present the held instrument handle-first to the surgeon's hand and let go once it is taken.

  plan      desired instrument pose: handle end `standoff` m short of the surgeon's hand, axis d = presentation
            direction. d is searched over yaw offsets from the shoulder->hand direction (towards the midline lets the
            forearm point forward with the hand supinated), a small downward tilt and `roll_candidates` rolls about d;
            candidates whose handover and pre-handover poses are both reachable are scored (IK residual, distance
            from joint limits, waist use, joint travel). Runs incrementally (plan_ms_per_tick) so the arm stream
            never stalls.
            Grasp transform = what ScriptedPick measured (ctx.board["T_tcp_obj"]), else the nominal grasp.
  to_pre    joint-space move to the same pose `pre_dist` m back along the axis
  present   straight in along the axis (Cartesian), then up to `servo_iters` corrections that move the observed
            handle end onto the hand (the grasp can slip while carrying)
  wait_grip hold still until the surgeon holds it: SurgeonHandSource.holding() -> True (sim: simulated grip = CHEAT);
            a source that cannot tell (None) is treated as taken after `blind_wait` s. `grip_timeout` -> FAILURE.
  release   open the hand
"""
import math

import numpy as np

from g1_mission.geometry import inv, make_T, rot_axis
from g1_mission.skills.base import HAND_OPEN, Skill, Status, drive_arm


class ScriptedHandover(Skill):
    name = "handover"

    def start(self, ctx, instrument=None, **goal):
        super().start(ctx)
        self.inst = instrument or ctx.board.get("instrument")
        meta = ctx.catalog.get(self.inst, {})
        self.grasp_z = float(meta.get("grasp_z", 0.09))
        self.handle_z = float(meta.get("handle_z", meta.get("length", 0.17)))
        self.dt = 1.0 / self.p("rate", 50.0)
        self.phase, self.phase_t = "plan", None
        self.gen = None
        self.servo_n = 0
        ctx.arm.init(ctx.robot.upper_q())

    def _candidates(self, ctx):
        """Generator of (d, T_tcp_present) candidates; also sets up the geometry. Handle end at the surgeon's hand,
        instrument axis d = presentation direction (handle first). d sweeps yaw around the shoulder->hand direction
        (towards the robot's midline: the forearm can then point forward with the hand supinated) and a small
        downward tilt; the roll about d is free."""
        hand = ctx.surgeon.get() if ctx.surgeon else None
        if hand is None:
            self.err = "no surgeon hand estimate"
            return
        T_pw = inv(ctx.robot.base_T())
        h = (T_pw @ hand.T)[:3, 3]
        sh = ctx.ik.fk(ctx.arm.q_des, f"{ctx.arm.side}_shoulder_pitch_link")[:3, 3]
        d0 = h - sh
        d0[2] = 0.0
        yaw0 = math.atan2(d0[1], d0[0])
        T_tcp_obj = ctx.board.get("T_tcp_obj")
        if T_tcp_obj is None:
            T_tcp_obj = make_T(np.eye(3), [0, 0, -self.grasp_z])
        T_obj_tcp = inv(T_tcp_obj)
        up = np.array([0.0, 0.0, 1.0])
        yaws = [math.radians(a) for a in self.p("yaw_offsets_deg", [0, -20, -40, -60, -80, 20])]
        tilts = [math.radians(a) for a in self.p("tilt_deg", [0, -20])]
        n = int(self.p("roll_candidates", 12))
        for dy in yaws:
            for tl in tilts:
                c, sn = math.cos(yaw0 + dy), math.sin(yaw0 + dy)
                d = np.array([c * math.cos(tl), sn * math.cos(tl), math.sin(tl)])
                x0 = np.cross(up, d)
                x0 /= np.linalg.norm(x0)
                R0 = np.c_[x0, np.cross(d, x0), d]
                o = h - (self.handle_z + self.p("standoff", 0.01)) * d
                for k in range(n):
                    phi = -math.pi + 2 * math.pi * k / n
                    T_obj = make_T(rot_axis(d, phi) @ R0, o)
                    yield d, T_obj @ T_obj_tcp

    def _handle_error(self, ctx):
        """Pelvis-frame vector from the observed handle end (+ standoff) to the surgeon's hand, or None."""
        est = ctx.objects.get(self.inst) if ctx.objects else None
        hand = ctx.surgeon.get() if ctx.surgeon else None
        if est is None or hand is None:
            return None
        T_pw = inv(ctx.robot.base_T())
        T_obj = T_pw @ est.T
        handle = T_obj[:3, :3] @ np.array([0, 0, self.handle_z]) + T_obj[:3, 3]
        target = (T_pw @ hand.T)[:3, 3] - self.p("standoff", 0.01) * self.d
        return target - handle

    def _score(self, ctx, d, T_tcp, q_now):
        ik = ctx.ik
        q, pe, re = ik.solve(T_tcp, q_now, rest=ctx.ready_pose, iters=80)
        if pe > 0.01 or re > 0.1:
            return None
        T_pre = T_tcp.copy()
        T_pre[:3, 3] -= self.p("pre_dist", 0.10) * d
        q_pre, pe2, re2 = ik.solve(T_pre, q, rest=ctx.ready_pose, iters=80)
        if pe2 > 0.01 or re2 > 0.1:
            return None
        idx = ik.active
        margin = np.minimum(q[idx] - ik.lo[idx], ik.hi[idx] - q[idx])
        limit_pen = float(np.sum(np.clip(0.25 - margin, 0.0, None) ** 2)) * 40.0     # stay 0.25 rad off the limits
        cost = pe * 100 + re * 5 + limit_pen + 1.0 * float(np.abs(q[:3]).sum()) \
            + 0.1 * float(np.linalg.norm(q_pre - q_now))
        return cost, q_pre

    def tick(self, ctx):
        now = ctx.robot.now()
        arm = ctx.arm
        if self.phase == "plan":
            # incremental search (a few IK solves per tick) so the 50 Hz target stream never stalls
            if self.gen is None:
                self.err, self.best, self.q_now = "", None, arm.q_des.copy()
                self.gen = self._candidates(ctx)
            t_budget = self.p("plan_ms_per_tick", 12.0) / 1000.0
            import time as _t
            t0 = _t.perf_counter()
            done = False
            while _t.perf_counter() - t0 < t_budget:
                try:
                    d, T_tcp = next(self.gen)
                except StopIteration:
                    done = True
                    break
                r = self._score(ctx, d, T_tcp, self.q_now)
                if r is not None and (self.best is None or r[0] < self.best[0]):
                    self.best = (r[0], d, T_tcp, r[1])
            drive_arm(ctx, self.dt)
            if not done:
                self.message = "planning the presentation"
                return Status.RUNNING
            if self.best is None:
                return self.fail(self.err or "surgeon hand out of reach (no reachable presentation)")
            _, self.d, self.T_present, self.q_pre = self.best
            ctx.board["handover_dir"] = self.d
            self.message = f"presenting along {np.round(self.d, 2).tolist()}"
            self.phase, self.phase_t = "to_pre", now
            arm.move_joints(self.q_pre, now, max_vel=self.p("joint_speed", 0.7))
        prog = drive_arm(ctx, self.dt)
        if prog is None:
            return Status.RUNNING
        el = now - self.phase_t
        if self.phase == "to_pre":
            if arm.done(now) and (arm.tracking_error(ctx.robot.upper_q()) < 0.05 or el > arm.seg["dur"] + 2.0):
                self.phase, self.phase_t = "present", now
                arm.move_cartesian(self.T_present, now, speed=self.p("present_speed", 0.06))
        elif self.phase == "present":
            if arm.done(now) and el > arm.seg["dur"] + 0.3:
                # closed-loop correction on the observed instrument (ObjectSource: ground truth now, perception
                # later): the grasp can slip while carrying, so move the handle end onto the hand, up to 3 times
                err = self._handle_error(ctx)
                tol, n = self.p("servo_tol", 0.015), self.p("servo_iters", 3)
                if err is not None and np.linalg.norm(err) > tol and self.servo_n < n:
                    self.servo_n += 1
                    step = err * min(1.0, self.p("servo_max_step", 0.05) / np.linalg.norm(err))
                    self.T_present[:3, 3] += step
                    arm.move_cartesian(self.T_present, now, speed=0.04, min_time=0.4)
                    self.phase_t = now
                    self.message = f"servo {self.servo_n}: handle {np.linalg.norm(err) * 100:.1f} cm off"
                else:
                    self.phase, self.phase_t = "wait_grip", now
                    arm.hold()
        elif self.phase == "wait_grip":
            held = ctx.surgeon.holding(self.inst) if ctx.surgeon else None
            if held or (held is None and el > self.p("blind_wait", 2.0)):
                self.phase, self.phase_t = "release", now
                arm.move_hand(arm.side, HAND_OPEN, now, self.p("release_time", 0.6))
                self.message = "surgeon took it" if held else "released (grip not observable)"
            elif el > self.p("grip_timeout", 15.0):
                return self.fail("surgeon did not take the instrument")
        elif self.phase == "release":
            if arm.hands_done() and el > self.p("release_time", 0.6) + 0.3:
                return Status.SUCCESS
        return Status.RUNNING
