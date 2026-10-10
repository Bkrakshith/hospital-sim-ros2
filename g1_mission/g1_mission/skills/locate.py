"""LocateInstrument: find the requested instrument through the ObjectSource (step 1: /sim/objects ground truth =
CHEAT; later /g1/perception/detections) and check it is available (in reach, not already with the surgeon)."""
import numpy as np

from g1_mission.geometry import inv
from g1_mission.skills.base import Skill, Status


class LocateInstrument(Skill):
    name = "locate"

    def start(self, ctx, instrument=None, **goal):
        super().start(ctx)
        self.inst = instrument or ctx.board.get("instrument")
        self.phase = "locate"

    def tick(self, ctx):
        est = ctx.objects.get(self.inst) if ctx.objects else None
        if est is None:
            if ctx.robot.now() - self.t0 > self.p("timeout", 5.0):
                return self.fail(f"{self.inst} not found")
            return Status.RUNNING
        if est.extra.get("held_by") == "surgeon":
            return self.fail(f"{self.inst} is already with the surgeon")
        p = (inv(ctx.robot.base_T()) @ est.T)[:3, 3]
        if np.linalg.norm(p[:2]) > self.p("max_range", 0.75):
            return self.fail(f"{self.inst} out of reach ({np.linalg.norm(p[:2]):.2f} m from the pelvis)")
        ctx.board["instrument_T"] = est.T
        self.message = f"{self.inst} at {p[0]:.2f} {p[1]:+.2f} {p[2]:+.2f} (pelvis frame, {est.source})"
        return Status.SUCCESS
