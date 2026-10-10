"""LearnedPolicySkill: run an imitation-learned policy (LeRobot ACT / Diffusion Policy, trained on episodes written
by g1_mission.recorder) in place of a scripted skill. Same Skill interface, same action channels.

Observation (recorder.py, LeRobot keys): observation.images.head (H, W, 3 uint8), observation.state (29,).
Action (32,): upper body 17 (waist 3, left arm 7, right arm 7) + left hand 6 + right hand 6 + cmd_vel 3.

Loading: `policy_path` = a LeRobot pretrained-policy directory (config.json + model.safetensors); needs the
`lerobot` package (not installed in .venv yet: pip install lerobot). Success is decided by the same checks the
scripted skill uses (`success` = "pick" | "handover"), or by `max_time`.
Until a policy exists the configured `fallback` scripted skill runs instead (logged).
"""
import numpy as np

from g1_mission.recorder import action_vector, state_vector, split_action
from g1_mission.skills.base import Skill, Status


class LearnedPolicySkill(Skill):
    def __init__(self, name, policy_path="", fallback=None, max_time=30.0, device="cuda", **params):
        super().__init__(**params)
        self.name = name
        self.policy_path, self.fallback, self.max_time, self.device = policy_path, fallback, max_time, device
        self.policy = None
        self.active = None

    def _load(self):
        if self.policy is not None or not self.policy_path:
            return self.policy
        try:
            import torch  # noqa: F401
            from lerobot.common.policies.factory import get_policy_class  # lerobot >= 0.2
            import json
            import os
            with open(os.path.join(self.policy_path, "config.json")) as f:
                typ = json.load(f).get("type", "act")
            self.policy = get_policy_class(typ).from_pretrained(self.policy_path).to(self.device).eval()
        except Exception as e:  # noqa: BLE001
            self.message = f"policy load failed ({e})"
            self.policy = None
        return self.policy

    def start(self, ctx, **goal):
        super().start(ctx)
        if self._load() is None:
            if self.fallback is None:
                self.active = None
                return
            ctx.log(f"{self.name}: no learned policy ({self.message or 'policy_path empty'}); using scripted fallback")
            self.active = self.fallback
            self.fallback.start(ctx, **goal)
            return
        self.active = self
        self.policy.reset()
        self.goal = goal

    def tick(self, ctx):
        if self.active is None:
            return self.fail(f"{self.name}: no policy and no fallback")
        if self.active is not self:
            st = self.active.tick(ctx)
            self.phase, self.message = f"fallback:{self.active.phase}", self.active.message
            return st
        import torch
        img = ctx.robot.head_image()
        st = state_vector(ctx.robot)
        if img is None or st is None:
            return Status.RUNNING
        obs = {"observation.state": torch.from_numpy(st).float()[None].to(self.device),
               "observation.images.head": torch.from_numpy(img).permute(2, 0, 1).float()[None].to(self.device) / 255.0}
        with torch.no_grad():
            a = self.policy.select_action(obs)[0].cpu().numpy()
        cmd = split_action(a)
        ctx.robot.command(**cmd)
        if ctx.recorder is not None:
            ctx.recorder.action(**cmd)
        if ctx.robot.now() - self.t0 > self.max_time:
            return Status.SUCCESS if self.p("success_on_timeout", False) else self.fail("policy timeout")
        return Status.RUNNING

    def stop(self, ctx):
        if self.active is not None and self.active is not self:
            self.active.stop(ctx)


__all__ = ["LearnedPolicySkill", "action_vector", "state_vector", "np"]
