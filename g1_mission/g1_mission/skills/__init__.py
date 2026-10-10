"""Skill library. `make_skill(name, cfg)` builds the implementation configured in config/mission.yaml:

    skills:
      pick:     {impl: scripted, ...params}          # ScriptedPick
      handover: {impl: policy, policy_path: ..., fallback: scripted}   # a learned policy (LeRobot ACT/Diffusion)

Every implementation follows skills.base.Skill (start / tick / stop) and acts only through ctx.robot.command.
"""
from g1_mission.skills.base import Skill, Status, SkillContext, ArmMotion  # noqa: F401
from g1_mission.skills.pick import ScriptedPick
from g1_mission.skills.handover import ScriptedHandover
from g1_mission.skills.retract import Carry, Retract, ReturnToReady
from g1_mission.skills.fine_align import FineAlign
from g1_mission.skills.locate import LocateInstrument
from g1_mission.skills.policy import LearnedPolicySkill

SCRIPTED = {
    "pick": ScriptedPick,
    "handover": ScriptedHandover,
    "retract": Retract,
    "return_to_ready": ReturnToReady,
    "carry": Carry,
    "fine_align": FineAlign,
    "locate": LocateInstrument,
}


def make_skill(name, cfg=None):
    cfg = dict(cfg or {})
    impl = cfg.pop("impl", "scripted")
    if impl == "scripted":
        return SCRIPTED[name](**cfg)
    if impl == "policy":
        fallback = cfg.pop("fallback", None)
        return LearnedPolicySkill(name, fallback=SCRIPTED[name](**cfg.pop("fallback_params", {})) if fallback else None,
                                  **cfg)
    raise ValueError(f"unknown skill impl '{impl}' for {name}")
