"""scrub_nurse behaviour tree (py_trees) and the mission core that runs it. ROS-free: the node (mission_node.py) and
the offline harness plug in the robot, the navigation client and the mode service.

    scrub_nurse (Sequence)
    |-- EnsureAuto                       request AUTO (source "mission"); robot must be in STAND
    |-- ToReady                          arms to the ready pose (ReturnToReady skill)
    |-- ReachStation (Selector)
    |   |-- AtStation?                   within at_station_tolerance of the station (skip Nav2)
    |   `-- GoTo(theatre_instrument_table)   Nav2 NavigateToPose, map frame
    |-- FineAlign                        last centimetres (StationEstimator: ground truth now = CHEAT)
    `-- Serve (Repeat until aborted, or once with repeat: false)
        `-- ServeOrRecover (Selector)
            |-- ServeOne (Sequence)
            |   |-- WaitForRequest       /g1/mission/request_instrument queue
            |   |-- LocateInstrument     ObjectSource (/sim/objects now = CHEAT; perception later)
            |   |-- Pick                 skill (scripted or learned)
            |   |-- ToSurgeon (Selector)             only if handover_station differs from the station
            |   |   |-- SurgeonInReach?              surgeon hand within handover_reach
            |   |   `-- CarryToSurgeon: Carry (instrument close to the body) ->
            |   |       GoToSurgeon: At(theatre_surgeon_side)? | GoTo + FineAlign   (fixed base: glide)
            |   |-- Handover             skill (scripted or learned)
            |   |-- Retract              skill
            |   |-- ReturnToReady        skill
            |   |-- BackToTable: At(station)? | GoTo + FineAlign                   (only with a handover_station)
            |   `-- Done                 log success
            `-- Recover (Sequence)       a failed request: log it, open the hand, back to ready, serve the next
                |-- LogFailure
                `-- RecoverToReady
    (repeat: false -> ReleaseAuto: AUTO -> STAND)

goto:<location> runs: EnsureAuto -> GoTo(location) (arms at walk_pose) -> [ToReady -> FineAlign if the location has
fine_align] -> ReleaseAuto.
"""
import math
import time as _time
import time

import py_trees
from py_trees.common import Status as PT

from g1_mission.skills import make_skill
from g1_mission.skills.base import Status, drive_arm

DAMP, STAND, AUTO, HOLD = 0, 1, 5, 6
MODE_NAMES = {0: "DAMP", 1: "STAND", 2: "LOCO", 3: "MANIP", 4: "NAV", 5: "AUTO", 6: "HOLD"}
_MAP = {Status.RUNNING: PT.RUNNING, Status.SUCCESS: PT.SUCCESS, Status.FAILURE: PT.FAILURE}


class MissionCore:
    """Everything the behaviours need. `mode_fn()` -> current mode (int or None); `request_mode(mode)` -> object
    with done() and result() -> (accepted, message)."""

    def __init__(self, ctx, config, nav=None, mode_fn=None, request_mode=None, recorder=None, log=print):
        self.ctx, self.cfg, self.nav = ctx, config, nav
        self.mode_fn = mode_fn or (lambda: AUTO)
        self.request_mode = request_mode
        self.recorder = recorder
        self.log = log
        self.queue = []
        self.skill_cfg = config.get("skills", {})
        self.record_skills = set(config.get("record_skills", ["pick", "handover"]))
        self.history = []
        self.current = {}
        self.last_failure = ""

    def rate(self):
        return float(self.cfg.get("rate", 50.0))

    def station(self):
        return self.ctx.locations.get(self.cfg.get("station", "theatre_instrument_table"))


# ---------------------------------------------------------------------------------------------------- behaviours
class _Base(py_trees.behaviour.Behaviour):
    def __init__(self, name, core):
        super().__init__(name)
        self.core = core

    def hold_arm(self):
        """Keep streaming the current arm/hand targets (the controller drops stale targets after 0.5 s)."""
        if self.core.ctx.arm is not None:
            drive_arm(self.core.ctx, 1.0 / self.core.rate())


class EnsureAuto(_Base):
    def initialise(self):
        self.fut, self.t0 = None, time.monotonic()

    def update(self):
        m = self.core.mode_fn()
        if m == AUTO:
            self.feedback_message = "AUTO"
            return PT.SUCCESS
        if m != STAND:
            self.feedback_message = f"robot must be in STAND to start (mode {MODE_NAMES.get(m, m)}; press r first)"
            return PT.FAILURE
        if self.fut is None:
            self.fut = self.core.request_mode(AUTO)
        elif self.fut.done():
            ok, msg = self.fut.result()
            if not ok:
                self.feedback_message = f"AUTO rejected: {msg}"
                return PT.FAILURE
        if time.monotonic() - self.t0 > 5.0:
            self.feedback_message = "no answer from the mode manager (/g1/set_mode)"
            return PT.FAILURE
        return PT.RUNNING


class ReleaseAuto(_Base):
    def initialise(self):
        self.fut = None

    def update(self):
        if self.core.mode_fn() != AUTO:
            return PT.SUCCESS
        if self.fut is None:
            self.fut = self.core.request_mode(STAND)
        return PT.SUCCESS if self.fut.done() else PT.RUNNING


class AtLocation(_Base):
    def __init__(self, name, core, location, tol):
        super().__init__(name, core)
        self.location, self.tol = location, tol

    def update(self):
        loc = self.core.ctx.locations.get(self.location)
        T = self.core.ctx.robot.base_T()
        if loc is None or T is None:
            return PT.FAILURE
        d = math.hypot(loc["x"] - T[0, 3], loc["y"] - T[1, 3])
        self.feedback_message = f"{d:.2f} m from {self.location}"
        return PT.SUCCESS if d < self.tol else PT.FAILURE


class GoTo(_Base):
    """Nav2 NavigateToPose to a named location. arms="walk": the arms first go to `walk_pose` (the locomotion
    policy's default: the placeholder was trained with its arms there, and the raised ready pose makes it drift
    backwards at ~0.1 m/s); arms="hold": keep the current arm targets (carrying an instrument)."""

    def __init__(self, name, core, location, arms="walk"):
        super().__init__(name, core)
        self.location = location
        self.arms = arms

    def initialise(self):
        self.t0 = time.monotonic()
        self.sent = False
        self.paused = False
        ctx = self.core.ctx
        walk = self.core.cfg.get("walk_pose")
        exact = getattr(self.core.nav, "exact", lambda: False)()
        if self.arms == "walk" and walk is not None and not exact and ctx.arm is not None \
                and ctx.robot.upper_q() is not None:
            import numpy as np
            ctx.arm.init(ctx.robot.upper_q())
            ctx.arm.move_joints(np.array(walk, float), ctx.robot.now(), max_vel=0.7)

    def _send(self):
        loc = self.core.ctx.locations[self.location]
        x, y, yaw = loc["x"], loc["y"], loc["yaw"]
        # Nav2 stops within its goal tolerance (0.2 m): aim `approach_back` m short of a fine-align station so the
        # robot never overshoots into the table; FineAlign closes the rest. The fixed-base glide is exact.
        exact = getattr(self.core.nav, "exact", lambda: False)()
        back = 0.0 if exact or not loc.get("fine_align") else float(self.core.cfg.get("approach_back", 0.25))
        x, y = x - back * math.cos(yaw), y - back * math.sin(yaw)
        self.core.nav.go(x, y, yaw)
        self.sent = True

    def update(self):
        nav = self.core.nav
        loc = self.core.ctx.locations.get(self.location)
        if loc is None:
            self.feedback_message = f"unknown location {self.location}"
            return PT.FAILURE
        if nav is None:
            self.feedback_message = "no navigation client (Nav2 not configured)"
            return PT.FAILURE
        self.hold_arm()
        if not self.sent:
            if not nav.available():
                if time.monotonic() - self.t0 > 15.0:
                    self.feedback_message = "Nav2 /navigate_to_pose not available (start Nav2 + localisation)"
                    return PT.FAILURE
                self.feedback_message = "waiting for Nav2"
                return PT.RUNNING
            self._send()
        st = nav.state
        rem = nav.distance_remaining()
        self.feedback_message = f"Nav2 {st}" + (f", {rem:.1f} m to go" if rem is not None else "")
        if st == "succeeded":
            return PT.SUCCESS
        if st in ("failed", "rejected"):
            self.feedback_message = nav.msg or f"Nav2 {st}"
            self.core.last_failure = f"{self.name}: {self.feedback_message}"
            return PT.FAILURE
        if time.monotonic() - self.t0 > float(self.core.cfg.get("nav_goal_timeout", 300.0)):
            nav.cancel()
            self.feedback_message = "Nav2 goal timeout"
            self.core.last_failure = f"{self.name}: {self.feedback_message}"
            return PT.FAILURE
        return PT.RUNNING

    def pause(self):
        if self.core.nav is not None and self.sent:
            self.core.nav.cancel()
        self.paused = True

    def resume(self, dt):
        self.t0 += dt
        if self.paused and self.sent:
            self._send()
        self.paused = False

    def terminate(self, new_status):
        if new_status == PT.INVALID and self.core.nav is not None and self.sent and \
                self.core.nav.state in ("pending", "active"):
            self.core.nav.cancel()


class WaitForRequest(_Base):
    def update(self):
        self.hold_arm()
        q = self.core.queue
        if q:
            inst = q.pop(0)
            self.core.ctx.board.clear()
            self.core.ctx.board["instrument"] = inst
            self.core.current = {"instrument": inst, "t0": self.core.ctx.robot.now()}
            self.core.last_failure = ""
            self.feedback_message = f"request: {inst}"
            self.core.log(f"serving request: {inst}")
            return PT.SUCCESS
        self.feedback_message = "waiting for an instrument request"
        return PT.RUNNING


class SkillBehaviour(_Base):
    """Runs a skill from the library (scripted or learned, config/mission.yaml `skills`)."""

    def __init__(self, name, core, skill_name, goal=None):
        super().__init__(name, core)
        self.skill_name = skill_name
        self.goal = goal or {}
        self.skill = None
        self.recording = False

    def initialise(self):
        core = self.core
        self.skill = make_skill(self.skill_name, core.skill_cfg.get(self.skill_name, {}))
        goal = dict(self.goal)
        goal.setdefault("instrument", core.ctx.board.get("instrument"))
        rec = core.recorder
        self.recording = rec is not None and self.skill_name in core.record_skills
        if self.recording:
            rec.start(f"{self.skill_name} {goal.get('instrument') or ''}".strip())
        self.skill.start(core.ctx, **goal)

    def update(self):
        st = self.skill.tick(self.core.ctx)
        self.feedback_message = " ".join(v for v in (self.skill.phase, self.skill.message) if v)
        if st != Status.RUNNING:
            self._close(st == Status.SUCCESS)
            if st == Status.FAILURE:
                self.core.last_failure = f"{self.name}: {self.skill.message}"
                self.core.log(f"{self.name} failed: {self.skill.message}")
        return _MAP[st]

    def _close(self, success):
        if self.recording and self.core.recorder is not None:
            self.core.recorder.stop(success)
        self.recording = False

    def terminate(self, new_status):
        # only a RUNNING skill is interrupted. py_trees also resets finished behaviours to INVALID (a memory Selector
        # clears the failed ServeOne branch once Recover runs); stopping those froze the shared ArmMotion under the
        # skill that was running then (RecoverToReady crashed on arm.seg = None). self.status is still the old one.
        if new_status == PT.INVALID and self.skill is not None and self.status == PT.RUNNING:
            self.skill.stop(self.core.ctx)
            self._close(False)

    def pause(self):
        if self.skill is not None:
            self.skill.stop(self.core.ctx) if self.skill_name == "fine_align" else None

    def resume(self, dt):
        if self.skill is not None:
            self.skill.shift_time(dt)
            if self.core.ctx.arm is not None:
                self.core.ctx.arm.shift_time(dt)


class RecordResult(_Base):
    """End of one served request: log the result (the failure branch records failures)."""

    def __init__(self, name, core, success=True):
        super().__init__(name, core)
        self.success = success

    def update(self):
        c = self.core
        if c.current:
            dur = c.ctx.robot.now() - c.current.get("t0", c.ctx.robot.now())
            c.history.append({"instrument": c.current.get("instrument"), "success": self.success,
                              "duration": round(dur, 1), "t": round(_time.time(), 1),
                              "message": "handed over" if self.success else (c.last_failure or "failed")})
            c.history = c.history[-50:]
            c.current = {}
        c.ctx.board.pop("instrument", None)
        return PT.SUCCESS


# ---------------------------------------------------------------------------------------------------- trees
def _to_ready(core):
    return SkillBehaviour("ToReady", core, "return_to_ready")


class SurgeonInReach(_Base):
    """Condition: the surgeon's hand is within `handover_reach` of the pelvis (no need to move for the handover)."""

    def update(self):
        ctx = self.core.ctx
        hand = ctx.surgeon.get() if ctx.surgeon else None
        T = ctx.robot.base_T()
        if hand is None or T is None:
            self.feedback_message = "no surgeon hand estimate"
            return PT.FAILURE
        d = math.hypot(hand.T[0, 3] - T[0, 3], hand.T[1, 3] - T[1, 3])
        self.feedback_message = f"surgeon hand {d:.2f} m away"
        return PT.SUCCESS if d < float(self.core.cfg.get("handover_reach", 0.70)) else PT.FAILURE


def _go_and_align(core, name, location, arms="walk"):
    """Selector: already there? else GoTo + [arms back to ready] + FineAlign (if the location is a fine-align
    station)."""
    loc = core.ctx.locations.get(location, {})
    seq = [GoTo(f"GoTo({location})", core, location, arms)]
    if loc.get("fine_align"):
        align = SkillBehaviour(f"FineAlign({location})", core, "fine_align", {"station": loc})
        ready = SkillBehaviour(f"ReadyAt({location})", core, "return_to_ready")
        exact = getattr(core.nav, "exact", lambda: False)()
        if arms == "walk":
            seq += [ready, align] if exact else [align, ready]
        else:
            seq.append(align)
    return py_trees.composites.Selector(name, memory=True, children=[
        AtLocation(f"At({location})?", core, location, float(core.cfg.get("at_location_tolerance", 0.06))),
        py_trees.composites.Sequence(f"MoveTo({location})", memory=True, children=seq)])


def build_scrub_nurse(core):
    cfg = core.cfg
    station = cfg.get("station", "theatre_instrument_table")
    reach = py_trees.composites.Selector("ReachStation", memory=True, children=[
        AtLocation("AtStation?", core, station, float(cfg.get("at_station_tolerance", 0.30))),
        GoTo(f"GoTo({station})", core, station)])
    hstation = cfg.get("handover_station", "")
    serve = [WaitForRequest("WaitForRequest", core),
             SkillBehaviour("LocateInstrument", core, "locate"),
             SkillBehaviour("Pick", core, "pick")]
    if hstation and hstation in core.ctx.locations and hstation != station:
        # the surgeon stands away from the instrument table: carry the instrument over (Nav2, or the fixed-base glide)
        serve.append(py_trees.composites.Selector("ToSurgeon", memory=True, children=[
            SurgeonInReach("SurgeonInReach?", core),
            py_trees.composites.Sequence("CarryToSurgeon", memory=True, children=[
                SkillBehaviour("Carry", core, "carry"), _go_and_align(core, "GoToSurgeon", hstation, "hold")])]))
    serve += [SkillBehaviour("Handover", core, "handover"),
              SkillBehaviour("Retract", core, "retract"),
              SkillBehaviour("ReturnToReady", core, "return_to_ready")]
    if hstation and hstation in core.ctx.locations and hstation != station:
        serve.append(_go_and_align(core, "BackToTable", station))
    serve.append(RecordResult("Done", core))
    serve_one = py_trees.composites.Sequence("ServeOne", memory=True, children=serve)
    # a failed request is logged, the hand opened and the arm returned to ready; then the next request is served
    rec = [RecordResult("LogFailure", core, success=False), SkillBehaviour("RecoverToReady", core, "return_to_ready")]
    if hstation and hstation in core.ctx.locations and hstation != station:
        rec.append(_go_and_align(core, "RecoverToTable", station))
    recover = py_trees.composites.Sequence("Recover", memory=True, children=rec)
    serve = py_trees.composites.Selector("ServeOrRecover", memory=True, children=[serve_one, recover])
    # fixed base (glide): raise the arms before gliding in (policy-default hands would sit in the rack). Walking:
    # align first with the arms at walk_pose (raised arms make the placeholder policy drift backwards), then raise.
    align = SkillBehaviour("FineAlign", core, "fine_align", {"station": core.station()})
    exact = getattr(core.nav, "exact", lambda: False)()
    children = [EnsureAuto("EnsureAuto", core), reach] + ([_to_ready(core), align] if exact else [align, _to_ready(core)])
    if cfg.get("repeat", True):
        children.append(py_trees.decorators.Repeat("Serve", child=serve, num_success=-1))
    else:
        children += [serve, ReleaseAuto("ReleaseAuto", core)]
    return py_trees.composites.Sequence("scrub_nurse", memory=True, children=children)


def build_goto(core, location):
    loc = core.ctx.locations.get(location, {})
    children = [EnsureAuto("EnsureAuto", core), GoTo(f"GoTo({location})", core, location)]
    if loc.get("fine_align"):
        align = SkillBehaviour("FineAlign", core, "fine_align", {"station": loc})
        exact = getattr(core.nav, "exact", lambda: False)()
        children += [_to_ready(core), align] if exact else [align, _to_ready(core)]
    children.append(ReleaseAuto("ReleaseAuto", core))
    return py_trees.composites.Sequence(f"goto_{location}", memory=True, children=children)


def tree_status(root):
    """[{name, status, depth, type, message}] in depth-first order (for /g1/mission/status)."""
    out = []

    def walk(b, depth):
        out.append({"name": b.name, "status": b.status.value if b.status != PT.INVALID else "IDLE",
                    "depth": depth, "type": type(b).__name__, "message": b.feedback_message or ""})
        for c in b.children:
            walk(c, depth + 1)
    walk(root, 0)
    return out


def active_behaviours(root):
    return [b for b in root.iterate() if b.status == PT.RUNNING]
