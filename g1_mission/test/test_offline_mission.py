"""Offline (no ROS) mission tests: the real scrub_nurse tree + scripted skills on the in-process MuJoCo harness
(fixed base at the instrument table). Slow-ish (~10 s each): they run the physics."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("mujoco")
pytest.importorskip("pinocchio")
pytest.importorskip("py_trees")
from py_trees.common import Status as PT  # noqa: E402

from g1_mission.offline import OfflineNav, OfflineRobot, make_context  # noqa: E402
from g1_mission.tree import AUTO, MissionCore, build_scrub_nurse, tree_status  # noqa: E402


class _Done:
    def __init__(self, ok=True):
        self.ok = ok

    def done(self):
        return True

    def result(self):
        return self.ok, "ok"


@pytest.fixture(scope="module")
def rig():
    robot = OfflineRobot()
    ctx = make_context(robot)
    robot.advance(0.5)
    return robot, ctx


def run_tree(robot, ctx, queue, until_history, max_t=90.0, repeat=True):
    cfg = dict(robot.mission_cfg, rate=50.0, repeat=repeat)
    modes = {"m": AUTO}
    core = MissionCore(ctx, cfg, nav=OfflineNav(robot), mode_fn=lambda: modes["m"],
                       request_mode=lambda m: (modes.__setitem__("m", m), _Done())[1], log=lambda s: None)
    core.queue.extend(queue)
    root = build_scrub_nurse(core)
    root.setup_with_descendants()
    t0 = robot.now()
    while robot.now() - t0 < max_t:
        root.tick_once()
        robot.advance(0.02)
        if len(core.history) >= until_history or root.status in (PT.FAILURE, PT.SUCCESS):
            break
    return core, root


def _two(ctx):
    """Two instruments that can be pulled straight out of the rack (scripts/check_extraction.py): with the
    current hospital-sim-env/objects rack, scissors and angled_tweezers jam in the guide holes."""
    names = list(ctx.catalog)
    second = "adson_forceps" if "adson_forceps" in names else "forceps"
    return ("scalpel_handle", second) if second in names else (names[0], names[1])


def test_serves_two_requests_handle_first(rig):
    robot, ctx = rig
    robot.reset_objects()
    robot.advance(0.3)
    a, b = _two(ctx)
    core, root = run_tree(robot, ctx, [a, b], until_history=2, max_t=150.0)
    assert [h["success"] for h in core.history] == [True, True], core.history
    snap = robot.objects.snapshot(robot.d)
    # the second instrument is the one the surgeon holds now (the first went back to the rack when it was offered)
    assert snap["surgeon_hand"]["holding"] == b
    names = [n["name"] for n in tree_status(root)]
    for n in ("EnsureAuto", "AtStation?", "FineAlign", "WaitForRequest", "Pick", "Handover", "Retract",
              "ReturnToReady"):
        assert n in names


def test_failed_request_is_logged_and_recovered(rig):
    robot, ctx = rig
    robot.reset_objects()
    robot.advance(0.3)
    ctx.catalog["ghost_clamp"] = {"grasp_z": 0.09, "handle_z": 0.16}   # in the catalogue, not in the world
    try:
        core, root = run_tree(robot, ctx, ["ghost_clamp"], until_history=1, max_t=30.0)
    finally:
        del ctx.catalog["ghost_clamp"]
    assert len(core.history) == 1 and not core.history[0]["success"]
    assert "not found" in core.history[0]["message"]
    assert root.status == PT.RUNNING          # still serving (waits for the next request)


def test_offsets_rack(rig):
    robot, ctx = rig
    robot.reset_objects(offset=(0.015, -0.015, 0.06))
    robot.advance(0.3)
    core, _ = run_tree(robot, ctx, [_two(ctx)[1]], until_history=1, max_t=90.0)
    assert core.history and core.history[0]["success"], core.history


def test_mixed_sequence_without_reset(rig):
    """The app's session: requests one after the other, no /sim/reset_objects in between, so the surgeon still holds
    the previous instrument when the next arrives. needle_driver -> scissors -> angled_tweezers failed the tweezers
    ("surgeon did not take the instrument": the handle jammed against the held scissors 5-9 cm from the hand) until
    the sim's surgeon put the held instrument down when the next one is offered (g1_mujoco_sim.scene_objects)."""
    robot, ctx = rig
    names = list(ctx.catalog)
    seq = ["needle_driver", "scissors", "angled_tweezers"]
    if not all(n in names for n in seq):
        pytest.skip("hospital-sim-env/objects layout only")
    robot.reset_objects()
    robot.advance(0.3)
    core, _ = run_tree(robot, ctx, seq, until_history=3, max_t=220.0)
    assert [h["success"] for h in core.history] == [True, True, True], core.history
    assert robot.objects.snapshot(robot.d)["surgeon_hand"]["holding"] == "angled_tweezers"
