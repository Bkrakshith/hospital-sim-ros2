"""Unit tests for the skill library pieces (no ROS graph, no physics except where noted)."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from g1_mission import geometry as G  # noqa: E402
from g1_mission.config import load_catalog, load_locations, load_mission  # noqa: E402
from g1_mission.interfaces import (DetectionObjects, GroundTruthObjects, GroundTruthStation,  # noqa: E402
                                   GroundTruthSurgeonHand)


def test_quat_roundtrip_and_interp():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        assert np.allclose(G.mat_to_quat(G.quat_to_mat(q)), q, atol=1e-9)
    A = G.make_T(G.rot_z(0.0), [0, 0, 0])
    B = G.make_T(G.rot_z(1.0), [1, 2, 3])
    M = G.interp_T(A, B, 0.5)
    assert np.allclose(M[:3, 3], [0.5, 1, 1.5]) and math.isclose(math.atan2(M[1, 0], M[0, 0]), 0.5, abs_tol=1e-9)
    assert np.allclose(G.inv(B) @ B, np.eye(4))


def test_configs_consistent():
    cat, locs, cfg = load_catalog(), load_locations(), load_mission()
    assert len(cat) >= 2 and "scalpel_handle" in cat          # stand-in and hospital-sim-env/objects layouts both have it
    for name, e in cat.items():
        assert 0 < e["grasp_z"] < e["handle_z"], name
        assert e.get("approach_axis", "y") in ("x", "y"), name
    for req in ("theatre_instrument_table", "theatre_surgeon_side", "sterile_store", "corridor_west"):
        assert req in locs
    assert locs["theatre_instrument_table"].get("fine_align")
    assert len(cfg["ready_pose"]) == 17
    for sk in ("pick", "handover", "retract", "return_to_ready", "fine_align", "locate"):
        assert sk in cfg["skills"]


def test_ground_truth_sources():
    gt = GroundTruthObjects()
    gt.update(json.dumps({"forceps": {"pose": [1, 2, 0.8, 1, 0, 0, 0], "type": "instrument", "held_by": None},
                          "surgeon_hand": {"pose": [3, 4, 1.0, 1, 0, 0, 0], "type": "surgeon_hand",
                                           "holding": None}}))
    assert gt.names() == ["forceps"]
    assert np.allclose(gt.get("forceps").T[:3, 3], [1, 2, 0.8])
    hand = GroundTruthSurgeonHand(gt)
    assert np.allclose(hand.get().T[:3, 3], [3, 4, 1.0]) and hand.holding("forceps") is False
    gt.data["forceps"]["held_by"] = "surgeon"
    assert hand.holding("forceps") is True
    assert hand.holding("nothing") is None


def test_detection_source_parses_contract_json():
    det = DetectionObjects(to_map=lambda frame, T: G.make_T(np.eye(3), [10, 0, 0]) @ T if frame == "odom" else None)
    det.update(json.dumps([
        {"label": "scissors", "score": 0.9, "bbox_obb": [320, 240, 40, 12, 0.3],
         "pose": {"frame": "map", "xyz": [1.0, 2.0, 0.8], "yaw": 0.5}},
        {"label": "scissors", "score": 0.6, "pose": {"frame": "map", "xyz": [9, 9, 9], "yaw": 0}},
        {"label": "forceps", "score": 0.8, "pose": {"frame": "odom", "xyz": [1, 0, 0], "yaw": 0}},
        {"label": "trocar", "score": 0.95, "pose": {"frame": "camera", "xyz": [0, 0, 1], "yaw": 0}},
        {"label": "weak", "score": 0.1, "pose": {"frame": "map", "xyz": [0, 0, 0], "yaw": 0}}]))
    e = det.get("scissors")
    assert e.source == "perception" and np.allclose(e.T[:3, 3], [1, 2, 0.8]) and e.extra["score"] == 0.9
    assert np.allclose(det.get("forceps").T[:3, 3], [11, 0, 0])
    assert det.get("trocar") is None and det.get("weak") is None


def test_ground_truth_station_error_in_body_frame():
    T = G.make_T(G.rot_z(-math.pi / 2), [26.0, -19.0, 0.78])        # facing south
    st = GroundTruthStation(lambda: T)
    dx, dy, dyaw = st.error({"x": 26.0, "y": -19.2, "yaw": -math.pi / 2})
    assert math.isclose(dx, 0.2, abs_tol=1e-9) and abs(dy) < 1e-9 and abs(dyaw) < 1e-9   # 0.2 m straight ahead
    dx, dy, dyaw = st.error({"x": 26.1, "y": -19.0, "yaw": -math.pi / 2 + 0.1})
    assert math.isclose(dy, 0.1, abs_tol=1e-9) and math.isclose(dyaw, 0.1, abs_tol=1e-9)  # east = robot's left


def test_skill_registry_and_policy_fallback():
    from g1_mission.skills import SCRIPTED, make_skill
    from g1_mission.skills.base import Skill
    from g1_mission.skills.policy import LearnedPolicySkill
    for name, cls in SCRIPTED.items():
        assert issubclass(cls, Skill) and cls.name == name
    sk = make_skill("pick", {"impl": "policy", "policy_path": "", "fallback": "scripted"})
    assert isinstance(sk, LearnedPolicySkill) and sk.fallback is not None and sk.fallback.name == "pick"
    with pytest.raises(ValueError):
        make_skill("pick", {"impl": "nonsense"})


def test_finished_skill_reset_does_not_freeze_the_running_skill():
    """py_trees resets finished behaviours with stop(INVALID) (a memory Selector clears the failed ServeOne branch on
    the tick after Recover started). A finished skill must not then call skill.stop(): it froze the shared ArmMotion
    (seg = None) under the running RecoverToReady, whose tick crashed on arm.seg["t0"] and killed mission_node
    (measured after a failed handover). A behaviour interrupted while RUNNING must still stop its skill."""
    pytest.importorskip("py_trees")
    from py_trees.common import Status as PT
    from g1_mission.tree import SkillBehaviour

    class _Skill:
        stops = 0

        def stop(self, ctx):
            _Skill.stops += 1

    class _Core:
        ctx, recorder = None, None

    for was, expect in ((PT.SUCCESS, 0), (PT.FAILURE, 0), (PT.RUNNING, 1)):
        _Skill.stops = 0
        b = SkillBehaviour("Handover", _Core(), "handover")
        b.skill, b.status = _Skill(), was
        b.stop(PT.INVALID)
        assert _Skill.stops == expect, was


def test_ik_reaches_fk_poses():
    pytest.importorskip("pinocchio")
    from g1_mission.kinematics import UpperBodyIK
    ik = UpperBodyIK(side="right")
    rng = np.random.default_rng(3)
    q0 = np.zeros(17)
    for _ in range(10):
        q = q0.copy()
        q[10:17] = rng.uniform(-0.6, 0.6, 7)
        q[13] = rng.uniform(-0.5, 0.8)
        T = ik.fk(q)
        qs, pe, re = ik.solve(T, q0, rest=q0, iters=300)
        assert pe < 2e-3 and re < 2e-2, (pe, re)


def test_recorder_writes_lerobot_v21(tmp_path):
    pa = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("imageio")
    from g1_mission.recorder import LeRobotRecorder

    class R:
        t = 0.0

        def now(self):
            return self.t

        def upper_q(self):
            return np.full(17, 0.1)

        def hand_q(self, side):
            return np.full(6, 0.5)

    rob = R()
    rec = LeRobotRecorder(str(tmp_path), fps=10, image_size=(48, 64))
    rec.start("pick scissors")
    for k in range(25):
        rob.t = k * 0.05
        rec.action(upper=np.full(17, 0.2), right_hand=np.ones(6), cmd_vel=(0.1, 0, 0))
        rec.frame(rob, image=np.zeros((48, 64, 3), np.uint8))
    idx = rec.stop(success=True)
    assert idx == 0
    info = json.load(open(tmp_path / "meta" / "info.json"))
    assert info["codebase_version"] == "v2.1" and info["total_episodes"] == 1
    assert info["features"]["observation.state"]["shape"] == [29]
    assert info["features"]["action"]["shape"] == [32]
    t = pa.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    assert t.num_rows == 13                                   # 1.2 s at 10 fps (+ frame 0)
    a = np.array(t.column("action").to_pylist()[-1])
    assert np.allclose(a[:17], 0.2) and np.allclose(a[23:29], 1.0) and np.isclose(a[29], 0.1)
    assert (tmp_path / "videos" / "chunk-000" / "observation.images.head" / "episode_000000.mp4").exists()
    tasks = [json.loads(line) for line in open(tmp_path / "meta" / "tasks.jsonl")]
    assert tasks == [{"task_index": 0, "task": "pick scissors"}]
