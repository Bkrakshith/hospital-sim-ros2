"""Pure tests for the g1-whole-body-rl/exports sidecar extensions: cmd_base_height obs term, joint_limits target clipping,
flat oldest-first history with first-frame reset fill, onnx.file/input/output names. Uses a fake ONNX session."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from g1_controller import policy as P  # noqa: E402

TERMS = [("base_ang_vel", 3), ("projected_gravity", 3), ("cmd_vel", 3), ("cmd_base_height", 1),
         ("upper_body_target_rel", 17), ("joint_pos_rel", 29), ("joint_vel", 29), ("last_action", 29),
         ("gait_phase", 2)]
FRAME = sum(d for _, d in TERMS)  # 116


class _IO:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class FakeSession:
    action = np.zeros(29, np.float32)
    feeds = []

    def __init__(self, path, *a, **k):
        self.H = FakeSession.H

    def get_inputs(self):
        return [_IO("obs_history", [1, self.H * FRAME])]

    def get_outputs(self):
        return [_IO("action", [1, 29])]

    def run(self, _, feeds):
        FakeSession.feeds.append(feeds["obs_history"].copy())
        return [FakeSession.action[None, :]]


def make_policy(tmp_path, monkeypatch, enabled=False, H=10):
    import onnxruntime
    FakeSession.H = H
    FakeSession.feeds = []
    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    lim = [[-0.5, 0.5]] * 29
    side = {
        "name": "fake_ultra", "control_dt": 0.02, "sim_dt": 0.002,
        "joint_names": [f"j{i}" for i in range(29)], "controlled_joints": list(range(29)),
        "default_qpos": [0.1] * 29, "action_scale": [0.25] * 12 + [3.0] * 17,
        "kp": [1.0] * 29, "kd": [0.1] * 29, "torque_limit": [28.0] * 29,
        "obs": [dict({"name": n, "dim": d, "scale": 1.0},
                     **({"joints": "all"} if n in ("joint_pos_rel", "joint_vel") else {}),
                     **({"period": 0.8} if n == "gait_phase" else {})) for n, d in TERMS],
        "history_len": H, "placeholder": False,
        "onnx": {"file": "student.onnx", "input": "obs_history", "output": "action"},
        "joint_limits": lim, "home_pelvis_height": 0.784,
        "command_details": {"base_height": {"enabled": enabled, "trained_range_offset_m": [0, 0]}},
        "robot": {"effort_source": "canonical"},
    }
    (tmp_path / "policy.json").write_text(json.dumps(side))
    return P.PolicyRunner(str(tmp_path))


def _step(p, base_height=None):
    q = np.full(29, 0.1)
    return p.step(q, np.zeros(29), [1, 0, 0, 0], [0, 0, 0], [0, 0, 0], None, base_height)


def test_frame_layout_and_history(tmp_path, monkeypatch):
    p = make_policy(tmp_path, monkeypatch)
    assert p.frame_dim == FRAME and p.obs_dim == 10 * FRAME
    _step(p)
    first = FakeSession.feeds[-1].reshape(10, FRAME)
    assert np.allclose(first, first[0])            # reset-fill with the first frame
    FakeSession.action = np.full(29, 0.2, np.float32)
    _step(p)
    h = FakeSession.feeds[-1].reshape(10, FRAME)
    la = sum(d for n, d in TERMS[:7])               # last_action offset
    assert np.allclose(h[-1, la:la + 29], 0.0)      # newest frame: last_action = previous output (0)
    _step(p)
    h = FakeSession.feeds[-1].reshape(10, FRAME)
    assert np.allclose(h[-1, la:la + 29], 0.2) and np.allclose(h[-2, la:la + 29], 0.0)  # oldest first
    FakeSession.action = np.zeros(29, np.float32)


@pytest.mark.parametrize("enabled,cmd,expect", [(False, 0.70, 0.0), (False, None, 0.0),
                                                (True, 0.70, 0.70 - 0.784), (True, None, 0.0)])
def test_cmd_base_height(tmp_path, monkeypatch, enabled, cmd, expect):
    p = make_policy(tmp_path, monkeypatch, enabled=enabled, H=1)
    _step(p, cmd)
    frame = FakeSession.feeds[-1].reshape(-1)
    assert frame[9] == pytest.approx(expect, abs=1e-6)   # right after cmd_vel (index 6..8)


def test_joint_limits_clip(tmp_path, monkeypatch):
    p = make_policy(tmp_path, monkeypatch, H=1)
    FakeSession.action = np.array([1.0] * 12 + [-1.0] * 17, np.float32)
    t = _step(p)
    # legs: 0.1 + 0.25 = 0.35 (inside), arms: 0.1 - 3.0 -> clipped to -0.5
    assert np.allclose(t[:12], 0.35) and np.allclose(t[12:], -0.5)
    FakeSession.action = np.array([4.0] * 29, np.float32)
    t = _step(p)
    assert np.allclose(t, 0.5)
    FakeSession.action = np.zeros(29, np.float32)


def test_placeholder_has_no_limits_clip():
    pdir = os.path.join(os.path.dirname(__file__), "..", "policies", "unitree_g1_12dof_placeholder")
    assert P.Sidecar(os.path.join(pdir, "policy.json")).joint_limits is None
