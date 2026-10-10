"""Pure tests (no ROS graph): ONNX placeholder reproduces the torch reference rollout; sidecar obs layout;
LowCmd CRC matches Unitree's bitwise algorithm."""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from g1_controller.policy import PolicyRunner, projected_gravity  # noqa: E402
from g1_controller import crc  # noqa: E402

PDIR = os.path.join(os.path.dirname(__file__), "..", "policies", "unitree_g1_12dof_placeholder")


def test_sidecar_and_obs_dim():
    p = PolicyRunner(PDIR)
    assert p.sc.placeholder is True
    assert p.obs_dim == 47
    assert list(p.sc.controlled) == list(range(12))
    assert len(p.sc.default_qpos) == 29


def test_onnx_matches_torch_reference_rollout():
    p = PolicyRunner(PDIR)
    ref = np.load(os.path.join(PDIR, "reference_rollout.npz"))
    for o, a in zip(ref["obs"], ref["actions"]):
        feeds = {p.in_name: o.astype(np.float32)}
        feeds.update(p.state)
        outs = dict(zip([x.name for x in p.sess.get_outputs()], p.sess.run(None, feeds)))
        for r in p.sc.recurrent:
            p.state[r["input"]] = outs[r["output"]]
        assert np.allclose(outs[p.out_name], a, atol=1e-4)


def test_obs_layout_matches_unitree_deploy():
    p = PolicyRunner(PDIR)
    q = p.sc.default_qpos.copy()
    q[3] += 0.1
    dq = np.zeros(29)
    dq[0] = 2.0
    obs = p.build_obs(q, dq, [1, 0, 0, 0], [0.4, 0, 0], [0.5, 0, 0.2], None)
    assert np.allclose(obs[0:3], [0.1, 0, 0])                 # ang vel * 0.25
    assert np.allclose(obs[3:6], [0, 0, -1])                  # gravity
    assert np.allclose(obs[6:9], [(0.5 * 0.4 - 0.13) * 2, 0, 0.2 * 0.25])  # (cmd*scale+offset)*obs scale
    assert np.isclose(obs[9 + 3], 0.1)                        # knee rel
    assert np.isclose(obs[21], 2.0 * 0.05)                    # dq * 0.05
    assert np.allclose(obs[45:47], [0.0, 1.0])                # phase at t = 0


def test_projected_gravity_tilted():
    import math
    a = 0.3
    g = projected_gravity([math.cos(a / 2), 0, math.sin(a / 2), 0])  # pitch a
    assert np.isclose(g[2], -math.cos(a), atol=1e-6)


def test_crc_table_equals_unitree_reference():
    rng = np.random.default_rng(0)
    for _ in range(3):
        words = rng.integers(0, 2 ** 32, 250, dtype=np.uint64).astype(np.uint32)
        b = words.astype("<u4").tobytes()
        assert crc.crc32_words(b) == crc.crc32_core_reference([int(w) for w in words])
    assert crc.SIZE == 1004
