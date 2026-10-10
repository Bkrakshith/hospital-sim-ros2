"""Policy runner: ONNX + JSON sidecar (docs/interfaces.md section 4). No ROS imports here.

The observation is built from the sidecar's "obs" list only. Supported terms (name -> meaning):
  base_ang_vel        pelvis IMU gyro, body frame                         dim 3
  projected_gravity   gravity unit vector in the pelvis frame             dim 3
  cmd_vel             (vx, vy, wz) command * cmd_vel_scale + cmd_vel_offset   dim 3
                      (optional sidecar keys, default 1 / 0; used to calibrate the placeholder)
  joint_pos_rel       q - default_qpos  for "joints" (controlled | all)   dim = len(joints)
  joint_pos           q (absolute)      for "joints"                      dim = len(joints)
  joint_vel           dq                for "joints"                      dim = len(joints)
  last_action         previous raw policy output                         dim = len(controlled_joints)
  gait_phase          [sin, cos](2 pi t / period)                         dim 2
  cmd_base_height     commanded pelvis height - sidecar home_pelvis_height dim 1
                      (0 unless sidecar command_details.base_height.enabled)
  upper_body_target   waist(3) + arms(14) target joint positions          dim 17
  upper_body_target_rel  same, minus default_qpos                        dim 17
Each term is multiplied by "scale" (scalar or per-element list) and optionally clipped by "clip".
If history_len > 1 the last H frames are concatenated oldest first.
Recurrent ONNX state (sidecar onnx.recurrent) is fed back each step and zeroed by reset().
Joint targets = default + action * action_scale, then clipped to sidecar "joint_limits" ([29][lo, hi]) if given.
"""
import json
import os

import numpy as np

UPPER_BODY = list(range(12, 29))  # waist 12..14, arms 15..28


def projected_gravity(quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    return np.array([2 * (-qz * qx + qw * qy),
                     -2 * (qz * qy + qw * qx),
                     1 - 2 * (qw * qw + qz * qz)], dtype=np.float32)


class Sidecar:
    def __init__(self, path):
        with open(path) as f:
            self.raw = json.load(f)
        r = self.raw
        self.name = r["name"]
        self.placeholder = bool(r.get("placeholder", False))
        self.control_dt = float(r["control_dt"])
        self.sim_dt = float(r.get("sim_dt", 0.002))
        self.joint_names = list(r["joint_names"])
        if len(self.joint_names) != 29:
            raise ValueError("sidecar joint_names must list the 29 body joints")
        self.controlled = np.array(r["controlled_joints"], dtype=int)
        self.default_qpos = np.array(r["default_qpos"], dtype=np.float32)
        a = r["action_scale"]
        self.action_scale = (np.full(len(self.controlled), a, np.float32) if np.isscalar(a)
                             else np.array(a, np.float32))
        self.kp = np.array(r["kp"], np.float32)
        self.kd = np.array(r["kd"], np.float32)
        self.torque_limit = np.array(r["torque_limit"], np.float32)
        self.obs_terms = r["obs"]
        self.history_len = int(r.get("history_len", 1))
        self.action_clip = r.get("action_clip")
        self.cmd_vel_offset = np.array(r.get("cmd_vel_offset", [0.0, 0.0, 0.0]), np.float32)
        self.cmd_vel_scale = np.array(r.get("cmd_vel_scale", [1.0, 1.0, 1.0]), np.float32)
        self.home_pelvis_height = float(r.get("home_pelvis_height", 0.784))
        self.base_height_enabled = bool(r.get("command_details", {}).get("base_height", {}).get("enabled", False))
        jl = r.get("joint_limits")
        self.joint_limits = None
        if jl is not None:
            jl = np.array(jl, np.float32)
            if jl.shape != (29, 2):
                raise ValueError("sidecar joint_limits must be [29][lo, hi]")
            self.joint_limits = jl
        onnx = r.get("onnx", {})
        self.onnx_file = onnx.get("file", "policy.onnx")
        self.onnx_input = onnx.get("input")
        self.onnx_output = onnx.get("output")
        self.recurrent = onnx.get("recurrent", [])
        for key, n in (("default_qpos", 29), ("kp", 29), ("kd", 29), ("torque_limit", 29)):
            if len(getattr(self, key)) != n:
                raise ValueError(f"sidecar {key} must have {n} entries")


class PolicyRunner:
    def __init__(self, policy_dir, providers=("CPUExecutionProvider",)):
        import onnxruntime as ort
        self.dir = policy_dir
        self.sc = Sidecar(os.path.join(policy_dir, "policy.json"))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(policy_dir, self.sc.onnx_file), so, providers=list(providers))
        ins = self.sess.get_inputs()
        rec_inputs = {r["input"] for r in self.sc.recurrent}
        self.in_name = self.sc.onnx_input or [i.name for i in ins if i.name not in rec_inputs][0]
        self.out_name = self.sc.onnx_output or self.sess.get_outputs()[0].name
        self.frame_dim = sum(self._term_dim(t) for t in self.sc.obs_terms)
        self.obs_dim = self.frame_dim * self.sc.history_len
        shape = [i for i in ins if i.name == self.in_name][0].shape
        if isinstance(shape[-1], int) and shape[-1] != self.obs_dim:
            raise ValueError(f"sidecar obs dim {self.obs_dim} != onnx input dim {shape[-1]}")
        self.reset()

    # ------------------------------------------------------------------ obs
    def _joints(self, t):
        j = t.get("joints", "controlled")
        if j == "controlled":
            return self.sc.controlled
        if j == "all":
            return np.arange(29)
        return np.array(j, dtype=int)

    def _term_dim(self, t):
        n = t["name"]
        if n in ("joint_pos_rel", "joint_pos", "joint_vel"):
            d = len(self._joints(t))
        else:
            d = {"base_ang_vel": 3, "projected_gravity": 3, "cmd_vel": 3, "gait_phase": 2, "cmd_base_height": 1,
                 "last_action": len(self.sc.controlled), "upper_body_target": 17,
                 "upper_body_target_rel": 17}.get(n)
            if d is None:
                raise ValueError(f"unknown obs term '{n}'")
        if int(t.get("dim", d)) != d:
            raise ValueError(f"obs term {n}: sidecar dim {t['dim']} != {d}")
        return d

    def reset(self):
        self.last_action = np.zeros(len(self.sc.controlled), np.float32)
        self.hist = []
        self.t = 0.0
        self.state = {r["input"]: np.zeros(r["shape"], np.float32) for r in self.sc.recurrent}

    def build_obs(self, q, dq, quat, gyro, cmd_vel, upper_target, base_height=None):
        parts = []
        for t in self.sc.obs_terms:
            n = t["name"]
            if n == "base_ang_vel":
                v = np.asarray(gyro, np.float32)
            elif n == "projected_gravity":
                v = projected_gravity(quat)
            elif n == "cmd_vel":
                v = np.asarray(cmd_vel, np.float32) * self.sc.cmd_vel_scale + self.sc.cmd_vel_offset
            elif n == "cmd_base_height":
                off = 0.0
                if self.sc.base_height_enabled and base_height is not None:
                    off = float(base_height) - self.sc.home_pelvis_height
                v = np.array([off], np.float32)
            elif n == "joint_pos_rel":
                j = self._joints(t)
                v = (q[j] - self.sc.default_qpos[j]).astype(np.float32)
            elif n == "joint_pos":
                v = q[self._joints(t)].astype(np.float32)
            elif n == "joint_vel":
                v = dq[self._joints(t)].astype(np.float32)
            elif n == "last_action":
                v = self.last_action
            elif n == "gait_phase":
                ph = (self.t % t["period"]) / t["period"]
                v = np.array([np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph)], np.float32)
            elif n == "upper_body_target":
                v = np.asarray(upper_target, np.float32)
            elif n == "upper_body_target_rel":
                v = np.asarray(upper_target, np.float32) - self.sc.default_qpos[UPPER_BODY]
            v = v * np.asarray(t.get("scale", 1.0), np.float32)
            if "clip" in t:
                v = np.clip(v, -t["clip"], t["clip"])
            parts.append(v.astype(np.float32))
        return np.concatenate(parts)

    def step(self, q, dq, quat, gyro, cmd_vel, upper_target=None, base_height=None):
        """One policy step. Returns 29 joint targets (controlled joints from the policy,
        others = default / upper_target)."""
        if upper_target is None:
            upper_target = self.sc.default_qpos[UPPER_BODY]
        frame = self.build_obs(np.asarray(q), np.asarray(dq), quat, gyro, cmd_vel, upper_target, base_height)
        self.hist.append(frame)
        H = self.sc.history_len
        if len(self.hist) > H:
            self.hist.pop(0)
        while len(self.hist) < H:
            self.hist.insert(0, frame)
        obs = np.concatenate(self.hist)[None, :]
        feeds = {self.in_name: obs}
        feeds.update(self.state)
        outs = self.sess.run(None, feeds)
        names = [o.name for o in self.sess.get_outputs()]
        res = dict(zip(names, outs))
        act = res[self.out_name].reshape(-1).astype(np.float32)
        for r in self.sc.recurrent:
            self.state[r["input"]] = res[r["output"]]
        if self.sc.action_clip:
            act = np.clip(act, -self.sc.action_clip, self.sc.action_clip)
        self.last_action = act
        self.t += self.sc.control_dt
        target = self.sc.default_qpos.copy()
        target[UPPER_BODY] = upper_target
        target[self.sc.controlled] = act * self.sc.action_scale + self.sc.default_qpos[self.sc.controlled]
        if self.sc.joint_limits is not None:
            target = np.clip(target, self.sc.joint_limits[:, 0], self.sc.joint_limits[:, 1])
        return target
