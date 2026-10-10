"""Convert Unitree's pre-trained G1 12-DoF locomotion policy (TorchScript) to ONNX + contract sidecar.

Run with the conda env that has torch (NOT inside env.sh shells):
    ~/miniconda3/envs/unitree-rl/bin/python ros2_ws/src/hospital-sim-ros2/g1_controller/scripts/convert_placeholder.py

Source: ~/ntu_dev/unitree_rl_gym/deploy/pre_train/g1/motion.pt, obs layout/gains/scales from
unitree_rl_gym/deploy/deploy_mujoco/configs/g1.yaml + deploy_mujoco.py (47-dim obs, 12 actions, 50 Hz).
"""
import json
import os

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "policies", "unitree_g1_12dof_placeholder")
GYM = os.path.expanduser("~/ntu_dev/unitree_rl_gym")

JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# Torque limits from the G1 29-DoF rev 1.0 MJCF (actuatorfrcrange).
TAU = [88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50, 88, 50, 50] + [25, 25, 25, 25, 25, 5, 5] * 2

# Upper body (not driven by this policy): PD hold. Gains in the range Unitree uses for G1 arms/waist.
UPPER_KP = [200, 200, 200] + [100, 100, 50, 50, 20, 20, 20] * 2
UPPER_KD = [5, 5, 5] + [2, 2, 2, 2, 1, 1, 1] * 2
# Upper body at zero = the rigid arm pose of the 12-DoF model the policy was trained on. Measured in MuJoCo
# (29-DoF model, arms PD-held): zero arms walk at ~0.6 m/s for cmd 0.5; bent elbows (CoM forward) only
# ~0.2-0.35 m/s with larger drift.
UPPER_DEFAULT = [0.0] * 17


class Stateless(torch.nn.Module):
    """motion.pt keeps its LSTM (h, c) in buffers. ONNX gets them as explicit inputs/outputs instead:
    inputs obs[1,47], h_in[1,1,64], c_in[1,1,64] -> actions[1,12], h_out, c_out."""

    def __init__(self, sd):
        super().__init__()
        self.lstm = torch.nn.LSTM(47, 64, 1)
        self.actor = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ELU(), torch.nn.Linear(32, 12))
        self.lstm.load_state_dict({k[len("memory."):]: v for k, v in sd.items() if k.startswith("memory.")})
        self.actor.load_state_dict({k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")})

    def forward(self, obs, h_in, c_in):
        out, (h, c) = self.lstm(obs.unsqueeze(0), (h_in, c_in))
        return self.actor(out.squeeze(0)), h, c


def main():
    cfg = yaml.safe_load(open(os.path.join(GYM, "deploy/deploy_mujoco/configs/g1.yaml")))
    pol = torch.jit.load(os.path.join(GYM, "deploy/pre_train/g1/motion.pt"), map_location="cpu").eval()
    os.makedirs(OUT, exist_ok=True)
    sd = {k: v.clone() for k, v in pol.state_dict().items()}
    net = Stateless(sd).eval()
    # Verify against the stateful TorchScript over a 200-step rollout (from zero state).
    pol.hidden_state.zero_()
    pol.cell_state.zero_()
    h = torch.zeros(1, 1, 64)
    c = torch.zeros(1, 1, 64)
    err = 0.0
    with torch.no_grad():
        for _ in range(200):
            o = torch.randn(1, cfg["num_obs"]) * 0.5
            a1 = pol(o)
            a2, h, c = net(o, h, c)
            err = max(err, (a1 - a2).abs().max().item())
    print("stateless vs torchscript max abs err:", err)
    assert err < 1e-5
    torch.onnx.export(net, (torch.zeros(1, cfg["num_obs"]), torch.zeros(1, 1, 64), torch.zeros(1, 1, 64)),
                      os.path.join(OUT, "policy.onnx"), input_names=["obs", "h_in", "c_in"],
                      output_names=["actions", "h_out", "c_out"], opset_version=13)
    # Reference rollout for g1_controller's test (onnxruntime must reproduce it).
    import numpy as np
    g = torch.Generator().manual_seed(0)
    obs_seq = torch.randn(20, 1, cfg["num_obs"], generator=g) * 0.5
    h = torch.zeros(1, 1, 64)
    c = torch.zeros(1, 1, 64)
    acts = []
    with torch.no_grad():
        for o in obs_seq:
            a_, h, c = net(o, h, c)
            acts.append(a_.numpy())
    np.savez(os.path.join(OUT, "reference_rollout.npz"), obs=obs_seq.numpy(), actions=np.stack(acts))
    n = cfg["num_actions"]
    side = {
        "name": "unitree_g1_12dof_placeholder",
        "placeholder": True,
        "description": "Unitree unitree_rl_gym pre-trained G1 12-DoF locomotion policy (legs only). "
                       "Waist/arms/hands are PD-held by g1_controller at default_qpos or /g1/upper_body_target. "
                       "PLACEHOLDER until g1-whole-body-rl/exports/ provides the ULTRA-style whole-body policy.",
        "source": "unitree_rl_gym/deploy/pre_train/g1/motion.pt + deploy/deploy_mujoco/configs/g1.yaml",
        "control_dt": cfg["simulation_dt"] * cfg["control_decimation"],
        "sim_dt": cfg["simulation_dt"],
        "joint_names": JOINTS,
        "controlled_joints": list(range(n)),
        "default_qpos": list(map(float, cfg["default_angles"])) + UPPER_DEFAULT,
        "action_scale": cfg["action_scale"],
        "kp": list(map(float, cfg["kps"])) + UPPER_KP,
        "kd": list(map(float, cfg["kds"])) + UPPER_KD,
        "torque_limit": TAU,
        # Observation terms in order. "scale" multiplies the term. Extra keys (joints, period, offset) are
        # interpreted by g1_controller's generic obs builder (see g1_controller/obs.py).
        "obs": [
            {"name": "base_ang_vel", "dim": 3, "scale": cfg["ang_vel_scale"]},
            {"name": "projected_gravity", "dim": 3, "scale": 1.0},
            {"name": "cmd_vel", "dim": 3, "scale": list(map(float, cfg["cmd_scale"]))},
            {"name": "joint_pos_rel", "dim": n, "scale": cfg["dof_pos_scale"], "joints": "controlled"},
            {"name": "joint_vel", "dim": n, "scale": cfg["dof_vel_scale"], "joints": "controlled"},
            {"name": "last_action", "dim": n, "scale": 1.0},
            {"name": "gait_phase", "dim": 2, "scale": 1.0, "period": 0.8},
        ],
        "history_len": 1,
        "commands": {"cmd_vel": [3], "upper_body_target": [17]},
        # Optional extensions (not in the contract): policy cmd = cmd_vel * cmd_vel_scale + cmd_vel_offset,
        # applied before the obs scale. Placeholder-only calibration, measured in MuJoCo on
        # g1-brainco-description/g1_29dof_brainco (BrainCo hands, arms at 0, 35 Nm ankle/waist limits from the robot model):
        #  * at cmd 0 the policy creeps forward ~0.15 m/s; offset -0.13 gives |drift| < 0.05 m/s over 20 s
        #  * forward inputs above ~0.2 switch it into a fast crouched gait that falls (it was trained with
        #    50 Nm ankles); vx scale 0.4 maps the teleop range 0.8 m/s to input 0.19 (~0.4 m/s actual).
        #    Lateral and yaw commands track fine over the full teleop range.
        "cmd_vel_scale": [0.4, 1.0, 1.0],
        "cmd_vel_offset": [-0.13, 0.0, 0.0],
        "cmd_vel_limits": {"vx": [-0.5, 0.8], "vy": [-0.4, 0.4], "wz": [-1.0, 1.0]},
        # ONNX I/O. "recurrent" pairs are fed back every policy step and zeroed when the policy (re)starts.
        "onnx": {"input": "obs", "output": "actions", "obs_dim": cfg["num_obs"], "action_dim": n,
                 "recurrent": [{"input": "h_in", "output": "h_out", "shape": [1, 1, 64]},
                               {"input": "c_in", "output": "c_out", "shape": [1, 1, 64]}]},
    }
    with open(os.path.join(OUT, "policy.json"), "w") as f:
        json.dump(side, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
