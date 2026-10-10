"""Index maps between the MuJoCo model and the contract joint order (docs/interfaces.md section 2).
Pure MuJoCo/numpy (no ROS) so tests and offline tools can use it."""
import numpy as np
import mujoco

BODY_JOINTS = [
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
HAND_SLOTS = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]


class ModelMap:
    def __init__(self, m):
        self.m = m
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in BODY_JOINTS]
        missing = [n for n, j in zip(BODY_JOINTS, jid) if j < 0]
        if missing:
            raise ValueError(f"model lacks joints {missing}")
        self.jid = np.array(jid)
        self.qadr = m.jnt_qposadr[self.jid].copy()
        self.dadr = m.jnt_dofadr[self.jid].copy()
        # body motors: actuator whose transmission targets each body joint
        act = []
        for j in jid:
            a = [i for i in range(m.nu) if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT
                 and m.actuator_trnid[i, 0] == j]
            if not a:
                raise ValueError(f"no actuator for joint {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)}")
            act.append(a[0])
        self.act = np.array(act)
        rng = m.actuator_ctrlrange[self.act]
        lim = np.where(m.actuator_ctrllimited[self.act].astype(bool), np.abs(rng).max(1), np.inf)
        jr = m.jnt_actfrcrange[self.jid]
        jl = np.where(m.jnt_actfrclimited[self.jid].astype(bool), np.abs(jr).max(1), np.inf)
        self.tau_limit = np.minimum(lim, jl)
        # free joint (pelvis)
        free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        self.free_q = m.jnt_qposadr[free[0]]
        self.free_d = m.jnt_dofadr[free[0]]
        self.pelvis = m.jnt_bodyid[free[0]]
        # hands: position actuators named {side}_hand_{slot}
        self.hand_act, self.hand_joint_names, self.hand_upper, self.hand_qadr = {}, {}, {}, {}
        for side in ("left", "right"):
            ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_hand_{s}") for s in HAND_SLOTS]
            if all(i >= 0 for i in ids):
                self.hand_act[side] = np.array(ids)
                js = [m.actuator_trnid[i, 0] for i in ids]
                self.hand_joint_names[side] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in js]
                self.hand_upper[side] = m.actuator_ctrlrange[ids, 1].copy()
                self.hand_qadr[side] = m.jnt_qposadr[js].copy()
        self.has_hands = len(self.hand_act) == 2
        # sites / bodies
        self.imu_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis")
        self.lidar_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "mid360_link")
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        # accelerometer on the pelvis IMU site (unitree_mujoco naming: imu_acc)
        acc = [i for i in range(m.nsensor) if m.sensor_type[i] == mujoco.mjtSensor.mjSENS_ACCELEROMETER
               and self.imu_site >= 0 and m.sensor_objid[i] == self.imu_site]
        self.acc_adr = m.sensor_adr[acc[0]] if acc else None

    def q(self, d):
        return d.qpos[self.qadr]

    def dq(self, d):
        return d.qvel[self.dadr]

    def set_body_q(self, d, q):
        d.qpos[self.qadr] = q
