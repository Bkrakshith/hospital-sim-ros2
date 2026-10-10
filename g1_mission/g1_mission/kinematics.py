"""Upper-body kinematics for the arm skills: pinocchio on g1-brainco-description/g1_29dof_brainco/urdf/g1_29dof_brainco_ik.urdf
(29 body joints, hands rigid, root = pelvis, so every pose here is in the **pelvis frame**).

Tool frames (TCP) on the wrist_yaw links, measured on the Revo2 model (fingers +x, finger stacking +z (index up
when the palm faces the body's left/right), palm normal +y for the right hand / -y for the left):
  <side>_grasp = wrist_yaw_link * translation(0.125, +-0.040, 0): centre of the power-grasp enclosure (object axis
                 along the TCP z axis).
Upper-body vector order = contract / manager UPPER_NAMES: waist yaw, roll, pitch, left arm 7, right arm 7.
"""
import os

import numpy as np

UPPER_NAMES = [
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
WAIST = [0, 1, 2]
ARM = {"left": list(range(3, 10)), "right": list(range(10, 17))}
TCP_OFFSET = {"right": (0.125, 0.040, 0.0), "left": (0.125, -0.040, 0.0)}
LEG_DEFAULT = [-0.1, 0, 0, 0.3, -0.2, 0] * 2


def repo_root():
    r = os.environ.get("HG1_ROOT")
    if r and os.path.isdir(r):
        return r
    here = os.path.realpath(__file__)
    for _ in range(8):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "g1-brainco-description", "g1_29dof_brainco")):
            return here
    return os.getcwd()


def default_urdf():
    return os.path.join(repo_root(), "g1-brainco-description", "g1_29dof_brainco", "urdf", "g1_29dof_brainco_ik.urdf")


class UpperBodyIK:
    """Damped-least-squares IK for one arm (+ optionally the waist) towards a TCP pose in the pelvis frame.

    Joint limits are respected by clamping; a posture term pulls unused freedom towards `rest`. The waist has a
    stronger posture weight than the arm (waist motion moves the centre of mass under a walking policy)."""

    def __init__(self, urdf=None, side="right", use_waist=True, waist_weight=3.0):
        import pinocchio as pin
        self.pin = pin
        self.side = side
        self.model = pin.buildModelFromUrdf(urdf or default_urdf())
        self.data = self.model.createData()
        m = self.model
        parent = m.getFrameId(f"{side}_wrist_yaw_link")
        fr = m.frames[parent]
        off = pin.SE3(np.eye(3), np.array(TCP_OFFSET[side]))
        self.tcp = m.addFrame(pin.Frame(f"{side}_grasp", fr.parentJoint, parent, fr.placement * off,
                                        pin.FrameType.OP_FRAME))
        self.data = m.createData()
        self.qidx = [m.joints[m.getJointId(n)].idx_q for n in UPPER_NAMES]
        self.vidx = [m.joints[m.getJointId(n)].idx_v for n in UPPER_NAMES]
        legs = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
                "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
                "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"]
        self.q_full = pin.neutral(m)
        for n, v in zip(legs, LEG_DEFAULT):
            self.q_full[m.joints[m.getJointId(n)].idx_q] = v
        self.lo = np.array([m.lowerPositionLimit[i] for i in self.qidx])
        self.hi = np.array([m.upperPositionLimit[i] for i in self.qidx])
        self.active = (WAIST if use_waist else []) + ARM[side]
        self.reg = np.array([waist_weight if i in WAIST else 1.0 for i in self.active])

    # ------------------------------------------------------------------ forward kinematics
    def _full(self, q_upper):
        q = self.q_full.copy()
        for k, i in enumerate(self.qidx):
            q[i] = q_upper[k]
        return q

    def fk(self, q_upper, frame=None):
        """4x4 pose of the TCP (or a named frame) in the pelvis frame."""
        pin = self.pin
        q = self._full(q_upper)
        pin.framesForwardKinematics(self.model, self.data, q)
        fid = self.tcp if frame is None else self.model.getFrameId(frame)
        M = self.data.oMf[fid]
        T = np.eye(4)
        T[:3, :3] = M.rotation
        T[:3, 3] = M.translation
        return T

    # ------------------------------------------------------------------ inverse kinematics
    def solve(self, T_target, q_init, rest=None, iters=150, w_rot=0.35, tol_pos=1e-3, tol_rot=1e-2, damping=1e-3,
              rot_mask=None):
        """Returns (q_upper (17), pos_err m, rot_err rad). q_init/rest: 17-vectors. rot_mask: optional 3-vector of
        weights on the rotation error expressed in the TARGET frame (e.g. (0, 0, 1)-ish masks)."""
        pin = self.pin
        q_up = np.array(q_init, float).copy()
        rest = np.array(rest if rest is not None else q_init, float)
        act = self.active
        Wr = np.ones(3) if rot_mask is None else np.asarray(rot_mask, float)
        for _ in range(iters):
            q = self._full(q_up)
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            M = self.data.oMf[self.tcp]
            ep = T_target[:3, 3] - M.translation
            Rerr = T_target[:3, :3] @ M.rotation.T                  # world-frame rotation error
            er = pin.log3(Rerr)
            er_t = T_target[:3, :3].T @ er                          # in target frame -> mask -> back
            er = T_target[:3, :3] @ (Wr * er_t)
            pe, re = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
            if pe < tol_pos and re < tol_rot:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.tcp, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jv = J[:, [self.vidx[i] for i in act]]
            Jr = Jv[3:]
            Jr = T_target[:3, :3] @ np.diag(Wr) @ T_target[:3, :3].T @ Jr
            A = np.vstack([Jv[:3], w_rot * Jr])
            e = np.concatenate([ep, w_rot * er])
            # weighted damped pseudo-inverse (waist columns cost more) + posture in the task nullspace
            Winv = np.diag(1.0 / self.reg)
            lam = damping + 0.05 * max(0.0, pe - 0.02)
            Jp = Winv @ A.T @ np.linalg.inv(A @ Winv @ A.T + lam * np.eye(6))
            N = np.eye(len(act)) - Jp @ A
            dq = Jp @ e + N @ (0.1 * (rest[act] - q_up[act]))
            step = np.clip(dq, -0.2, 0.2)
            q_up[act] = np.clip(q_up[act] + step, self.lo[act], self.hi[act])
        q = self._full(q_up)
        pin.framesForwardKinematics(self.model, self.data, q)
        M = self.data.oMf[self.tcp]
        pe = float(np.linalg.norm(T_target[:3, 3] - M.translation))
        er = pin.log3(T_target[:3, :3] @ M.rotation.T)
        er = T_target[:3, :3] @ (Wr * (T_target[:3, :3].T @ er))
        return q_up, pe, float(np.linalg.norm(er))
