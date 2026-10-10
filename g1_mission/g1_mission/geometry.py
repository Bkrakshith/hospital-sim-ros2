"""Small rigid-transform helpers (numpy only; quaternions are w, x, y, z like MuJoCo and the contract)."""
import math

import numpy as np


def quat_to_mat(q):
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q if q[0] >= 0 else -q


def yaw_of(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def pose_to_T(pose7):
    """[x, y, z, qw, qx, qy, qz] -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(pose7[3:7])
    T[:3, 3] = pose7[:3]
    return T


def T_to_pose(T):
    return list(T[:3, 3]) + list(mat_to_quat(T[:3, :3]))


def make_T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_axis(axis, a):
    axis = np.asarray(axis, float) / (np.linalg.norm(axis) or 1.0)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def rot_log(R):
    """Rotation vector of R."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if math.pi - a < 1e-6:          # ~180 deg: axis from the symmetric part
        M = (R + np.eye(3)) / 2
        k = int(np.argmax(np.diag(M)))
        ax = M[:, k] / math.sqrt(max(M[k, k], 1e-12))
        return ax * a
    return v * a / (2 * math.sin(a))


def interp_T(A, B, s):
    """Pose interpolation: linear position, geodesic rotation; s in [0, 1]."""
    s = min(1.0, max(0.0, s))
    w = rot_log(A[:3, :3].T @ B[:3, :3])
    ang = np.linalg.norm(w)
    Rs = A[:3, :3] @ (rot_axis(w, ang * s) if ang > 1e-9 else np.eye(3))
    return make_T(Rs, A[:3, 3] + s * (B[:3, 3] - A[:3, 3]))


def smoothstep(s):
    s = min(1.0, max(0.0, s))
    return s * s * (3 - 2 * s)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
