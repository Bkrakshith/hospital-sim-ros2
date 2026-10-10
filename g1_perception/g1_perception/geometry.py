"""Frames and projection helpers.

Camera convention: images are in the OpenCV optical convention (x right, y down, z forward). The head camera's
frame in the robot model is `d435_link` (x forward, z up, pitched down 47.6 deg); the fixed rotation is
    R_link_optical = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]      (columns: optical x, y, z expressed in d435_link)
so p_link = R_link_optical @ p_optical. Both sims publish image headers with frame_id `d435_link` (contract 6).
"""
from __future__ import annotations

import math

import numpy as np

R_LINK_OPT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_wxyz(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q if q[0] >= 0 else -q


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def pose7_to_T(p):
    """[x, y, z, qw, qx, qy, qz] (the /sim/objects layout) -> 4x4."""
    return make_T(quat_wxyz_to_R(p[3:7]), p[:3])


def yaw_of(R):
    """Heading of the body x axis projected on the horizontal plane."""
    return math.atan2(R[1, 0], R[0, 0])


def backproject(u, v, z, K):
    K = np.asarray(K, float).reshape(3, 3)
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def project(P, K):
    """Nx3 optical-frame points -> Nx2 pixels (nan where z <= 0)."""
    K = np.asarray(K, float).reshape(3, 3)
    P = np.atleast_2d(P)
    z = P[:, 2:3]
    uv = (P @ K.T)[:, :2] / np.where(z > 1e-6, z, np.nan)
    return uv


def box_corners(bmin, bmax):
    return np.array([[x, y, z] for x in (bmin[0], bmax[0]) for y in (bmin[1], bmax[1]) for z in (bmin[2], bmax[2])], float)


def min_area_rect(pts):
    """Minimum-area rectangle of 2-D points -> [cx, cy, w, h, angle_deg] with w >= h, angle of the long side in [0, 180)."""
    import cv2

    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(pts, np.float32))
    if w < h:
        w, h, a = h, w, a + 90.0
    return [float(cx), float(cy), float(w), float(h), float(a % 180.0)]
