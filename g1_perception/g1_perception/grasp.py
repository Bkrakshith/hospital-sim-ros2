"""Grasp pose (x, y, z, yaw) from an oriented box + aligned depth. In short:

1. Sample the depth image inside the OBB shrunk to 80 % across (drops the long edges, mostly background) and
   98 % along (keeps the instrument's ends: the top of an upright instrument sets its origin height).
2. The instrument is the nearest surface in its box, slice by slice: split the box into 8 slices along its long
   side; in each, z_b = 15th percentile of the valid depths and the pixels with |z - z_b| <= 1.5 cm are kept ->
   the instrument's pixels S (per slice, because an upright instrument seen from the head spans ~10 cm of depth).
3. Back-project S with K: P_i = z_i K^-1 [u_i, v_i, 1]^T (optical frame), then to the output frame (z up).
4. Grasp point g = component-wise median of P_i. Axis a = median(P in the last non-empty slice) - median(P in
   the first); if the box is not elongated (w/h < 2) or S is too small, a = the OBB long side back-projected at z_ref.
5. upright = |a_z| > cos 30 deg. Upright (in the rack): the hand approaches horizontally from the camera, so
   yaw = heading of (g - camera) in the horizontal plane. Lying: yaw = heading of a (mod pi, the end nearer the
   camera's heading is chosen), i.e. the hand closes across the long axis.
6. Object origin (the /sim/objects frame: tip end on the long axis) for the contract `pose` field: upright ->
   origin = (g_x, g_y, top_z - L) with top_z the highest point of S and L the catalogue length; lying ->
   origin = centroid - L/2 * a (tip end ambiguous: flagged). yaw of the pose = the grasp yaw (in the rack the
   instruments' +x axis points away from the robot, i.e. along the camera heading).
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import backproject

COS30 = math.cos(math.radians(30))


def obb_mask(shape, obb, shrink=0.8, shrink_along=0.98):
    """Pixels (ys, xs) inside the OBB (shrunk) and their coordinate along the long side (pixels from the centre)."""
    cx, cy, w, h, ang = obb
    H, W = shape
    a = math.radians(ang)
    ux, uy = math.cos(a), math.sin(a)
    x0, x1 = int(max(0, cx - (w + h) / 2 - 1)), int(min(W, cx + (w + h) / 2 + 2))
    y0, y1 = int(max(0, cy - (w + h) / 2 - 1)), int(min(H, cy + (w + h) / 2 + 2))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - cx, yy - cy
    along = dx * ux + dy * uy
    across = -dx * uy + dy * ux
    m = (np.abs(along) <= shrink_along * w / 2) & (np.abs(across) <= max(1.0, shrink * h / 2))
    return yy[m], xx[m], along[m]


def grasp_from_obb(obb, depth, K, T_out_cam=None, length=None, min_points=12, bins=8, band=0.015):
    """obb [cx, cy, w, h, angle_deg] (pixels, long side angle); depth HxW metres (nan / 0 = invalid);
    K 3x3; T_out_cam 4x4 optical camera frame -> output frame (identity: result in the optical frame).
    Returns dict(grasp_xyz, grasp_yaw, axis, upright, origin_xyz, n_points, depth_m, ok) or None if no depth."""
    T = np.eye(4) if T_out_cam is None else np.asarray(T_out_cam, float)
    K = np.asarray(K, float).reshape(3, 3)
    ys, xs, al = obb_mask(depth.shape, obb)
    z = depth[ys, xs].astype(float)
    ok = np.isfinite(z) & (z > 0.05)
    if ok.sum() < 3:
        return None
    ys, xs, al, z = ys[ok], xs[ok], al[ok], z[ok]
    cx, cy, w, h, ang = obb
    # nearest surface per slice along the long side (an upright instrument seen obliquely spans ~10 cm of depth)
    edges = np.linspace(-w / 2, w / 2, bins + 1)
    bi = np.clip(np.digitize(al, edges) - 1, 0, bins - 1)
    keep = np.zeros(len(z), bool)
    for b in range(bins):
        m = bi == b
        if m.sum() == 0:
            continue
        zb = np.percentile(z[m], 15)
        keep |= m & (z >= zb - band) & (z <= zb + band)
    ys, xs, al, z, bi = ys[keep], xs[keep], al[keep], z[keep], bi[keep]
    zref = float(np.percentile(z, 15))
    Pc = np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], 1)
    P = Pc @ T[:3, :3].T + T[:3, 3]
    g = np.median(P, 0)
    cam = T[:3, 3]
    ends = [b for b in range(bins) if (bi == b).sum() >= 2]
    if len(P) >= min_points and w / max(h, 1e-6) >= 2.0 and len(ends) >= 2:
        a = np.median(P[bi == ends[-1]], 0) - np.median(P[bi == ends[0]], 0)
    else:
        e = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))]) * w / 2
        p1 = backproject(cx - e[0], cy - e[1], zref, K)
        p2 = backproject(cx + e[0], cy + e[1], zref, K)
        a = T[:3, :3] @ (p2 - p1)
    a = a / (np.linalg.norm(a) + 1e-12)
    if a[2] < 0:
        a = -a  # axis points up (towards the handle for an upright instrument)
    upright = abs(a[2]) > COS30
    view = g - cam
    view_yaw = math.atan2(view[1], view[0])
    if upright:
        yaw = view_yaw
        top = float(P[:, 2].max())
        origin = np.array([g[0], g[1], top - length]) if length else None
    else:
        yaw = math.atan2(a[1], a[0])
        # choose the direction of the long axis closest to the viewing heading (180 deg ambiguity)
        if math.cos(yaw - view_yaw) < 0:
            yaw += math.pi
        yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
        origin = (P.mean(0) - 0.5 * length * a) if length else None
    return dict(grasp_xyz=g.tolist(), grasp_yaw=yaw, axis=a.tolist(), upright=bool(upright),
                origin_xyz=origin.tolist() if origin is not None else None, n_points=int(len(P)), depth_m=zref, ok=True)
