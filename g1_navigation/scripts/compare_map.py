#!/usr/bin/env python3
"""Compare a SLAM map (map_server yaml/pgm, world frame) with ground-truth occupancy.

Ground truths:
  --gt-yaml  maps/hospital_gt.yaml          (rasterize_world.py from the MuJoCo world: AABB approximation)
  --gt-npz   collider_raster.npz            (g1-isaac-sim/tools/rasterize_colliders.py: the Isaac static colliders the
                                             simulated lidar actually ray-casts; 'scan' band 1.0-1.4 m)
Metrics:
  * alignment: distance from every SLAM-occupied cell to the nearest GT-occupied cell (median / mean / p95 / share
    within 0.10 m), plus the best rigid offset found by a small grid search (dx, dy +-0.5 m, dyaw +-2 deg);
  * coverage: share of the GT reachable floor (connected free space from the start, robot radius 0.35 m) that the
    SLAM map marks as known free; share of GT wall cells next to that floor that SLAM marks occupied (+-0.10 m).

    compare_map.py --slam maps/hospital_full.yaml --gt-npz ... --gt-yaml maps/hospital_gt.yaml --json out.json
"""
import argparse
import json
import math
import os
import warnings

import numpy as np
import yaml
from PIL import Image

warnings.filterwarnings("ignore")
from scipy import ndimage  # noqa: E402


def load_yaml_map(path):
    y = yaml.safe_load(open(path))
    img = np.array(Image.open(os.path.join(os.path.dirname(path), y["image"])))[::-1]  # row 0 = bottom
    occ = img < 100          # trinary: 0 occupied, 254 free, 205 unknown
    free = img > 250
    return occ, free, float(y["resolution"]), float(y["origin"][0]), float(y["origin"][1])


def resample(occ, res, ox, oy, X0, Y0, H, W, R):
    """nearest-neighbour resample a grid into the reference grid (origin X0, Y0, res R, H x W)."""
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    xs, ys = X0 + (jj + 0.5) * R, Y0 + (ii + 0.5) * R
    c, r = ((xs - ox) / res).astype(int), ((ys - oy) / res).astype(int)
    ok = (r >= 0) & (r < occ.shape[0]) & (c >= 0) & (c < occ.shape[1])
    out = np.zeros((H, W), bool)
    out[ok] = occ[r[ok], c[ok]]
    return out


def metrics(s_occ, s_free, g_occ, g_floor, R, X0, Y0, start):
    dt = ndimage.distance_transform_edt(~g_occ) * R
    d = dt[s_occ]
    al = {"slam_occupied_cells": int(s_occ.sum()), "median_m": round(float(np.median(d)), 3),
          "mean_m": round(float(d.mean()), 3), "p95_m": round(float(np.percentile(d, 95)), 3),
          "within_0.10m": round(float((d <= 0.10 + 1e-6).mean()), 3),
          "within_0.25m": round(float((d <= 0.25 + 1e-6).mean()), 3)}
    # reachable GT floor from the start (robot radius 0.35 m), then dilate back by the radius = floor a robot sees
    clear = ndimage.distance_transform_edt(~g_occ) * R > 0.35
    clear &= g_floor
    lab, _ = ndimage.label(clear)
    si, sj = int((start[1] - Y0) / R), int((start[0] - X0) / R)
    reach = lab == lab[si, sj]
    reach = ndimage.binary_dilation(reach, iterations=int(0.35 / R)) & ~g_occ & g_floor
    walls = ndimage.binary_dilation(reach, iterations=2) & g_occ
    s_occ_d = ndimage.binary_dilation(s_occ, iterations=2)
    cov = {"gt_reachable_floor_m2": round(float(reach.sum() * R * R), 1),
           "slam_known_free_of_reachable": round(float((s_free & reach).sum() / reach.sum()), 3),
           "slam_free_m2_total": round(float(s_free.sum() * R * R), 1),
           "gt_wall_cells_seen": round(float((s_occ_d & walls).sum() / max(1, walls.sum())), 3)}
    return al, cov


def best_offset(s_occ, g_occ, R):
    dt = ndimage.distance_transform_edt(~g_occ) * R
    ii, jj = np.nonzero(s_occ)
    cy, cx = ii.mean(), jj.mean()
    best = None
    for dyaw in np.radians(np.arange(-2.0, 2.01, 0.5)):
        c, s = math.cos(dyaw), math.sin(dyaw)
        ri = cy + s * (jj - cx) + c * (ii - cy)
        rj = cx + c * (jj - cx) - s * (ii - cy)
        for dx in np.arange(-0.5, 0.51, 0.05):
            for dy in np.arange(-0.5, 0.51, 0.05):
                a = np.clip((ri + dy / R).round().astype(int), 0, dt.shape[0] - 1)
                b = np.clip((rj + dx / R).round().astype(int), 0, dt.shape[1] - 1)
                m = float(np.median(dt[a, b]))
                if best is None or m < best[0]:
                    best = (m, dx, dy, math.degrees(dyaw))
    return {"median_after_m": round(best[0], 3), "dx_m": round(best[1], 2), "dy_m": round(best[2], 2),
            "dyaw_deg": round(best[3], 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slam", required=True)
    ap.add_argument("--gt-npz", default="")
    ap.add_argument("--gt-yaml", default="")
    ap.add_argument("--start", default="0,0")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    start = tuple(float(v) for v in a.start.split(","))
    s_occ0, s_free0, sres, sox, soy = load_yaml_map(a.slam)
    out = {"slam_map": a.slam}
    if a.gt_npz:
        d = np.load(a.gt_npz)
        X0, Y0 = (float(v) for v in d["origin"])
        R = float(d["res"])
        H, W = d["scan"].shape
        s_occ = resample(s_occ0, sres, sox, soy, X0, Y0, H, W, R)
        s_free = resample(s_free0, sres, sox, soy, X0, Y0, H, W, R)
        g_occ = d["scan"] | d["band"]
        al, cov = metrics(s_occ, s_free, d["scan"], d["floor"], R, X0, Y0, start)
        al_band, _ = metrics(s_occ, s_free, g_occ, d["floor"], R, X0, Y0, start)
        al["median_m_vs_any_height_0.2_1.8"] = al_band["median_m"]
        al["within_0.10m_vs_any_height"] = al_band["within_0.10m"]
        out["vs_isaac_colliders"] = {"alignment": al, "coverage": cov, "best_rigid_offset": best_offset(s_occ, d["scan"], R)}
    if a.gt_yaml:
        g_occ, g_free, R, X0, Y0 = load_yaml_map(a.gt_yaml)
        H, W = g_occ.shape
        s_occ = resample(s_occ0, sres, sox, soy, X0, Y0, H, W, R)
        s_free = resample(s_free0, sres, sox, soy, X0, Y0, H, W, R)
        al, cov = metrics(s_occ, s_free, g_occ, g_free | g_occ, R, X0, Y0, start)
        out["vs_hospital_gt_mujoco"] = {"alignment": al, "coverage": cov, "best_rigid_offset": best_offset(s_occ, g_occ, R)}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
