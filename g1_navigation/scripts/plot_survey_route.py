#!/usr/bin/env python3
"""Check and plot the survey route (config/survey_route.yaml) on the Isaac collider raster.

For every leg it runs A* on a 0.1 m grid (robot radius 0.35 m, stairs keep-out from the route file) and reports
leg length, clearance and distance to the stairs; then draws a top-down plot. Optionally overlays the walked trail
(survey_runner's bag/trail CSV: x,y per line).

    python3 plot_survey_route.py --route config/survey_route.yaml \
        --raster g1-isaac-sim/tools/results/collider_raster.npz --out maps/survey_route.png [--trail walked.csv]
The raster comes from g1-isaac-sim/tools/rasterize_colliders.py (static colliders = what the sim lidar sees).
"""
import argparse
import heapq
import math
import warnings

import numpy as np
import yaml

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True)
    ap.add_argument("--raster", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trail", default="")
    ap.add_argument("--radius", type=float, default=0.35)
    a = ap.parse_args()
    from scipy import ndimage
    route = yaml.safe_load(open(a.route))
    d = np.load(a.raster)
    X0, Y0 = d["origin"]
    R = float(d["res"])
    occ = d["band"] | ~d["floor"]
    f = 2
    H, W = occ.shape[0] // f, occ.shape[1] // f
    occ_c = occ[:H * f, :W * f].reshape(H, f, W, f).any(axis=(1, 3))
    RC = R * f
    dist = ndimage.distance_transform_edt(~occ_c) * RC
    keep = float(route.get("stairs_keepout_m", 1.5))
    stairs = [tuple(s["box"]) for s in route.get("stairs", [])]
    ii, jj = np.mgrid[0:H, 0:W]
    xs, ys = X0 + (jj + 0.5) * RC, Y0 + (ii + 0.5) * RC
    ko = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in stairs:
        ko |= np.hypot(np.maximum(np.maximum(x0 - xs, 0), xs - x1), np.maximum(np.maximum(y0 - ys, 0), ys - y1)) < keep
    free = (dist > a.radius) & ~ko
    pen = 1.0 + 2.0 * np.clip(1.0 - dist / 1.2, 0, 1)

    def cell(x, y):
        return int((y - Y0) / RC), int((x - X0) / RC)

    def xy(c):
        return X0 + (c[1] + 0.5) * RC, Y0 + (c[0] + 0.5) * RC

    def sd(x, y):
        return min(math.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1)) for x0, y0, x1, y1 in stairs)

    def astar(p, q):
        s, g = cell(*p), cell(*q)
        if not free[s] or not free[g]:
            return None
        op, gs, par = [(0, s)], {s: 0.0}, {}
        nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414),
              (1, 1, 1.414)]
        while op:
            _, c = heapq.heappop(op)
            if c == g:
                break
            for di, dj, w in nb:
                n = (c[0] + di, c[1] + dj)
                if 0 <= n[0] < H and 0 <= n[1] < W and free[n]:
                    ng = gs[c] + w * pen[n]
                    if ng < gs.get(n, 1e18):
                        gs[n], par[n] = ng, c
                        heapq.heappush(op, (ng + math.hypot(n[0] - g[0], n[1] - g[1]), n))
        if g not in par and g != s:
            return None
        path = [g]
        while path[-1] != s:
            path.append(par[path[-1]])
        return [xy(c) for c in path[::-1]]

    st = route["start"]
    pts = [(st["x"], st["y"])] + [(w["x"], w["y"]) for w in route["waypoints"]]
    paths, total, bad = [], 0.0, 0
    for k in range(1, len(pts)):
        p = astar(pts[k - 1], pts[k])
        paths.append(p)
        w = route["waypoints"][k - 1]
        if p is None:
            bad += 1
            print(f"FAIL {w['name']}")
            continue
        L = sum(math.hypot(p[i + 1][0] - p[i][0], p[i + 1][1] - p[i][1]) for i in range(len(p) - 1))
        total += L
        print(f"ok   {w['name']:28s} {L:6.1f} m  min clearance {min(dist[cell(*q)] for q in p):.2f} m  "
              f"min stairs {min(sd(*q) for q in p):.1f} m")
    print(f"planned length {total:.0f} m, {len(pts) - 1} waypoints, {bad} unreachable")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    img = np.ones(occ.shape + (3,)) * 0.86
    img[d["floor"]] = 1.0
    img[d["band"]] = [0.25, 0.27, 0.32]
    fig, ax = plt.subplots(figsize=(16, 12.5))
    ax.imshow(img, origin="lower", extent=[X0, X0 + occ.shape[1] * R, Y0, Y0 + occ.shape[0] * R],
              interpolation="nearest")
    for x0, y0, x1, y1 in stairs:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color="#d62728", alpha=0.6))
        ax.add_patch(plt.Rectangle((x0 - keep, y0 - keep), x1 - x0 + 2 * keep, y1 - y0 + 2 * keep, fill=False,
                                   ec="#d62728", ls="--", lw=1.2))
    for p in paths:
        if p:
            ax.plot([q[0] for q in p], [q[1] for q in p], "-", color="#1f77b4", lw=1.6, alpha=0.9)
    if a.trail:
        t = np.loadtxt(a.trail, delimiter=",", ndmin=2)
        ax.plot(t[:, 0], t[:, 1], "-", color="#ff7f0e", lw=1.0, alpha=0.9, label="walked (SLAM pose)")
    for k, (x, y) in enumerate(pts[1:], 1):
        ax.plot(x, y, "o", color="#ffbf00", mec="k", ms=5)
        ax.text(x + 0.25, y + 0.25, str(k), fontsize=7)
    ax.plot(st["x"], st["y"], "*", color="#2ca02c", mec="k", ms=16, label="start / end (world origin)")
    ax.plot([], [], "-", color="#1f77b4", label=f"planned legs (A*, {total:.0f} m)")
    ax.plot([], [], "o", color="#ffbf00", mec="k", label="waypoints")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="#d62728", alpha=0.6, label=f"stairs (+{keep} m keep-out)"))
    ax.set_xlim(-51, 34)
    ax.set_ylim(-26, 38.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, world / map frame)")
    ax.set_ylabel("y (m)")
    ax.set_title("G1 hospital survey route: Isaac hospital ground floor + surgical wing (simulation)")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    plt.savefig(a.out, dpi=110, bbox_inches="tight")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
