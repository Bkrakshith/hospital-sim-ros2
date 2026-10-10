#!/usr/bin/env python3
"""Rasterise the hospital world into a GROUND-TRUTH occupancy map (map_server pgm + yaml).

This is NOT a SLAM map: it is computed from the simulator's geometry and is meant as a seed /
reference (e.g. to test Nav2 before a SLAM map exists, or to compare SLAM results against).
The primary way to make maps is SLAM (g1_navigation slam.launch.py + the dashboard's Mapping page).

Two backends:
  --mjcf PATH    exact: every non-robot geom whose vertical extent overlaps the robot's body band
                 (default 0.08..1.80 m) is projected onto the floor (boxes/meshes as their true
                 convex footprint, other primitives as their bounding box).
  --layout       approximate: walls + doors from hospital-sim-env/isaac/scene/layout.py (no furniture except a few
                 labelled boxes). Used by the dashboard's mock robot when no sim world exists.

The map frame equals the simulator's world frame (origin of the MJCF / wing-local frame of
layout.py). For AMCL, set the initial pose from the dashboard.

    rasterize_world.py --mjcf g1-mujoco-sim/hospital_g1.xml --out maps/hospital_gt
    rasterize_world.py --layout --out maps/hospital_layout_gt
"""
import argparse
import os
import sys

import numpy as np

RES = 0.05


# ---------------------------------------------------------------------------------------------
# Polygon rasterisation
# ---------------------------------------------------------------------------------------------
def _convex_hull(pts):
    pts = sorted(set(map(tuple, np.round(pts, 5))))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


class Grid:
    """Occupancy grid in world metres; cell (0,0) is at (x0, y0) (bottom-left)."""

    def __init__(self, x0, y0, x1, y1, res=RES):
        self.res = res
        self.x0, self.y0 = x0, y0
        self.w = int(np.ceil((x1 - x0) / res))
        self.h = int(np.ceil((y1 - y0) / res))
        self.occ = np.zeros((self.h, self.w), dtype=bool)
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        self.cx = x0 + (xx + 0.5) * res
        self.cy = y0 + (yy + 0.5) * res

    def fill_polygon(self, poly):
        poly = np.asarray(poly, dtype=float)
        if len(poly) < 3:
            return
        xmin, ymin = poly.min(axis=0)
        xmax, ymax = poly.max(axis=0)
        i0 = max(int((xmin - self.x0) / self.res) - 1, 0)
        i1 = min(int((xmax - self.x0) / self.res) + 2, self.w)
        j0 = max(int((ymin - self.y0) / self.res) - 1, 0)
        j1 = min(int((ymax - self.y0) / self.res) + 2, self.h)
        if i0 >= i1 or j0 >= j1:
            return
        cx = self.cx[j0:j1, i0:i1]
        cy = self.cy[j0:j1, i0:i1]
        inside = np.ones(cx.shape, dtype=bool)
        n = len(poly)
        # convex polygon (counter-clockwise): inside = left of every edge, with half-cell slack so
        # thin walls (10 cm) are never lost between cell centres
        slack = self.res * 0.5
        for k in range(n):
            ax, ay = poly[k]
            bx, by = poly[(k + 1) % n]
            ex, ey = bx - ax, by - ay
            ln = np.hypot(ex, ey) or 1.0
            inside &= ((ex * (cy - ay) - ey * (cx - ax)) / ln) >= -slack
        self.occ[j0:j1, i0:i1] |= inside

    def fill_rect(self, x0, y0, x1, y1):
        xa, xb = sorted((x0, x1))
        ya, yb = sorted((y0, y1))
        self.fill_polygon([(xa, ya), (xb, ya), (xb, yb), (xa, yb)])

    def save(self, out_base, free_value=254, comment=''):
        """Write out_base.pgm + out_base.yaml (map_server format; row 0 of the pgm is the top)."""
        img = np.where(self.occ, 0, free_value).astype(np.uint8)[::-1]
        os.makedirs(os.path.dirname(os.path.abspath(out_base)) or '.', exist_ok=True)
        with open(out_base + '.pgm', 'wb') as f:
            f.write(b'P5\n# %s\n%d %d\n255\n' % (comment.encode()[:60], self.w, self.h))
            f.write(img.tobytes())
        with open(out_base + '.yaml', 'w') as f:
            f.write(f"# GROUND-TRUTH map rasterised from simulator geometry ({comment}); not a SLAM map\n")
            f.write(f"image: {os.path.basename(out_base)}.pgm\n")
            f.write("mode: trinary\n")
            f.write(f"resolution: {self.res}\n")
            f.write(f"origin: [{self.x0:.3f}, {self.y0:.3f}, 0.0]\n")
            f.write("negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
        return out_base + '.yaml'


# ---------------------------------------------------------------------------------------------
# Backend 1: MuJoCo MJCF
# ---------------------------------------------------------------------------------------------
def rasterize_mjcf(path, zmin=0.08, zmax=1.80, margin=1.0, robot_root='pelvis', res=RES):
    import mujoco
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # bodies belonging to the robot (subtree of robot_root) are excluded
    robot_bodies = set()
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root)
    if rid >= 0:
        for b in range(model.nbody):
            p = b
            while p > 0:
                if p == rid:
                    robot_bodies.add(b)
                    break
                p = model.body_parentid[p]

    footprints = []
    G = mujoco.mjtGeom
    for g in range(model.ngeom):
        if model.geom_bodyid[g] in robot_bodies:
            continue
        typ = model.geom_type[g]
        if typ in (G.mjGEOM_PLANE, G.mjGEOM_HFIELD):
            continue
        # skip purely visual geoms that do not collide (contype=conaffinity=0), they are not walls
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        pos = data.geom_xpos[g]
        R = data.geom_xmat[g].reshape(3, 3)
        size = model.geom_size[g]
        if typ == G.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            local = model.mesh_vert[a:a + n]
        else:
            if typ == G.mjGEOM_BOX:
                hx, hy, hz = size
            elif typ == G.mjGEOM_SPHERE:
                hx = hy = hz = size[0]
            elif typ in (G.mjGEOM_CYLINDER, G.mjGEOM_CAPSULE):
                hx = hy = size[0]
                hz = size[1] + (size[0] if typ == G.mjGEOM_CAPSULE else 0)
            elif typ == G.mjGEOM_ELLIPSOID:
                hx, hy, hz = size
            else:
                continue
            local = np.array([[sx * hx, sy * hy, sz * hz]
                              for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        world = local @ R.T + pos
        if world[:, 2].max() < zmin or world[:, 2].min() > zmax:
            continue
        # clip to the band before projecting (so e.g. a tall column only counts where it is)
        footprints.append(world[:, :2])

    if not footprints:
        raise SystemExit('no obstacle geoms found in ' + path)
    allpts = np.vstack(footprints)
    x0, y0 = allpts.min(axis=0) - margin
    x1, y1 = allpts.max(axis=0) + margin
    # avoid giant maps if a huge ground box exists: cap at 80 m
    if (x1 - x0) > 150 or (y1 - y0) > 150:
        print('warning: world extent > 150 m, check for large non-plane floor geoms', file=sys.stderr)
    grid = Grid(x0, y0, x1, y1, res)
    for fp in footprints:
        hull = _convex_hull(fp)
        # a floor slab (large and thin, top below zmin) was filtered above; skip footprints that
        # cover more than 30 % of the map (a ceiling or a slab we could not classify)
        area = (np.ptp(fp[:, 0]) * np.ptp(fp[:, 1]))
        if area > 0.3 * (x1 - x0) * (y1 - y0):
            continue
        grid.fill_polygon(hull)
    return grid


# ---------------------------------------------------------------------------------------------
# Backend 2: hospital-sim-env/isaac/scene/layout.py (approximate walls + doors, wing-local frame)
# ---------------------------------------------------------------------------------------------
def layout_walls():
    """Wall segments (x0, y0, x1, y1) of the surgical wing in wing-local metres, doors cut out,
    plus a few labelled furniture boxes. Approximation of hospital-sim-env/isaac/scene/layout.py for the mock robot."""
    rooms = {
        'Link': (0, -6, 2, 0), 'PreOp': (-10, -6, 0, 0), 'PACU': (2, -6, 14, 0),
        'Corridor': (-12, -10, 14, -6),
        'Store': (-12, -18, -4, -10), 'Scrub': (-4, -18, 0, -10), 'Theatre': (0, -20, 10, -10),
    }
    # openings on wall lines: (axis, line coordinate, a, b)
    openings = [
        ('h', -6, 0, 2),        # link <-> corridor (fully open)
        ('h', -6, -5, -3),      # pre-op double door
        ('h', -6, 7, 9),        # PACU double door
        ('h', -10, -9, -7),     # store door
        ('h', -10, -3, -1),     # scrub door
        ('h', -10, 7, 9),       # theatre door
        ('v', -4, -15, -13),    # scrub <-> store door (position approximate)
        ('v', 0, -17, -15),     # scrub <-> theatre door (position approximate)
    ]
    segs = []
    for (x0, y0, x1, y1) in rooms.values():
        segs += [('h', y0, x0, x1), ('h', y1, x0, x1), ('v', x0, y0, y1), ('v', x1, y0, y1)]
    # the link corridor's east/west walls and the shared walls appear twice; harmless
    out = []
    for axis, c, a, b in segs:
        pieces = [(a, b)]
        for oax, oc, oa, ob in openings:
            if oax != axis or abs(oc - c) > 1e-6:
                continue
            nxt = []
            for pa, pb in pieces:
                if ob <= pa or oa >= pb:
                    nxt.append((pa, pb))
                    continue
                if oa > pa:
                    nxt.append((pa, oa))
                if ob < pb:
                    nxt.append((ob, pb))
            pieces = nxt
        for pa, pb in pieces:
            out.append((pa, c, pb, c) if axis == 'h' else (c, pa, c, pb))
    # the link's north end (y=0, x 0..2) opens into the hospital, which is not modelled: close it
    # 2 m further north so the robot cannot leave the map
    out += [(0, 2, 2, 2), (0, 0, 0, 2), (2, 0, 2, 2)]
    out = [s for s in out if not (s[1] == 0 and s[3] == 0 and s[0] >= 0 and s[2] <= 2)]
    boxes = [  # (x0, y0, x1, y1) approximate furniture footprints
        (-9.9, -6.8, -9.1, -6.1), (-7.4, -6.8, -6.6, -6.1),       # corridor gurneys (north wall)
        (3.6, -6.6, 4.4, -6.1), (11.0, -9.9, 11.8, -9.4),         # corridor carts
        (4.2, -16.0, 5.8, -14.0),                                 # theatre operating table
        (-11.8, -17.9, -6.0, -17.4), (-9.3, -14.5, -7.7, -13.5),  # store shelves + prep island
        (-9.0, -1.2, -7.0, -0.1), (-5.0, -1.2, -3.0, -0.1),       # pre-op beds
        (4.0, -1.2, 6.0, -0.1), (8.0, -1.2, 10.0, -0.1),          # PACU beds
    ]
    return out, boxes


def rasterize_layout(res=RES, wall_t=0.10):
    segs, boxes = layout_walls()
    grid = Grid(-13.0, -21.0, 15.0, 3.0, res)
    for x0, y0, x1, y1 in segs:
        if y0 == y1:
            grid.fill_rect(x0 - wall_t / 2, y0 - wall_t / 2, x1 + wall_t / 2, y0 + wall_t / 2)
        else:
            grid.fill_rect(x0 - wall_t / 2, y0 - wall_t / 2, x0 + wall_t / 2, y1 + wall_t / 2)
    for b in boxes:
        grid.fill_rect(*b)
    return grid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--mjcf', help='MuJoCo scene MJCF of the hospital world')
    src.add_argument('--layout', action='store_true', help='approximate map from hospital-sim-env/isaac/scene/layout.py')
    ap.add_argument('--out', required=True, help='output path without extension')
    ap.add_argument('--res', type=float, default=RES)
    ap.add_argument('--zmin', type=float, default=0.08)
    ap.add_argument('--zmax', type=float, default=1.80)
    ap.add_argument('--robot-root', default='pelvis')
    a = ap.parse_args()
    if a.mjcf:
        grid = rasterize_mjcf(a.mjcf, a.zmin, a.zmax, robot_root=a.robot_root, res=a.res)
        comment = 'mjcf ' + os.path.basename(a.mjcf)
    else:
        grid = rasterize_layout(a.res)
        comment = 'hospital-sim-env/isaac/scene/layout.py approx'
    y = grid.save(a.out, comment=comment)
    print(f'wrote {y} ({grid.w}x{grid.h} @ {grid.res} m, origin {grid.x0:.2f},{grid.y0:.2f})')


if __name__ == '__main__':
    main()
