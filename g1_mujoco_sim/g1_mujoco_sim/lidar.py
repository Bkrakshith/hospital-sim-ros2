"""Simulated Livox MID-360 (non-repetitive scan) and a 2-D LaserScan slice, via mujoco.mj_multiRay.

MID-360: 360 deg horizontal, -7..+52 deg vertical FOV (sensor frame), 200 k pts/s, 10 Hz frames, 4 lines.
The real scan pattern is a non-repetitive rosette; here each frame is a low-discrepancy (golden-ratio)
sequence over the FOV with a per-frame phase shift, so coverage keeps densifying over successive frames
like the real sensor. Rays are cast in slices every physics step from the sensor's pose at that instant
(so fast motion distorts the cloud exactly as with the real, time-sliced sensor).

Only static world geoms (geom group `world_group`) are hit, never the robot.
"""
import math

import numpy as np
import mujoco

GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
PLASTIC = 0.7548776662466927  # plastic number based R2 sequence constant


class Mid360:
    def __init__(self, m, points_per_frame=10000, rate_hz=10.0, sim_dt=0.002, fov_down_deg=-7.0,
                 fov_up_deg=52.0, range_min=0.1, range_max=40.0, world_group=0):
        self.m = m
        self.n = int(points_per_frame)
        self.steps_per_frame = max(1, int(round(1.0 / rate_hz / sim_dt)))
        self.per_step = int(math.ceil(self.n / self.steps_per_frame))
        self.rmin, self.rmax = range_min, range_max
        self.group = np.zeros(6, np.uint8)
        self.group[world_group] = 1
        lo, hi = math.radians(fov_down_deg), math.radians(fov_up_deg)
        # R2 low-discrepancy points in [0,1)^2 -> azimuth, sin(elevation) (uniform over the sphere band)
        i = np.arange(self.n)
        u = (0.5 + i / PLASTIC) % 1.0
        v = (0.5 + i / PLASTIC ** 2) % 1.0
        self.az0 = 2 * np.pi * u
        s_lo, s_hi = math.sin(lo), math.sin(hi)
        self.el = np.arcsin(s_lo + (s_hi - s_lo) * v)
        self.line = (i % 4).astype(np.uint8)
        self.frame = 0
        self.cursor = 0
        self._reset_buffers()

    def _reset_buffers(self):
        self.buf_xyz = np.zeros((self.n, 3), np.float32)
        self.buf_ok = np.zeros(self.n, bool)
        self.buf_t = np.zeros(self.n, np.float64)
        self.buf_geom = np.zeros(self.n, np.int32)

    def _dirs(self, idx):
        az = self.az0[idx] + self.frame * GOLDEN * 2 * np.pi / 97.0
        el = self.el[idx]
        ce = np.cos(el)
        return np.stack([ce * np.cos(az), ce * np.sin(az), np.sin(el)], 1)

    def step(self, d, pos, R, t):
        """Cast this step's slice from sensor pose (pos, R world-from-sensor). Returns a finished frame
        dict when the frame completes, else None."""
        a = self.cursor
        b = min(self.n, a + self.per_step)
        if b > a:
            idx = np.arange(a, b)
            dl = self._dirs(idx)
            dw = dl @ R.T
            k = b - a
            geomid = np.zeros(k, np.int32)
            dist = np.zeros(k, np.float64)
            mujoco.mj_multiRay(self.m, d, pos, dw.ravel().copy(), self.group, 1, -1, geomid, dist, None, k,
                               self.rmax)
            ok = (dist >= self.rmin) & (dist <= self.rmax) & (geomid >= 0)
            # store in sensor frame at capture time (like the real sensor)
            self.buf_xyz[a:b] = (dl * dist[:, None]).astype(np.float32)
            self.buf_ok[a:b] = ok
            self.buf_t[a:b] = t
            self.buf_geom[a:b] = geomid
            self.cursor = b
        self._step_in_frame = getattr(self, "_step_in_frame", 0) + 1
        if self._step_in_frame >= self.steps_per_frame:
            ok = self.buf_ok
            out = {"xyz": self.buf_xyz[ok].copy(), "t": self.buf_t[ok].copy(), "line": self.line[ok].copy(),
                   "geom": self.buf_geom[ok].copy(), "t0": float(self.buf_t[0])}
            self.frame += 1
            self.cursor = 0
            self._step_in_frame = 0
            self._reset_buffers()
            return out
        return None


class Scan2D:
    """Planar 360 deg scan in the sensor's x-y plane (a 2-D slice as pointcloud_to_laserscan would give)."""

    def __init__(self, m, beams=720, range_min=0.1, range_max=30.0, world_group=0):
        self.m = m
        self.beams = beams
        self.rmin, self.rmax = range_min, range_max
        self.group = np.zeros(6, np.uint8)
        self.group[world_group] = 1
        self.angle_min = -math.pi
        self.inc = 2 * math.pi / beams
        a = self.angle_min + self.inc * np.arange(beams)
        self.dl = np.stack([np.cos(a), np.sin(a), np.zeros(beams)], 1)

    def cast(self, d, pos, R):
        dw = self.dl @ R.T
        geomid = np.zeros(self.beams, np.int32)
        dist = np.zeros(self.beams, np.float64)
        mujoco.mj_multiRay(self.m, d, pos, dw.ravel().copy(), self.group, 1, -1, geomid, dist, None,
                           self.beams, self.rmax)
        r = np.where((geomid >= 0) & (dist >= self.rmin) & (dist <= self.rmax), dist, np.inf)
        return r.astype(np.float32)
