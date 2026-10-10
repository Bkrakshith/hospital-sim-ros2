"""Free objects of the sim world: ground-truth poses (/sim/objects), object reset with rack offsets, the simulated
surgeon grip, and the fixed-base weld. Pure MuJoCo/numpy (no ROS) so the offline skill harness and tests share it.

GROUND TRUTH = CHEAT. /sim/objects exists only in simulation; the mission reads it through an ObjectSource interface
that the perception pipeline (/g1/perception/detections) replaces.

Simulated surgeon grip (CHEAT, stands in for a real person closing their hand): when an instrument's `<name>_handle`
site has stayed within `grip_radius` of the `surgeon_hand` site, moving slower than `grip_speed` (speed low-pass
filtered over ~0.3 s: an instrument held in a robot hand micro-vibrates at a few cm/s), for `grip_dwell` seconds, the inactive weld `grip_<name>` (surgeon_hand <-> instrument) is activated at the current relative pose.
The robot must then open its hand; the instrument stays with the surgeon until reset_objects(), or until the next
instrument is offered: the surgeon stand-in holds one instrument at a time, and once another instrument that is out
of its rack slot comes within `put_down_radius` of the hand, the held one is put back in its rack slot (a sim
shortcut standing in for "the surgeon hands the used instrument back when the next one arrives"). It must happen
before the new grip, not on it: the held instrument lies welded along the previous handover axis, which is where the
next instrument arrives, and it blocked the handle short of grip_radius (angled_tweezers after scissors).
"""
import json
import math
import os

import numpy as np
import mujoco

OBJ = mujoco.mjtObj


def _quat_z(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _rotate(q, v):
    return _qmul(_qmul(q, np.r_[0.0, v]), _qconj(q))[1:]


def weld_to_world(m, d, eq_name, body):
    """Activate weld `eq_name` (body1=`body`, body2=world) holding `body` at its current world pose.
    MuJoCo weld data = [anchor (weld point in body2 coordinates), relpos (the same point in body1 coordinates),
    relquat (body2 orientation in body1's frame), torquescale]. The weld point must be at the body: at the world
    origin (tens of metres away) a milliradian of yaw error becomes centimetres of drift."""
    eid = mujoco.mj_name2id(m, OBJ.mjOBJ_EQUALITY, eq_name)
    if eid < 0:
        raise ValueError(f"model has no equality '{eq_name}' (rebuild g1-mujoco-sim with build_world.py)")
    b = mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, body)
    p, q = d.xpos[b].copy(), d.xquat[b].copy()
    m.eq_data[eid, 0:3] = p                 # weld point in world (body2) coordinates = the pelvis origin
    m.eq_data[eid, 3:6] = 0.0               # ... = the origin of body1
    m.eq_data[eid, 6:10] = _qconj(q)
    d.eq_active[eid] = 1
    return eid


def set_weld_world_pose(m, eid, pos, quat):
    """Move the world target of a weld made by weld_to_world (fixed-base glide)."""
    m.eq_data[eid, 0:3] = pos
    m.eq_data[eid, 3:6] = 0.0
    m.eq_data[eid, 6:10] = _qconj(np.asarray(quat, float))


class BaseGlide:
    """FIXED-BASE CHEAT: moves the welded pelvis from its current pose to a goal pose (x, y, yaw) along a straight
    line (no collision checking) at `speed` m/s / `yaw_rate` rad/s, lifted `lift` m mid-way so the feet clear the
    floor. It stands in for walking (Nav2 + locomotion policy) while arm skills are developed with the base welded;
    whatever the hands hold travels along physically."""

    def __init__(self, speed=0.3, yaw_rate=0.6, lift=0.03):
        self.speed, self.yaw_rate, self.lift = speed, yaw_rate, lift
        self.seg = None

    @property
    def active(self):
        return self.seg is not None

    def start(self, t, p0, yaw0, goal_xy, goal_yaw):
        dyaw = (goal_yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
        dist = math.hypot(goal_xy[0] - p0[0], goal_xy[1] - p0[1])
        dur = max(0.5, dist / self.speed, abs(dyaw) / self.yaw_rate)
        self.seg = {"t0": t, "dur": dur, "p0": np.array(p0, float), "yaw0": yaw0, "dyaw": dyaw,
                    "p1": np.array([goal_xy[0], goal_xy[1], p0[2]], float), "dist": dist}

    def step(self, m, eid, t):
        """Returns remaining distance (m) while gliding, None when idle. Call every physics step."""
        g = self.seg
        if g is None:
            return None
        s = min(1.0, (t - g["t0"]) / g["dur"])
        a = s * s * (3 - 2 * s)
        p = g["p0"] + a * (g["p1"] - g["p0"])
        p[2] += self.lift * math.sin(math.pi * s)
        yaw = g["yaw0"] + a * g["dyaw"]
        set_weld_world_pose(m, eid, p, _quat_z(yaw))
        if s >= 1.0:
            self.seg = None
            return 0.0
        return (1.0 - a) * g["dist"]


class SceneObjects:
    # m: an instrument out of its rack slot (see OUT_OF_SLOT) whose handle end comes this close to the surgeon's hand
    # makes the surgeon put the held one down. The robot's wrist touched the held scissors with the next handle
    # 0.22 m from the hand (arrival at theatre_surgeon_side), so this triggers during the last part of the approach.
    PUT_DOWN_RADIUS = 0.40
    OUT_OF_SLOT = 0.05          # m from its (offset) rack pose: "out of the rack", i.e. carried by the robot

    def __init__(self, m, setup_path=None, grip_radius=0.045, grip_speed=0.08, grip_dwell=0.6, speed_tau=0.3,
                 soften_hand=True, soft_radius=0.22, put_down_radius=PUT_DOWN_RADIUS):
        self.soft_radius = soft_radius
        self.put_down_radius = put_down_radius
        self.m = m
        self.setup = {}
        if setup_path and os.path.exists(setup_path):
            with open(setup_path) as f:
                self.setup = json.load(f)
        inst = set(self.setup.get("instruments", {}))
        self.bodies = {}            # name -> (body id, qpos adr, dof adr, type)
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            b = m.jnt_bodyid[j]
            name = mujoco.mj_id2name(m, OBJ.mjOBJ_BODY, b)
            if name == "pelvis" or name is None:
                continue
            typ = "instrument" if name in inst else ("rack" if "rack" in name else "object")
            self.bodies[name] = (b, m.jnt_qposadr[j], m.jnt_dofadr[j], typ)
        self.hand_site = mujoco.mj_name2id(m, OBJ.mjOBJ_SITE, "surgeon_hand")
        # handle end = the instrument point the surgeon receives: site <name>_handle (stand-ins) or +z * handle_z
        # in the object frame (hospital-sim-env/objects assets, handle_z from theatre_setup.json)
        self.handle_site = {n: mujoco.mj_name2id(m, OBJ.mjOBJ_SITE, f"{n}_handle") for n in self.bodies}
        self.handle_z = {n: float(e.get("handle_z", 0.0)) for n, e in self.setup.get("instruments", {}).items()}
        self.grip_eq = {n: mujoco.mj_name2id(m, OBJ.mjOBJ_EQUALITY, f"grip_{n}") for n in self.bodies}
        self.grip_eq = {n: e for n, e in self.grip_eq.items() if e >= 0}
        self.q0 = {n: m.qpos0[qa:qa + 7].copy() for n, (_, qa, _, _) in self.bodies.items()}
        self.rack = next((n for n, v in self.bodies.items() if v[3] == "rack"), None)
        # a static rack (hospital-sim-env/objects tool_rack has no joint) is moved through body_pos/body_quat for offsets
        rb = mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, "tool_rack")
        self.static_rack = rb if (rb >= 0 and self.rack is None and m.body_jntnum[rb] == 0) else -1
        if self.static_rack >= 0:
            self.rack_pos0 = m.body_pos[rb].copy()
            self.rack_quat0 = m.body_quat[rb].copy()
        self.grip_radius, self.grip_speed, self.grip_dwell = grip_radius, grip_speed, grip_dwell
        self.soft_geoms = self._soften_surgeon_hand() if soften_hand else []
        self.dwell = {n: 0.0 for n in self.grip_eq}
        self.speed = {n: 0.0 for n in self.grip_eq}
        self.speed_tau = speed_tau
        self.held = None
        self.offset = (0.0, 0.0, 0.0)

    def _soften_surgeon_hand(self):
        """CHEAT (with the simulated grip): the surgeon stand-in's hand and forearm do not collide. A rigid mannequin
        hand cannot yield or close around a handle, so the robot presents into it and the weld does the gripping.
        Every collidable geom of the hand's body within soft_radius of the surgeon_hand site is made contact-free.
        Returns the geom ids changed."""
        m = self.m
        if self.hand_site < 0:
            return []
        d = mujoco.MjData(m)
        mujoco.mj_kinematics(m, d)
        hb = m.site_bodyid[self.hand_site]
        hp = d.site_xpos[self.hand_site]
        ids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == hb and (m.geom_contype[g] or m.geom_conaffinity[g])
               and np.linalg.norm(d.geom_xpos[g] - hp) < self.soft_radius]
        for g in ids:
            m.geom_contype[g] = 0
            m.geom_conaffinity[g] = 0
        return ids

    @property
    def has_surgeon(self):
        return self.hand_site >= 0 and bool(self.grip_eq)

    # ------------------------------------------------------------------ ground truth
    def snapshot(self, d):
        """{name: {"pose": [x,y,z,qw,qx,qy,qz], "type": ...}} (+ "held_by" for instruments, surgeon hand entry)."""
        out = {}
        for n, (b, qa, _, typ) in self.bodies.items():
            e = {"pose": [round(float(v), 5) for v in d.qpos[qa:qa + 7]], "type": typ}
            if n in self.grip_eq:
                e["held_by"] = "surgeon" if d.eq_active[self.grip_eq[n]] else None
            out[n] = e
        if self.static_rack >= 0:
            out["tool_rack"] = {"pose": [round(float(v), 5) for v in np.r_[d.xpos[self.static_rack],
                                                                              d.xquat[self.static_rack]]],
                                "type": "rack", "static": True}
        if self.hand_site >= 0:
            p = d.site_xpos[self.hand_site]
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, d.site_xmat[self.hand_site])
            out["surgeon_hand"] = {"pose": [round(float(v), 5) for v in np.r_[p, q]], "type": "surgeon_hand",
                                   "holding": self.held}
        return out

    def to_json(self, d):
        return json.dumps(self.snapshot(d), separators=(",", ":"))

    # ------------------------------------------------------------------ simulated surgeon grip (CHEAT)
    def update_grip(self, d, dt):
        """Call every step (or every k steps with dt = k * timestep). Returns the name just gripped, else None."""
        if self.hand_site < 0:
            return None
        hp = d.site_xpos[self.hand_site]
        for n, eid in self.grip_eq.items():
            if n == self.held:
                continue
            hp_n = self.handle_point(d, n)
            if hp_n is None:
                continue
            if self.held is not None and np.linalg.norm(hp_n - hp) < self.put_down_radius and self._out_of_slot(d, n):
                self._return_to_rack(d, self.held)          # the next instrument is offered: hand free first
                self.held = None
            _, _, da, _ = self.bodies[n]
            a = min(1.0, dt / self.speed_tau)
            self.speed[n] += a * (float(np.linalg.norm(d.qvel[da:da + 3])) - self.speed[n])
            close = np.linalg.norm(hp_n - hp) < self.grip_radius
            slow = self.speed[n] < self.grip_speed
            self.dwell[n] = self.dwell[n] + dt if (close and slow) else 0.0
            if self.dwell[n] >= self.grip_dwell:
                if self.held is not None:
                    self._return_to_rack(d, self.held)
                self._activate_grip(d, n, eid)
                self.held = n
                self.dwell = {k: 0.0 for k in self.dwell}
                return n
        return None

    def handle_point(self, d, name):
        s = self.handle_site.get(name, -1)
        if s >= 0:
            return d.site_xpos[s]
        hz = self.handle_z.get(name, 0.0)
        if hz <= 0:
            return None
        b = self.bodies[name][0]
        return d.xpos[b] + d.xmat[b].reshape(3, 3)[:, 2] * hz

    def _activate_grip(self, d, name, eid):
        m = self.m
        b1 = m.eq_obj1id[eid]            # surgeon_hand body
        b2 = m.eq_obj2id[eid]            # instrument
        q1, p1 = d.xquat[b1], d.xpos[b1]
        q2, p2 = d.xquat[b2], d.xpos[b2]
        # relpose = pose of body2 in body1's frame
        q1c = _qconj(q1)
        m.eq_data[eid, 0:3] = 0.0
        m.eq_data[eid, 3:6] = _rotate(q1c, p2 - p1)
        m.eq_data[eid, 6:10] = _qmul(q1c, q2)
        d.eq_active[eid] = 1

    def _slot_pose(self, name):
        """qpos (7) of `name` in its rack slot, with the current rack offset."""
        dx, dy, dyaw = self.offset
        c = self._rack_centre0()
        qz = _quat_z(dyaw)
        q = self.q0[name].copy()
        q[:3] = c + _rotate(qz, q[:3] - c) + np.array([dx, dy, 0.0])
        q[3:7] = _qmul(qz, q[3:7])
        return q

    def _out_of_slot(self, d, name):
        qa = self.bodies[name][1]
        return float(np.linalg.norm(d.qpos[qa:qa + 3] - self._slot_pose(name)[:3])) > self.OUT_OF_SLOT

    def _return_to_rack(self, d, name):
        d.eq_active[self.grip_eq[name]] = 0
        _, qa, da, _ = self.bodies[name]
        d.qpos[qa:qa + 7] = self._slot_pose(name)
        d.qvel[da:da + 6] = 0.0

    def _rack_centre0(self):
        if self.rack:
            return self.q0[self.rack][:3]
        if self.static_rack >= 0:
            return self.rack_pos0.copy()
        return np.zeros(3)

    # ------------------------------------------------------------------ reset
    def reset(self, d, offset=(0.0, 0.0, 0.0), jitter=0.0, rng=None):
        """Put every free object back at its initial pose, releasing grips. `offset` = (dx, dy, dyaw) applied rigidly
        to the rack and everything that started in it (about the rack's initial centre, world frame). `jitter` adds
        a uniform per-instrument xy offset (m) *inside* the slot clearance (0 keeps them centred)."""
        rng = rng or np.random.default_rng()
        for eid in self.grip_eq.values():
            d.eq_active[eid] = 0
        self.held = None
        self.dwell = {n: 0.0 for n in self.grip_eq}
        dx, dy, dyaw = offset
        self.offset = (float(dx), float(dy), float(dyaw))
        c = self._rack_centre0()
        qz = _quat_z(dyaw)
        if self.static_rack >= 0:
            self.m.body_pos[self.static_rack] = c + _rotate(qz, self.rack_pos0 - c) + np.array([dx, dy, 0.0])
            self.m.body_quat[self.static_rack] = _qmul(qz, self.rack_quat0)
        for n, (b, qa, da, typ) in self.bodies.items():
            q = self.q0[n].copy()
            if typ in ("rack", "instrument"):
                rel = q[:3] - c
                q[:3] = c + _rotate(qz, rel) + np.array([dx, dy, 0.0])
                q[3:7] = _qmul(qz, q[3:7])
                if typ == "instrument" and jitter > 0:
                    q[:2] += rng.uniform(-jitter, jitter, 2)
            d.qpos[qa:qa + 7] = q
            d.qvel[da:da + 6] = 0.0
        mujoco.mj_forward(self.m, d)
