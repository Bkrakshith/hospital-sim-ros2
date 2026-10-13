"""Moving building fixtures in the MuJoCo sim: fire doors, bed-lift doors and buttons, in the whole hospital.

The control laws live in hospital-sim-env/fixtures/hospital_fixtures.py (`FixtureController`), shared with
Isaac Sim, so both simulators behave the same. This adapter maps the controller's joints onto the MuJoCo model
(worlds/fixtures*.json = the spec g1-mujoco-sim/build_world.py built the bodies from), casts the light curtains of
every lift of every bank (wing: Lift1, Lift2; main building: MainLift1, MainLift2) with mj_ray and writes the joint
efforts into d.qfrc_applied. Pure MuJoCo/numpy (no ROS), so tests use it directly.

Fixture tiers (hospital_fixtures.TIERS: wing, tier1, tier2, full) decide which doors and lifts the model has.
build_world.py writes one scene per tier (<world>_g1_<tier>.xml) and records the tier and the spec file in the world
MJCF (<custom><text name="hg1_fixtures_tier"/"hg1_fixtures_spec">). resolve_tier() and scene_for_tier() pick the
scene; default_paths() reads the spec name from the model.
"""
import importlib.util
import json
import os

import numpy as np
import mujoco


def load_fixtures_module(path):
    """Import hospital_fixtures.py by path (it is not a ROS package)."""
    spec = importlib.util.spec_from_file_location("hospital_fixtures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TIER_ENV = "HG1_FIXTURES"       # same variable as hospital_fixtures.TIER_ENV (all consumers use one tier)
DEFAULT_TIER = "tier2"
TIER_NAMES = ("wing", "tier1", "tier2", "full")


def resolve_tier(tier=None):
    """Tier name from the argument (e.g. the node parameter), else $HG1_FIXTURES, else tier2."""
    t = (tier or "").strip() or os.environ.get(TIER_ENV) or DEFAULT_TIER
    if t not in TIER_NAMES:
        raise ValueError(f"unknown fixture tier {t!r}: use one of {list(TIER_NAMES)}")
    return t


def scene_for_tier(sim_dir, world, tier):
    """Scene MJCF of a world at a fixture tier: <sim_dir>/<world>_g1_<tier>.xml when build_world.py wrote it,
    else <sim_dir>/<world>_g1.xml (worlds without fixtures, such as flat, have one scene only)."""
    p = os.path.join(sim_dir, f"{world}_g1_{tier}.xml")
    return p if os.path.exists(p) else os.path.join(sim_dir, f"{world}_g1.xml")


def model_text(m, name):
    """A <custom><text> value of the model, or None."""
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXT, name)
    if i < 0:
        return None
    a = m.text_adr[i]
    return bytes(m.text_data[a:a + m.text_size[i] - 1]).decode()


def model_fixtures_info(m):
    """(tier, spec file name) recorded by build_world.py in the model, or (None, None) for an older model."""
    return model_text(m, "hg1_fixtures_tier"), model_text(m, "hg1_fixtures_spec")


def default_paths(root, model_path, m=None):
    """(spec path, controller module path). The spec is worlds/<hg1_fixtures_spec> of the model m when it records
    one, else worlds/fixtures.json next to the scene."""
    name = (model_fixtures_info(m)[1] if m is not None else None) or "fixtures.json"
    return (os.path.join(os.path.dirname(model_path), "worlds", name),
            os.path.join(root, "hospital-sim-env", "fixtures", "hospital_fixtures.py"))


class MjFixtures:
    """One FixtureController on a MuJoCo model. Call `step(d, dt)` before every mj_step."""

    def __init__(self, m, spec, F):
        self.m, self.spec, self.F = m, spec, F
        self.names = F.joint_names(spec)
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.names]
        missing = [n for n, j in zip(self.names, jid) if j < 0]
        if missing:
            raise ValueError(f"model lacks fixture joints {missing[:3]}{'...' if len(missing) > 3 else ''}")
        jid = np.array(jid)
        self.qadr = m.jnt_qposadr[jid].copy()
        self.dadr = m.jnt_dofadr[jid].copy()
        self.ctl = F.FixtureController(spec)
        self.origin = np.array(spec["wing_origin"], float)
        self._geomid = np.zeros(1, np.int32)
        # Curtain pre-filter: only geoms on moving bodies can enter a doorway. Static world geoms and the fixtures'
        # own bodies are left out; the rays are cast only when a candidate's bounding sphere reaches the curtain.
        fx_bodies = {m.jnt_bodyid[j] for j in jid}
        self._dyn = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] != 0
                              and m.geom_bodyid[g] not in fx_bodies and m.geom_rbound[g] > 0], np.int32)
        self._rb = m.geom_rbound[self._dyn].copy()
        self.n_rays = 0   # rays cast in the last step (for timing)

    def q(self, d):
        return dict(zip(self.names, d.qpos[self.qadr].tolist()))

    def button_names(self):
        """Short names of every button of the spec (landing calls of every bank, car Open/Close buttons)."""
        return {b["joint"][len(self.F.PREFIX):] for b in self.ctl.buttons}

    def reset(self, d):
        """Controller reset and joints to initial_q() (held doors open on their magnets, all else closed).
        Call between mj_resetData and mj_forward."""
        self.ctl.reset()
        q0 = self.ctl.initial_q()
        d.qpos[self.qadr] = [q0[n] for n in self.names]
        d.qvel[self.dadr] = 0.0
        d.qfrc_applied[self.dadr] = 0.0

    def curtain_blocked(self, d, lift_name, q):
        """True if any light-curtain ray across the lift's door gap hits a geom (all groups: an optical beam)."""
        rays = self.F.curtain_rays(self.spec, lift_name, q)
        if not rays:
            return False
        (x, y0, z0), _, length = rays[0]
        z1 = rays[-1][0][2]
        x, y0 = x + self.origin[0], y0 + self.origin[1]
        p, r = d.geom_xpos[self._dyn], self._rb
        near = ((np.abs(p[:, 0] - x) < r) & (p[:, 1] + r > y0) & (p[:, 1] - r < y0 + length)
                & (p[:, 2] + r > z0 + self.origin[2]) & (p[:, 2] - r < z1 + self.origin[2]))
        if not near.any():
            return False
        for o, u, length in rays:
            pnt = np.array(o, float) + self.origin
            dist = mujoco.mj_ray(self.m, d, pnt, np.array(u, float), None, 1, -1, self._geomid)
            self.n_rays += 1
            if 0.0 <= dist <= length:
                return True
        return False

    def step(self, d, dt):
        names = self.names
        q = dict(zip(names, d.qpos[self.qadr].tolist()))
        qd = dict(zip(names, d.qvel[self.dadr].tolist()))
        self.n_rays = 0
        blocked = {}
        for name, lift in self.ctl.lifts.items():
            if lift.state in ("open", "closing"):   # the controller only reads the curtain in these states
                blocked[name] = self.curtain_blocked(d, name, q)
        tau = self.ctl.step(dt, q, qd, blocked)
        # These dofs belong to the fixtures only; nothing else writes qfrc_applied there, so assign (adding would
        # accumulate: MuJoCo never clears qfrc_applied).
        d.qfrc_applied[self.dadr] = [tau[n] for n in names]

    def state_json(self, d):
        return self.ctl.state_json(self.q(d))


def load(m, spec_path, module_path):
    """MjFixtures for model m, or (None, reason) when the model or the files have no fixtures."""
    if not os.path.exists(spec_path):
        return None, f"no fixture spec at {spec_path}"
    if not os.path.exists(module_path):
        return None, f"no fixture controller at {module_path}"
    spec = json.load(open(spec_path))
    F = load_fixtures_module(module_path)
    try:
        return MjFixtures(m, spec, F), None
    except ValueError as e:
        return None, str(e)
