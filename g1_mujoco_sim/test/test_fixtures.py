"""Fire doors and bed lifts of the whole hospital (surgical wing and main building) in the real MuJoCo scene
(robot + world), driven by the shared controller (hospital-sim-env/fixtures/hospital_fixtures.py) through
g1_mujoco_sim.fixtures. Pure MuJoCo, no ROS.

    taskset -c 0-7,10-31 python -m pytest -q -s ros2_ws/src/hospital-sim-ros2/g1_mujoco_sim/test/test_fixtures.py

The dynamic tests use the scene of the "full" tier (g1-mujoco-sim/hospital_g1_full.xml: every door of every group,
both lift banks, both calls). The spec tests check every tier. Also: the spawn poses of worlds/spawns.json are
collision-free, and a 2.2 m bed footprint passes the 4 m link from the hospital to the wing corridor.
"""
import json
import math
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
mujoco = pytest.importorskip("mujoco")
from g1_mujoco_sim import fixtures as FX  # noqa: E402
from g1_mujoco_sim.model_map import ModelMap  # noqa: E402

# master repo root: $HG1_ROOT (env.sh), else from this file (ros2_ws/src/hospital-sim-ros2/g1_mujoco_sim/test/)
ROOT = os.environ.get("HG1_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
SIM = os.path.join(ROOT, "g1-mujoco-sim")
TIER = "full"
SCENE = os.path.join(SIM, f"hospital_g1_{TIER}.xml")
_, MODULE = FX.default_paths(ROOT, SCENE)
pytestmark = pytest.mark.skipif(not os.path.exists(SCENE), reason="run g1-mujoco-sim/build_world.py first")
BOX_HALF = 0.15
BED = (1.1, 0.5, 0.45)     # half sizes of the bed probe: 2.2 m x 1.0 m footprint, z 0.05 .. 0.95
PARK = (15.0, -11.0)       # wing-local test-box parking spot (wing lift lobby corner)
FAR = (80.0, -60.0)        # world xy far outside the building: robot and bed probe parked there on the floor

# expected spec per tier (WP1 contract 4.1): 2-leaf doors, 1-leaf doors, lifts, calls, joints
TIERS = {"wing": (7, 0, 2, 1, 23), "tier1": (9, 2, 4, 2, 38), "tier2": (12, 2, 4, 2, 44), "full": (12, 10, 4, 2, 52)}


def scene_spec(scene):
    """Scene with one free 0.3 m test box and a free bed probe (both parked out of the way), compiled."""
    s = mujoco.MjSpec.from_file(scene)
    b = s.worldbody.add_body(name="test_box", pos=[0, 0, 0])
    b.add_freejoint(name="test_box")
    b.add_geom(name="test_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[BOX_HALF] * 3, mass=5.0)
    p = s.worldbody.add_body(name="bed_probe", pos=[0, 0, 0])
    p.add_freejoint(name="bed_probe")
    p.add_geom(name="bed_probe", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(BED), mass=50.0)
    return s.compile()


@pytest.fixture(scope="module")
def model():
    return scene_spec(SCENE)


class Rig:
    def __init__(self, m, scene=SCENE):
        self.m, self.d = m, mujoco.MjData(m)
        spec_path, module = FX.default_paths(ROOT, scene, m)
        self.fx, why = FX.load(m, spec_path, module)
        assert self.fx is not None, why
        self.F, self.spec = self.fx.F, self.fx.spec
        self.ox, self.oy, _ = self.spec["wing_origin"]
        self.free = {}
        for n in ("test_box", "bed_probe"):
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            self.free[n] = (m.jnt_qposadr[j], m.jnt_dofadr[j])
        mujoco.mj_resetData(m, self.d)
        self.fx.reset(self.d)
        self.M = ModelMap(m)
        self.d.qpos[self.M.free_q:self.M.free_q + 3] = [FAR[0], FAR[1] + 5.0, 0.79]   # robot out of the way
        self.park_box()
        self.park_bed()
        mujoco.mj_forward(m, self.d)
        self.step_s = []

    def W(self, x, y, z=0.0):
        return np.array([x + self.ox, y + self.oy, z])

    def put(self, body, xyz, yaw=0.0):
        qa, da = self.free[body]
        self.d.qpos[qa:qa + 7] = [*xyz, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.d.qvel[da:da + 6] = 0
        mujoco.mj_forward(self.m, self.d)

    def put_box(self, x, y):
        """Test box at wing-local (x, y)."""
        self.put("test_box", self.W(x, y, BOX_HALF + 0.001))

    def put_box_world(self, x, y):
        self.put("test_box", np.array([x, y, BOX_HALF + 0.001]))

    def park_box(self):
        self.put_box(*PARK)

    def park_bed(self):
        self.put("bed_probe", np.array([FAR[0], FAR[1], BED[2] + 0.001]))

    def run(self, seconds, push=None, until=None):
        dt = self.m.opt.timestep
        for _ in range(int(round(seconds / dt))):
            if push is not None:
                push(self)
            t0 = time.perf_counter()
            self.fx.step(self.d, dt)
            self.step_s.append(time.perf_counter() - t0)
            mujoco.mj_step(self.m, self.d)
            if until is not None and until(self):
                return True
        return until is None

    def q(self, joint):
        return self.fx.q(self.d)[joint]

    def state(self):
        return self.fx.ctl.state(self.fx.q(self.d))

    def body(self, name):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)

    def door(self, name):
        return next(x for x in self.spec["doors"] if x["name"] == name)

    def leaf(self, door, i):
        return self.door(door)["leaves"][i]


@pytest.fixture
def rig(model):
    return Rig(model)


# ---------------------------------------------------------------- spec and model per tier


@pytest.mark.parametrize("tier", list(TIERS))
def test_tier_spec_and_scene(tier):
    """Every tier has its scene and spec; the model has exactly the spec's fixture joints."""
    scene = os.path.join(SIM, f"hospital_g1_{tier}.xml")
    m = mujoco.MjModel.from_xml_path(scene)
    assert FX.model_fixtures_info(m) == (tier, f"fixtures_{tier}.json")
    spec = json.load(open(FX.default_paths(ROOT, scene, m)[0]))
    g = spec["groups"]
    got = (len(g["Doors"]), len(g["Doors1"]), len(g["Lifts"]), len(g["Calls"]))
    F = FX.load_fixtures_module(MODULE)
    names = F.joint_names(spec)
    assert got + (len(names),) == TIERS[tier], got
    fx = sorted(n for n in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt))
                if n and n.startswith("fx_"))
    assert fx == sorted(names)
    assert all(d["n_leaves"] == (2 if d["group"] == "Doors" else 1) == len(d["leaves"]) for d in spec["doors"])
    geoms = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    glass = [g for g in geoms if g.endswith(("_GlassDoor_55", "_GlassDoor2_5"))]
    closed = [g for g in geoms if g.startswith("closed_")]
    # below tier2 the main entrance glass stays closed and D07/D34 keep their closed double doors (as in Isaac);
    # below full the 8 FD30S hazard-room doorways keep the asset's own static leaves (closed_<door>, asset pose)
    assert len(glass) == (2 if tier in ("wing", "tier1") else 0), glass
    L = F.load_layout(os.path.join(ROOT, "hospital-sim-env", "isaac", "scene", "layout.py"))
    hazard = [d for d, v in L.BUILDING_FIRE_DOORS.items() if v.get("replaces") and not v.get("opening")]
    assert len(hazard) == 8, hazard
    want = (["closed_D07", "closed_D34"] if tier in ("wing", "tier1") else []) + \
        ([f"closed_{d}" for d in hazard] if tier != "full" else [])
    assert sorted(closed) == sorted(want), closed
    excluded = set(spec["excluded"]["doors"])
    assert all(c.removeprefix("closed_") in excluded or c in ("closed_D07", "closed_D34") for c in closed), closed
    for c in closed:                       # every closed box is a solid world geom (lidar group 0, collides)
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, c)
        assert m.geom_bodyid[g] == 0 and m.geom_group[g] == 0 and m.geom_contype[g] != 0, c
    print(f"\n{tier}: nq={m.nq} nv={m.nv} nu={m.nu}; doors {got[0]}+{got[1]}, lifts {got[2]}, calls {got[3]}")


def test_default_scene_is_tier2():
    m = mujoco.MjModel.from_xml_path(os.path.join(SIM, "hospital_g1.xml"))
    assert FX.model_fixtures_info(m)[0] == FX.resolve_tier(os.environ.get("HG1_FIXTURES"))


def test_whole_hospital_lists(rig):
    names = [d["name"] for d in rig.spec["doors"]]
    assert names[:7] == ["Compartment", "CrossCorridor", "LiftLobby", "Store", "ScrubStore", "Theatre", "FinalExit"]
    assert {"W_Compartment", "E_Compartment", "W_Stair", "E_Stair", "Main_LiftLobby", "D07_Exit", "D34_Exit",
            "Ward_DirtyUtility", "Central_Store"} <= set(names)
    assert sorted(rig.fx.ctl.lifts) == ["Lift1", "Lift2", "MainLift1", "MainLift2"]
    assert {"LobbyCall", "MainCall", "MainLift1_Open", "MainLift2_Close"} <= rig.fx.button_names()
    held = [lf for d in rig.spec["doors"] for lf in d["leaves"] if lf["hold"] is not None]
    assert len(held) == 14


# ---------------------------------------------------------------- doors


def fixture_contacts(rig):
    bad = []
    for c in rig.d.contact[:rig.d.ncon]:
        names = [mujoco.mj_id2name(rig.m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in (c.geom1, c.geom2)]
        if any(n.startswith("fx_") and rig.m.geom_bodyid[g] != 0 for n, g in zip(names, (c.geom1, c.geom2))):
            bad.append(names)
    return bad


def test_no_fixture_contacts_at_rest(rig):
    """Leaves, panels and buttons of the whole hospital touch nothing at reset (held-open angle included) or after
    settling."""
    assert not fixture_contacts(rig), fixture_contacts(rig)[:5]
    rig.run(2.0)
    assert not fixture_contacts(rig), fixture_contacts(rig)[:5]


def test_held_doors_start_open_and_stay(rig):
    rig.run(3.0)
    held, shut = [], []
    for d in rig.spec["doors"]:
        for lf in d["leaves"]:
            a = rig.q(lf["joint"])
            target = lf["hold"] if lf["hold"] is not None else 0.0
            assert abs(a - target) < math.radians(3), (lf["joint"], math.degrees(a))
        (held if any(lf["hold"] is not None for lf in d["leaves"]) else shut).append(d["name"])
    st = rig.state()["doors"]
    assert all(st[n]["held_open"] and not st[n]["closed"] for n in held), {n: st[n] for n in held}
    assert all(st[n]["closed"] and not st[n]["held_open"] for n in shut), {n: st[n] for n in shut}
    print(f"\nheld open: {held}\nkept shut: {shut}")
    assert {"W_Compartment", "E_Compartment", "Main_LiftLobby"} <= set(held)


def test_alarm_closes_all_fire_doors(rig):
    rig.run(0.5)
    rig.fx.ctl.set_alarm(True)
    ok = rig.run(15.0, until=lambda r: all(v["closed"] for v in r.state()["doors"].values()))
    t = rig.fx.ctl.t - 0.5
    st = rig.state()
    print(f"\nalarm -> all {len(st['doors'])} fire doors closed after {t:.2f} s; lifts "
          f"{ {n: v['state'] for n, v in st['lifts'].items()} }")
    assert ok, {n: v for n, v in st["doors"].items() if not v["closed"]}
    assert sorted(st["doors"]) == sorted(d["name"] for d in rig.spec["doors"])
    rig.run(2.0)  # and they stay shut; every lift is on fire recall (open)
    st = rig.state()
    assert all(v["closed"] for v in st["doors"].values())
    assert all(v["state"] in ("opening", "open") for v in st["lifts"].values()), st["lifts"]


PUSH = [("W_Stair", 0), ("E_Stair", 0),                  # stair doors (never held open)
        ("Ward_DirtyUtility", 0), ("Central_Store", 0),   # hazard rooms (kept shut)
        ("D07_Exit", 0), ("D34_Exit", 1),                 # final exits
        ("Store", 0), ("FinalExit", 1)]                    # wing


@pytest.mark.parametrize("door,leaf", PUSH)
def test_push_opens_shut_leaf_and_it_self_closes(rig, door, leaf):
    """A kept-shut leaf opens under a 40 N push at the handle, only in its swing direction, and closes by itself."""
    lf = rig.leaf(door, leaf)
    bid = rig.body(lf["joint"])
    F = 40.0

    def push(r):
        a = r.q(lf["joint"])
        dx, dy = r.F._rot(lf["closed_dir"], a)
        n = np.array([dy, -dx, 0.0])          # leaf normal on the CW side (swing "-y" = negative angle here)
        if lf["range"][1] > 0:
            n = -n
        p = r.W(lf["hinge"][0] + dx * lf["lever"], lf["hinge"][1] + dy * lf["lever"], 1.0)
        f = F * n
        r.d.xfrc_applied[bid, :3] = f
        r.d.xfrc_applied[bid, 3:] = np.cross(p - r.d.xipos[bid], f)

    rig.run(2.5, push=push)
    a_open = rig.q(lf["joint"])
    print(f"\n{door} leaf{leaf} after 2.5 s of 40 N at the handle: {math.degrees(a_open):.1f} deg "
          f"(limit {rig.door(door)['swing_limit_deg']})")
    assert abs(a_open) > math.radians(30)
    assert lf["range"][0] - 1e-3 <= a_open <= lf["range"][1] + 1e-3
    rig.d.xfrc_applied[bid] = 0
    t0 = rig.fx.ctl.t
    ok = rig.run(15.0, until=lambda r: abs(r.q(lf["joint"])) < math.radians(2))
    print(f"self-closed: {ok} after {rig.fx.ctl.t - t0:.2f} s")
    assert ok
    rig.run(1.0)   # and it stays shut (the state rounds to 0.1 deg; "closed" is < 2 deg)
    assert rig.state()["doors"][door]["closed"], rig.state()["doors"][door]


# ---------------------------------------------------------------- lifts


def press_call(rig, name, seconds=0.3, force=6.0):
    c = next(c for c in rig.spec["calls"] if c["joint"] == "fx_" + name)
    bid = rig.body(c["joint"])
    rig.d.xfrc_applied[bid, :3] = force * np.array(c["axis"])
    rig.run(seconds)
    travel = rig.q(c["joint"])
    rig.d.xfrc_applied[bid] = 0
    rig.run(0.2)
    return c, travel


@pytest.mark.parametrize("call", ["LobbyCall", "MainCall"])
def test_call_button_force_opens_its_bank_then_dwell_and_close(rig, call):
    c, travel = press_call(rig, call)
    print(f"\n{call} travel under 6 N: {travel * 1000:.2f} mm; back out: {rig.q(c['joint']) * 1000:.2f} mm")
    assert travel > 0.0025 and rig.q(c["joint"]) < 0.0012
    states = {n: v["state"] for n, v in rig.state()["lifts"].items()}
    moving = [n for n, s in states.items() if s == "opening"]
    assert len(moving) == 1 and moving[0] in c["lifts"], states    # one car of this bank only
    name = moving[0]
    t0 = rig.fx.ctl.t
    assert rig.run(4.0, until=lambda r: r.state()["lifts"][name]["state"] == "open")
    t_open = rig.fx.ctl.t - t0
    gap = rig.state()["lifts"][name]["door_gap_m"]
    print(f"{name} ({c['bank']} bank) open after {t_open:.2f} s, gap {gap:.3f} m")
    assert gap > 1.38
    t1 = rig.fx.ctl.t
    assert rig.run(8.0, until=lambda r: r.state()["lifts"][name]["state"] == "closing")
    dwell = rig.fx.ctl.t - t1
    assert rig.run(5.0, until=lambda r: r.state()["lifts"][name]["state"] == "closed")
    assert 6.5 < dwell < 7.5
    # the round robin is per call button: the next press opens the other car of the same bank
    press_call(rig, call)
    nxt = [n for n, v in rig.state()["lifts"].items() if v["state"] == "opening"]
    assert nxt and nxt[0] in c["lifts"] and nxt[0] != name, nxt


@pytest.mark.parametrize("lift", ["MainLift1", "Lift1"])
def test_box_in_doorway_reopens_closing_lift(rig, lift):
    rig.fx.ctl.press(f"{lift}_Open")
    assert rig.run(4.0, until=lambda r: r.state()["lifts"][lift]["state"] == "open")
    assert rig.run(8.0, until=lambda r: r.state()["lifts"][lift]["state"] == "closing")
    rig.run(1.0)  # closing for 1 s
    gap = rig.state()["lifts"][lift]["door_gap_m"]
    spec = next(x for x in rig.spec["lifts"] if x["name"] == lift)
    rig.put_box(spec["front_x"] + 0.06, spec["center_y"])   # on the threshold, in the curtain plane
    assert rig.run(0.05, until=lambda r: r.state()["lifts"][lift]["state"] == "opening")
    st = rig.state()["lifts"][lift]
    print(f"\n{lift}: box in the doorway at gap {gap:.2f} m -> {st['state']}, reopens {st['reopens']}")
    assert st["reopens"] == 1
    rig.run(10.0)   # box stays: the lift stays open past the 7 s dwell
    assert rig.state()["lifts"][lift]["state"] == "open"
    rig.park_box()
    assert rig.run(4.0, until=lambda r: r.state()["lifts"][lift]["state"] == "closing")
    assert rig.run(5.0, until=lambda r: r.state()["lifts"][lift]["state"] == "closed")


# ---------------------------------------------------------------- lidar


def lidar_range(rig, origin, direction):
    group = np.zeros(6, np.uint8)
    group[0] = 1   # lidar.py: static world group only (walls, furniture boxes, fixtures)
    geomid = np.zeros(1, np.int32)
    dist = np.zeros(1)
    mujoco.mj_multiRay(rig.m, rig.d, origin, np.array(direction, float), group, 1, -1, geomid, dist, None, 1, 40.0)
    name = mujoco.mj_id2name(rig.m, mujoco.mjtObj.mjOBJ_GEOM, int(geomid[0])) if geomid[0] >= 0 else None
    return float(dist[0]), name


@pytest.mark.parametrize("door,origin,expect", [
    ("E_Compartment", (9.0, 6.9), 11.21 - 9.0),               # world frame: main building, C2|C3 screen at x 11.21
    ("CrossCorridor", (17.93 + 0.0, -4.45 - 7.6), 4.0),       # wing corridor, x = 4 doors
])
def test_lidar_sees_door_close(rig, door, origin, expect):
    o = np.array([origin[0], origin[1], 1.0])
    r_open, g_open = lidar_range(rig, o, [1, 0, 0])
    rig.fx.ctl.set_alarm(True)
    assert rig.run(12.0, until=lambda r: r.state()["doors"][door]["closed"])
    mujoco.mj_forward(rig.m, rig.d)
    r_closed, g_closed = lidar_range(rig, o, [1, 0, 0])
    t = rig.leaf(door, 0)["thickness"] / 2
    print(f"\nlidar ray east to {door}: held open {r_open:.2f} m ({g_open}) -> closed {r_closed:.2f} m ({g_closed})")
    assert r_open > expect + 1.0
    assert abs(r_closed - (expect - t)) < 0.05 and g_closed.startswith(f"fx_{door}_leaf")


# ---------------------------------------------------------------- static geometry: spawns and the 4 m link


def static_hits(rig, body_ids):
    """Contacts of the given bodies with anything that is not them, not the floor plane and not the probes."""
    floor = mujoco.mj_name2id(rig.m, mujoco.mjtObj.mjOBJ_GEOM, "world_floor")
    probes = {rig.body("test_box"), rig.body("bed_probe")}
    out = []
    for c in rig.d.contact[:rig.d.ncon]:
        b1, b2 = rig.m.geom_bodyid[c.geom1], rig.m.geom_bodyid[c.geom2]
        mine = (b1 in body_ids, b2 in body_ids)
        if mine[0] == mine[1] or floor in (c.geom1, c.geom2):
            continue
        other = c.geom2 if mine[0] else c.geom1
        if rig.m.geom_bodyid[other] in probes:
            continue
        out.append((mujoco.mj_id2name(rig.m, mujoco.mjtObj.mjOBJ_GEOM, other), round(float(c.dist), 4)))
    return out


def test_spawns_collision_free(rig):
    """The robot in its spawn pose (sim_node.respawn: pelvis 0.79 m, legs slightly bent, arms down) touches nothing
    but the floor at every hospital spawn of worlds/spawns.json."""
    spawns = json.load(open(os.path.join(SIM, "worlds", "spawns.json")))["hospital"]
    M = rig.M
    robot = set(range(rig.m.nbody))
    robot = {b for b in robot if b != 0 and _root(rig.m, b) == M.pelvis}
    spawn_q = np.array([-0.1, 0, 0, 0.3, -0.2, 0] * 2 + [0.0] * 17)
    bad = {}
    for name, sp in spawns.items():
        if name == "default":
            continue
        fq = M.free_q
        rig.d.qpos[fq:fq + 7] = [sp["x"], sp["y"], 0.79, math.cos(sp["yaw"] / 2), 0, 0, math.sin(sp["yaw"] / 2)]
        M.set_body_q(rig.d, spawn_q)
        mujoco.mj_forward(rig.m, rig.d)
        hits = static_hits(rig, robot)
        if hits:
            bad[name] = hits[:4]
        print(f"\nspawn {name} ({sp['x']:.2f}, {sp['y']:.2f}): {'clear' if not hits else hits[:4]}")
    assert not bad, bad


def _root(m, b):
    while m.body_parentid[b] != 0:
        b = m.body_parentid[b]
    return b


def test_bed_footprint_passes_the_4m_link(rig):
    """A 2.2 m x 1.0 m bed footprint (z 0.05..0.95) sweeps from the hospital SE lobby through the 2.73 m opening,
    the held-open compartment doorset (FD60S, ecw 2.29 m) and the 4 m link into the wing corridor, in 2 cm steps,
    without a contact (static world and fixtures). The x = 19.4 line is the route of layout.BUILDING_ROUTES."""
    rig.run(0.5)                       # held leaves settled on their magnets
    pb = {rig.body("bed_probe")}
    x = 19.4
    ys = np.arange(-1.6, -12.45 - 1e-9, -0.02)
    hits = []
    for y in ys:
        rig.put("bed_probe", np.array([x, y, BED[2] + 0.05]), yaw=math.pi / 2)   # long axis along the route (y)
        h = static_hits(rig, pb)
        if h:
            hits.append((round(float(y), 2), h[:3]))
    # the clear width at the hospital opening and at the doorset, measured with lidar rays across (z 1.0 m)
    widths = {}
    # world y: -4.47 is just inside the link (between its west kit wall face and the CloseEast junction box)
    for label, y in (("hospital opening", -4.47), ("compartment doorset", -4.45 - 0.6), ("link", -4.45 - 3.0)):
        rig.park_bed()
        o = np.array([x, y, 1.0])
        w = lidar_range(rig, o, [-1, 0, 0])[0] + lidar_range(rig, o, [1, 0, 0])[0]
        widths[label] = round(w, 3)
    print(f"\nbed sweep x {x}, y {ys[0]:.2f} .. {ys[-1]:.2f}: {len(hits)} blocked steps; clear widths {widths}")
    assert not hits, hits[:5]
    assert widths["hospital opening"] > 2.6 and widths["compartment doorset"] > 2.2 and widths["link"] > 3.7


def test_main_entrance_is_open(rig):
    """At tier2 and up the main entrance (two sliding glass units, x -5.64..3.16, deactivated in Isaac with their
    frames) is open: a 0.3 m box passes at five places across it, and lidar rays pass from the forecourt into the
    reception. The walk-in route of layout.BUILDING_ROUTES crosses the facade at x -1.24."""
    hits = {}
    for xc in (-4.6, -3.136, -1.24, 0.655, 2.2):
        for y in np.arange(-4.0, -1.0, 0.05):
            rig.put_box_world(xc, y)
            h = static_hits(rig, {rig.body("test_box")})
            if h:
                hits[(xc, round(float(y), 2))] = h[:3]
        r, g = lidar_range(rig, np.array([xc, -5.0, 1.0]), [0, 1, 0])
        print(f"\nentrance bay x {xc}: ray north from y -5.0 runs {r:.2f} m ({g})")
        assert r > 3.0
    assert not hits, list(hits.items())[:5]


# ---------------------------------------------------------------- cost


@pytest.mark.parametrize("tier", ["tier2", "full"])
def test_step_cost(tier):
    scene = os.path.join(SIM, f"hospital_g1_{tier}.xml")
    r = Rig(scene_spec(scene), scene)
    r.run(1.0)
    idle = np.mean(r.step_s[-400:]) * 1e6
    for n in ("Lift1", "Lift2", "MainLift1", "MainLift2"):
        r.fx.ctl.press(f"{n}_Open")
    r.run(4.0)
    r.step_s.clear()
    r.run(1.0)   # all four lifts open: curtains of both banks checked every step
    busy = np.mean(r.step_s) * 1e6
    t = time.perf_counter()
    for _ in range(500):
        mujoco.mj_step(r.m, r.d)
    mj = (time.perf_counter() - t) / 500 * 1e6
    print(f"\n{tier}: {len(r.fx.names)} fixture joints; fixture step {idle:.1f} us idle, {busy:.1f} us with 4 lifts "
          f"open; mj_step {mj:.0f} us")
    assert idle < 150 and busy < 250
