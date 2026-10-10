"""Offline harness: the same skills on an in-process MuJoCo model, without ROS. For fast skill development, unit
tests and batch trials. It emulates the parts of the stack the skills talk to:

  * g1_controller in fixed-base mode: PD on all 29 motors with the placeholder sidecar gains, legs held at the
    default pose, upper body = last /g1/auto/upper_body_target, rate-limited at 1.5 rad/s (controller default)
  * g1_mujoco_sim: pelvis weld (fixed_base), Revo2 hand position actuators, SceneObjects (ground truth + simulated
    surgeon grip + object reset)
It is NOT a substitute for the ROS stack test (message timing, the real controller/manager); scripts/run_trials.py
runs the same trials through ROS.
"""
import json
import math
import os
import sys

import numpy as np

from g1_mission.geometry import make_T, quat_to_mat
from g1_mission.interfaces import GroundTruthObjects, GroundTruthStation, GroundTruthSurgeonHand, RobotInterface
from g1_mission.kinematics import UpperBodyIK, repo_root


def _import_sim():
    try:
        from g1_mujoco_sim import scene_objects, model_map
    except ImportError:
        sys.path.insert(0, os.path.join(repo_root(), "ros2_ws", "src", "hospital-sim-ros2", "g1_mujoco_sim"))
        from g1_mujoco_sim import scene_objects, model_map
    return scene_objects, model_map


class OfflineRobot(RobotInterface):
    def __init__(self, scene=None, station=None, fixed_base=True, sidecar=None, render=None):
        import mujoco
        self.mj = mujoco
        so, mm = _import_sim()
        root = repo_root()
        scene = scene or os.path.join(root, "g1-mujoco-sim", "hospital_g1.xml")
        self.m = mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        self.M = mm.ModelMap(self.m)
        setup = os.path.join(os.path.dirname(scene), "worlds", "theatre_setup.json")
        self.setup = json.load(open(setup))
        self.objects = so.SceneObjects(self.m, setup)
        sc = sidecar or os.path.join(root, "ros2_ws", "src", "hospital-sim-ros2", "g1_controller", "policies",
                                     "unitree_g1_12dof_placeholder", "policy.json")
        s = json.load(open(sc))
        self.kp, self.kd = np.array(s["kp"], float), np.array(s["kd"], float)
        self.default = np.array(s["default_qpos"], float)
        # default spawn = the sim's fixed-base spawn "theatre_table" (0.12 m behind the stance: the policy-default
        # arms would otherwise start inside the rack); the mission's FineAlign glides the base onto the stance
        spawns = json.load(open(os.path.join(os.path.dirname(scene), "worlds", "spawns.json")))["hospital"]
        st = station or spawns.get("theatre_table", self.setup["station"])
        self.station = st
        fq = self.M.free_q
        yaw = st["yaw"]
        self.d.qpos[fq:fq + 3] = [st["x"], st["y"], 0.78]
        self.d.qpos[fq + 3:fq + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.M.set_body_q(self.d, self.default)
        mujoco.mj_forward(self.m, self.d)
        self.fixed_eq = -1
        if fixed_base:
            self.fixed_eq = so.weld_to_world(self.m, self.d, "fixed_base", "pelvis")
        self.glide = so.BaseGlide()
        self.upper_cmd = self.default[12:].copy()
        self.upper_tgt = self.default[12:].copy()
        self.hand_cmd = {"left": np.zeros(6), "right": np.zeros(6)}
        self.cmd_vel = (0.0, 0.0, 0.0)
        self.rate_limit = 1.5
        self.renderer = None
        self.frames = []
        self.render_cfg = render           # dict(camera=..., size=(w,h), fps=..)
        self._next_frame = 0.0
        self.head = None

    # ------------------------------------------------------------------ RobotInterface
    def now(self):
        return float(self.d.time)

    def upper_q(self):
        return self.d.qpos[self.M.qadr[12:]].copy()

    def hand_q(self, side):
        return np.clip(self.d.qpos[self.M.hand_qadr[side]] / self.M.hand_upper[side], 0, 1)

    def base_T(self):
        fq = self.M.free_q
        return make_T(quat_to_mat(self.d.qpos[fq + 3:fq + 7]), self.d.qpos[fq:fq + 3])

    def command(self, upper=None, left_hand=None, right_hand=None, cmd_vel=None):
        if upper is not None:
            self.upper_tgt = np.array(upper, float)
        if left_hand is not None:
            self.hand_cmd["left"] = np.clip(np.array(left_hand, float), 0, 1)
        if right_hand is not None:
            self.hand_cmd["right"] = np.clip(np.array(right_hand, float), 0, 1)
        if cmd_vel is not None:
            self.cmd_vel = tuple(cmd_vel)

    # ------------------------------------------------------------------ physics
    def advance(self, dt):
        mj, m, d, M = self.mj, self.m, self.d, self.M
        n = max(1, int(round(dt / m.opt.timestep)))
        for _ in range(n):
            step = self.rate_limit * m.opt.timestep
            self.upper_cmd += np.clip(self.upper_tgt - self.upper_cmd, -step, step)
            q_des = np.r_[self.default[:12], self.upper_cmd]
            q, dq = d.qpos[M.qadr], d.qvel[M.dadr]
            d.ctrl[M.act] = np.clip(self.kp * (q_des - q) - self.kd * dq, -M.tau_limit, M.tau_limit)
            for s in ("left", "right"):
                d.ctrl[M.hand_act[s]] = self.hand_cmd[s] * M.hand_upper[s]
            if self.glide.active:
                self.glide.step(m, self.fixed_eq, d.time)
            mj.mj_step(m, d)
            self.objects.update_grip(d, m.opt.timestep)
        if self.render_cfg and d.time >= self._next_frame:
            self._render()
            self._next_frame = d.time + 1.0 / self.render_cfg.get("fps", 25)

    def _render(self):
        mj = self.mj
        w, h = self.render_cfg.get("size", (640, 480))
        if self.renderer is None:
            self.m.vis.global_.offwidth = max(self.m.vis.global_.offwidth, w)
            self.m.vis.global_.offheight = max(self.m.vis.global_.offheight, h)
            self.renderer = mj.Renderer(self.m, h, w)
        self.renderer.update_scene(self.d, camera=self.render_cfg.get("camera", "theatre_station"))
        self.frames.append(self.renderer.render().copy())

    def reset_objects(self, offset=(0.0, 0.0, 0.0)):
        self.objects.reset(self.d, offset)

    def glide_to(self, x, y, yaw):
        """Fixed-base glide (same as the sim node's /sim/fixed_base_goal)."""
        q0 = self.m.eq_data[self.fixed_eq, 6:10]
        yaw0 = -math.atan2(2 * (q0[0] * q0[3] + q0[1] * q0[2]), 1 - 2 * (q0[2] ** 2 + q0[3] ** 2))
        self.glide.start(self.d.time, self.m.eq_data[self.fixed_eq, 0:3].copy(), yaw0, (x, y), yaw)

    # ------------------------------------------------------------------ sources for the skill context
    def sources(self):
        gt = GroundTruthObjects()
        outer = self

        class _Live(GroundTruthObjects):
            def get(self, name):
                self.update(outer.objects.snapshot(outer.d), stamp=outer.now())
                return GroundTruthObjects.get(self, name)

            def names(self):
                self.update(outer.objects.snapshot(outer.d), stamp=outer.now())
                return GroundTruthObjects.names(self)
        gt = _Live()
        return gt, GroundTruthSurgeonHand(gt), GroundTruthStation(self.base_T)


class OfflineNav:
    """NavClient interface on the offline fixed-base glide."""

    def __init__(self, robot):
        self.robot, self.state, self.msg, self.goal = robot, "idle", "", None

    def available(self):
        return True

    def exact(self):
        return True

    def go(self, x, y, yaw):
        self.goal = (x, y, yaw)
        self.robot.glide_to(x, y, yaw)
        self.state = "active"

    def cancel(self):
        self.robot.glide.seg = None
        self.state = "canceled"

    def distance_remaining(self):
        T = self.robot.base_T()
        d = math.hypot(self.goal[0] - T[0, 3], self.goal[1] - T[1, 3]) if self.goal else None
        if self.state == "active" and not self.robot.glide.active:
            self.state = "succeeded" if d is not None and d < 0.03 else "failed"
        return d


def make_context(robot, catalog=None, locations=None, params=None):
    from g1_mission.skills.base import ArmMotion, SkillContext
    from g1_mission.config import load_catalog, load_locations, load_mission, resolve
    objects, surgeon, station = robot.sources()
    ik = UpperBodyIK(side="right")
    cfg = resolve(params or load_mission())
    ctx = SkillContext(robot=robot, objects=objects, surgeon=surgeon, station=station, ik=ik,
                       catalog=catalog or load_catalog(), locations=locations or load_locations(),
                       params=cfg.get("skills", {}), ready_pose=np.array(cfg.get("ready_pose", [0.0] * 17), float))
    ctx.arm = ArmMotion(ik)
    ctx.nav = OfflineNav(robot) if robot.fixed_eq >= 0 else None
    robot.mission_cfg = cfg
    return ctx


def run_skill(ctx, skill, rate=50.0, timeout=60.0, **goal):
    """Tick a skill to completion on the offline robot. Returns (Status, message, duration)."""
    from g1_mission.skills.base import Status
    robot = ctx.robot
    skill.start(ctx, **goal)
    t0 = robot.now()
    st = Status.RUNNING
    while st == Status.RUNNING:
        st = skill.tick(ctx)
        robot.advance(1.0 / rate)
        if ctx.recorder is not None:
            ctx.recorder.frame(robot, image=robot.head)
        if robot.now() - t0 > timeout:
            skill.stop(ctx)
            return Status.FAILURE, f"{skill.name}: timeout", robot.now() - t0
    return st, skill.message, robot.now() - t0
