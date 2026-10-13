"""MuJoCo sim + robot description + policy controller + teleop manager.

    ros2 launch g1_bringup sim.launch.py [backend:=mujoco|isaac] [world:=hospital|flat] [viewer:=true|false] [use_sim_time:=true]
                                         [spawn:=corridor_west|link|lobby|"x,y,yaw"] [policy:=<name|dir>]
                                         [harness:=true] [lidar:=true] [manager:=true]
                                         [effort_limits:=auto|model|ultra] [record:=/path/run.npz]
                                         [fixed_base:=false] [camera:=true] [fixtures_tier:=wing|tier1|tier2|full]

backend:=isaac starts everything except the MuJoCo sim node; run the Isaac sim yourself first:
    ./g1-isaac-sim/run.sh --headless [--flat] [--spawn link]     (see g1-isaac-sim/README.md)

Then drive it with the keyboard node in its own terminal (see teleop.launch.py):
    ros2 run g1_teleop_keyboard keyboard_node        # r (stand), l (loco), w/a/s/d/q/e, p, m, n, x
"""
import json
import os

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _truthy(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def _setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    use_sim_time = _truthy(lc("use_sim_time"))
    world = lc("world")
    # master repo root: $HG1_ROOT (env.sh), else from this file (ros2_ws/src/hospital-sim-ros2/g1_bringup/launch/)
    root = os.environ.get("HG1_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                                    "..", "..", "..", "..", "..")))
    actions = []
    policy = lc("policy")
    sidecar = {}
    try:
        from g1_controller.controller_node import resolve_policy
        with open(os.path.join(resolve_policy(policy), "policy.json")) as f:
            sidecar = json.load(f)
    except Exception as e:  # noqa: BLE001
        actions.append(LogInfo(msg=f"could not read policy sidecar ({e})"))
    effort = lc("effort_limits")
    if effort == "auto":
        # Placeholder (unitree_rl_gym G1 policy): trained with 50 Nm ankle/waist limits, marginal on the model's
        # 35 Nm under realistic 2-4 ms loop latency -> ultra. Trained g1-whole-body-rl/exports policies: follow the limits they
        # were trained with, sidecar robot.effort_source ("ultra" = 50 Nm, "canonical" = robot model, 35 Nm).
        if sidecar.get("placeholder"):
            effort = "ultra"
        else:
            effort = "ultra" if sidecar.get("robot", {}).get("effort_source") == "ultra" else "model"
        actions.append(LogInfo(msg=f"effort_limits auto -> {effort}"))
    if lc("backend") == "isaac":
        actions.append(LogInfo(msg="backend isaac: not starting g1_mujoco_sim; start ./g1-isaac-sim/run.sh "
                                   "(its effort limits and spawn are set there)"))
    else:
        actions.append(Node(
            package="g1_mujoco_sim", executable="sim_node", name="g1_mujoco_sim", output="screen",
            parameters=[{"world": world, "viewer": _truthy(lc("viewer")), "spawn": lc("spawn"),
                         "harness": _truthy(lc("harness")), "lidar": _truthy(lc("lidar")),
                         "model": os.path.join(root, "g1-mujoco-sim", f"{world}_g1.xml"),
                         "record_path": lc("record"), "effort_limits": effort,
                         "fixed_base": _truthy(lc("fixed_base")), "camera": _truthy(lc("camera")),
                         "fixtures_tier": lc("fixtures_tier")}],
            # offscreen head-camera rendering: EGL when there is no viewer window (GLFW needs one)
            additional_env={} if _truthy(lc("viewer")) else {"MUJOCO_GL": "egl"}))
    try:
        desc = get_package_share_directory("g1_description")
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(desc, "launch", "description.launch.py")),
            launch_arguments={"use_sim_time": str(use_sim_time).lower(), "variant": "full"}.items()))
    except PackageNotFoundError:
        actions.append(LogInfo(msg="g1_description not built: no robot_state_publisher"))
    actions.append(Node(
        package="g1_controller", executable="controller_node", name="g1_controller", output="screen",
        parameters=[{"policy": policy, "use_sim_time": use_sim_time,
                     "compute_crc": _truthy(lc("compute_crc"))}]))
    if _truthy(lc("manager")):
        # manager blends arms back to the policy's default upper-body pose when leaving MANIP
        upper = [float(v) for v in sidecar.get("default_qpos", [0.0] * 29)[12:29]]
        actions.append(Node(
            package="g1_teleop_manager", executable="manager_node", name="g1_teleop_manager", output="screen",
            parameters=[{"use_sim_time": use_sim_time, "upper_body_default": upper}]))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("backend", default_value="mujoco", choices=["mujoco", "isaac"],
                              description="simulator: mujoco (started here) or isaac (start g1-isaac-sim/run.sh separately)"),
        DeclareLaunchArgument("world", default_value="hospital", choices=["hospital", "flat"]),
        DeclareLaunchArgument("viewer", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("spawn", default_value="", description="spawn name (g1-mujoco-sim/worlds/spawns.json) or 'x,y,yaw'"),
        DeclareLaunchArgument("policy", default_value="unitree_g1_12dof_placeholder",
                              description="policy name (g1-whole-body-rl/exports/<name> or g1_controller/policies/<name>) or dir"),
        DeclareLaunchArgument("harness", default_value="true"),
        DeclareLaunchArgument("lidar", default_value="true"),
        DeclareLaunchArgument("manager", default_value="true"),
        DeclareLaunchArgument("effort_limits", default_value="auto", choices=["auto", "model", "ultra"],
                              description="sim motor torque limits: model (robot MJCF), ultra (robot_spec "
                                          "effort_limit_ultra, 50 Nm ankles), auto (placeholder -> ultra; else sidecar "
                                          "robot.effort_source: ultra -> ultra, canonical -> model)"),
        DeclareLaunchArgument("record", default_value="", description="save sim qpos to this .npz (for render_replay.py)"),
        DeclareLaunchArgument("fixed_base", default_value="false",
                              description="weld the pelvis at the spawn pose (arm-skill development; no walking)"),
        DeclareLaunchArgument("camera", default_value="true",
                              description="head RGB-D camera /g1/head_camera/* (rendered only while subscribed)"),
        DeclareLaunchArgument("fixtures_tier", default_value="",
                              description="building fixture tier of the MuJoCo world: wing, tier1, tier2 or full "
                                          "(empty: $HG1_FIXTURES, else tier2). The sim node loads "
                                          "g1-mujoco-sim/<world>_g1_<tier>.xml. Use the same tier as Isaac."),
        DeclareLaunchArgument("compute_crc", default_value="false",
                              description="LowCmd CRC (the sim ignores it; set true for the real robot)"),
        OpaqueFunction(function=_setup),
    ])
