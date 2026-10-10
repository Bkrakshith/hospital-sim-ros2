"""Scrub-nurse mission in the MuJoCo hospital: sim (theatre set, optional fixed base) + controller + mode manager
+ [Nav2 localisation on a saved map] + g1_mission mission_node [+ dashboard].

    ros2 launch g1_bringup mission.launch.py                       # fixed base at the instrument table (arm skills)
    ros2 launch g1_bringup mission.launch.py fixed_base:=false     # walk: spawn corridor_west, Nav2 + AMCL on hospital_gt
    args: fixed_base:=true|false  spawn:=<name|x,y,yaw> (default theatre_table / corridor_west)
          nav:=auto|true|false (auto = Nav2 unless fixed_base)  map:=hospital_gt (name in g1_navigation/maps or path; hospital_full = Isaac SLAM map)
          initial_pose:=ground_truth|""|"x,y,yaw" (AMCL seed; ground_truth = CHEAT)  dashboard:=false
          record_dir:="" (LeRobot recording, off)  record:="" (sim qpos .npz for videos)  viewer:=false
          autostart:=false  repeat:=true
          object_source:=ground_truth|perception  surgeon_source:=ground_truth|perception  policy:=<controller policy>

Then: press r (keyboard node or dashboard) -> STAND, and request an instrument:
    ros2 service call /g1/mission/request_instrument g1_interfaces/srv/RequestInstrument "{instrument: scalpel_handle, requester: cli}"
or use the dashboard's Mission page. p (STAND) / x (DAMP) override the mission at any time.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _truthy(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def _setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    here = os.path.join(get_package_share_directory("g1_bringup"), "launch")
    root = os.environ.get("HG1_ROOT", "")
    fixed = _truthy(lc("fixed_base"))
    spawn = lc("spawn") or ("theatre_table" if fixed else "corridor_west")
    nav = lc("nav")
    use_nav = (not fixed) if nav == "auto" else _truthy(nav)
    actions = [LogInfo(msg=f"mission: fixed_base={fixed} spawn={spawn} nav={use_nav}")]
    actions.append(IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(here, "sim.launch.py")),
        launch_arguments={"world": "hospital", "spawn": spawn, "fixed_base": str(fixed).lower(), "camera": "true",
                          "viewer": lc("viewer"), "policy": lc("policy"), "use_sim_time": "true",
                          "record": lc("record")}.items()))
    if use_nav:
        m = lc("map")
        if not os.path.isabs(m):
            m = os.path.join(root, "ros2_ws", "src", "hospital-sim-ros2", "g1_navigation", "maps", m if m.endswith(".yaml") else m + ".yaml")
        nav_launch = os.path.join(get_package_share_directory("g1_navigation"), "launch", "nav.launch.py")
        # give the sim a few seconds to publish /clock and TF before the costmaps come up
        actions.append(TimerAction(period=5.0, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(nav_launch),
            launch_arguments={"use_sim_time": "true", "map": m}.items())]))
    init = lc("initial_pose") if use_nav else ""
    actions.append(Node(
        package="g1_mission", executable="mission_node", name="g1_mission", output="screen",
        parameters=[{"use_sim_time": True, "use_nav": use_nav, "initial_pose": init,
                     "record_dir": lc("record_dir"), "autostart": _truthy(lc("autostart")),
                     "repeat": _truthy(lc("repeat")), "object_source": lc("object_source"),
                     "surgeon_source": lc("surgeon_source")}]))
    if _truthy(lc("dashboard")):
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(here, "dashboard.launch.py")),
            launch_arguments={"use_sim_time": "true"}.items()))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("fixed_base", default_value="true",
                              description="weld the pelvis at the instrument table (arm skills with the placeholder policy)"),
        DeclareLaunchArgument("spawn", default_value="", description="default: theatre_table (fixed) / corridor_west"),
        DeclareLaunchArgument("nav", default_value="auto", choices=["auto", "true", "false"]),
        DeclareLaunchArgument("map", default_value="hospital_gt"),
        DeclareLaunchArgument("initial_pose", default_value="ground_truth"),
        DeclareLaunchArgument("dashboard", default_value="false"),
        DeclareLaunchArgument("record_dir", default_value=""),
        DeclareLaunchArgument("viewer", default_value="false"),
        DeclareLaunchArgument("record", default_value="", description="sim qpos .npz for g1-mujoco-sim/render_replay.py"),
        DeclareLaunchArgument("autostart", default_value="false"),
        DeclareLaunchArgument("repeat", default_value="true"),
        DeclareLaunchArgument("object_source", default_value="ground_truth", choices=["ground_truth", "perception"]),
        DeclareLaunchArgument("surgeon_source", default_value="ground_truth", choices=["ground_truth", "perception"]),
        DeclareLaunchArgument("policy", default_value="unitree_g1_12dof_placeholder"),
        OpaqueFunction(function=_setup),
    ])
