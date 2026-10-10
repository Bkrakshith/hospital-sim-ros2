"""Everything for the simulated hospital: sim.launch.py (sim + description + controller + teleop manager)
plus slam / nav / dashboard launch files from this package's launch/ directory when they exist.

    ros2 launch g1_bringup all.launch.py [backend:=mujoco|isaac] [world:=hospital] [viewer:=false] [dashboard:=true]
                                         [slam:=false] [nav:=false] [mission:=false]

SLAM and Nav2 are started/stopped from the dashboard (Mapping / Navigation pages), so they are off here by
default; set slam:=true / nav:=true only when running without the dashboard (not both with it: duplicates).
mission:=true adds the scrub-nurse mission node (g1_mission; Mission page in the dashboard). It navigates with Nav2,
so start Nav2 with a saved map (default hospital_full) (dashboard Navigation page or nav:=true) or use mission.launch.py, which
brings up everything including Nav2 and the fixed-base option.

Extra launch files are included only if present (they are provided by the navigation/UI part of the stack):
  slam.launch.py, nav.launch.py, dashboard.launch.py   (all receive use_sim_time)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _setup(context):
    here = os.path.join(get_package_share_directory("g1_bringup"), "launch")
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    sim_args = {k: lc(k) for k in ("backend", "world", "viewer", "use_sim_time", "spawn", "policy", "effort_limits")}
    actions = [IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(here, "sim.launch.py")),
                                        launch_arguments=sim_args.items())]
    if lc("mission").lower() in ("1", "true", "yes"):
        from launch_ros.actions import Node
        actions.append(Node(package="g1_mission", executable="mission_node", name="g1_mission", output="screen",
                            parameters=[{"use_sim_time": lc("use_sim_time").lower() in ("1", "true", "yes"),
                                         "use_nav": True}]))
    for name in ("slam", "nav", "dashboard"):
        path = os.path.join(here, f"{name}.launch.py")
        if lc(name).lower() not in ("1", "true", "yes"):
            continue
        if os.path.exists(path):
            largs = {"use_sim_time": lc("use_sim_time")}
            if name == "nav":  # nav alone: AMCL on the default saved map; with slam:=true: Nav2 on the live SLAM map
                largs["slam"] = lc("slam")
            actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                                    launch_arguments=largs.items()))
        else:
            actions.append(LogInfo(msg=f"{name}.launch.py not installed in g1_bringup: skipped"))
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("backend", default_value="mujoco", choices=["mujoco", "isaac"]),
        DeclareLaunchArgument("world", default_value="hospital"),
        DeclareLaunchArgument("viewer", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("spawn", default_value=""),
        DeclareLaunchArgument("policy", default_value="unitree_g1_12dof_placeholder"),
        DeclareLaunchArgument("effort_limits", default_value="auto"),
        DeclareLaunchArgument("slam", default_value="false"),
        DeclareLaunchArgument("nav", default_value="false"),
        DeclareLaunchArgument("dashboard", default_value="true"),
        DeclareLaunchArgument("mission", default_value="false", description="scrub-nurse mission node (g1_mission)"),
        OpaqueFunction(function=_setup),
    ])
