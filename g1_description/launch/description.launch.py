"""robot_state_publisher for the G1 29-DoF rev 1.0 + BrainCo Revo2 model.

Args:
  use_sim_time (default false)   set true whenever the MuJoCo/Isaac sim publishes /clock
  variant      (default full)    full -> g1_29dof_brainco_ros.urdf (29 body + 12 actuated hand joints;
                                 the 10 coupled finger joints are URDF <mimic> joints, computed by RSP)
                                 ik   -> g1_29dof_brainco_ik.urdf (29 body joints, rigid hands)
Subscribes /joint_states, publishes /tf, /tf_static, /robot_description. Root link: pelvis.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

URDF = {"full": "g1_29dof_brainco_ros.urdf", "ik": "g1_29dof_brainco_ik.urdf"}


def _setup(context):
    share = get_package_share_directory("g1_description")
    variant = LaunchConfiguration("variant").perform(context)
    path = os.path.join(share, "urdf", URDF[variant])
    with open(path) as f:
        urdf = f.read()
    if variant == "ik":  # the ik variant keeps relative mesh paths; make them resolvable for RViz
        urdf = urdf.replace("../meshes/", "package://g1_description/meshes/")
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context).lower() in ("1", "true", "yes")
    return [Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": urdf, "use_sim_time": use_sim_time, "publish_frequency": 50.0}],
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("variant", default_value="full", choices=list(URDF)),
        OpaqueFunction(function=_setup),
    ])
