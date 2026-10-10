"""Perception for the scrub-nurse mission.

    ros2 launch g1_perception perception.launch.py                         # gt / gt (sim, flagged ground truth)
    ros2 launch g1_perception perception.launch.py detector:=onnx hand:=mediapipe model:=/path/model.onnx
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory("g1_perception"), "config", "perception.yaml")
    args = [DeclareLaunchArgument("detector", default_value="gt"), DeclareLaunchArgument("hand", default_value="gt"),
            DeclareLaunchArgument("model", default_value=os.path.join(os.environ.get("HG1_ROOT", ""),
                                                                      "hospital-sim-perception/models/instrument_obb.onnx")),
            DeclareLaunchArgument("use_sim_time", default_value="true")]
    st = {"use_sim_time": LaunchConfiguration("use_sim_time")}
    return LaunchDescription(args + [
        Node(package="g1_perception", executable="detector_node", name="g1_perception_detector", output="screen",
             parameters=[cfg, st, {"mode": LaunchConfiguration("detector"), "model": LaunchConfiguration("model")}]),
        Node(package="g1_perception", executable="surgeon_hand_node", name="g1_perception_surgeon_hand", output="screen",
             parameters=[cfg, st, {"mode": LaunchConfiguration("hand")}]),
    ])
