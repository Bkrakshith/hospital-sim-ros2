"""Teleop manager on its own (e.g. against the real robot or a sim started without manager:=true).

    ros2 launch g1_bringup teleop.launch.py [use_sim_time:=false]

The keyboard node needs a real terminal (raw key input), so it is NOT started by launch. Run it in its own
terminal:
    source env.sh && ros2 run g1_teleop_keyboard keyboard_node
Keys: r ready (DAMP->STAND) | l loco | w/s a/d q/e velocity | space zero | p pause | m manip | n manip off |
      g nav | x e-stop.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("keys_file", default_value="", description="override config/keys.yaml"),
        Node(package="g1_teleop_manager", executable="manager_node", name="g1_teleop_manager", output="screen",
             parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time"),
                          "keys_file": LaunchConfiguration("keys_file")}]),
        LogInfo(msg="teleop manager up. Keyboard: in another terminal run "
                    "'source env.sh && ros2 run g1_teleop_keyboard keyboard_node'"),
    ])
