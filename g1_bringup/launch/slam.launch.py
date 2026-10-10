"""SLAM for the G1 (slam_toolbox online async on /scan). Wrapper around g1_navigation/slam.launch.py.

    ros2 launch g1_bringup slam.launch.py [use_sim_time:=true]
Save the map from the dashboard (Mapping -> Save map) or with
    ros2 run nav2_map_server map_saver_cli -f $HG1_ROOT/ros2_ws/src/hospital-sim-ros2/g1_navigation/maps/<name>
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    nav_launch = os.path.join(get_package_share_directory('g1_navigation'), 'launch')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav_launch, 'slam.launch.py')),
            launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()),
    ])
