"""slam_toolbox online-async mapping for the G1 (publishes /map and map -> odom).

    ros2 launch g1_navigation slam.launch.py [use_sim_time:=true] [slam_params_file:=...]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('g1_navigation')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params = LaunchConfiguration('slam_params_file')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'slam_params_file',
            default_value=os.path.join(share, 'config', 'slam_toolbox_online_async.yaml')),
        DeclareLaunchArgument(
            'base_frame', default_value='pelvis',
            description='robot base frame for slam_toolbox; base_footprint (planar, Isaac --base-footprint) keeps '
                        'map->odom flat when the pelvis tilts while walking'),
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time,
                                 'base_frame': LaunchConfiguration('base_frame')}],
        ),
    ])
