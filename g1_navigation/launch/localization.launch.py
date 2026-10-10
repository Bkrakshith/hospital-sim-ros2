"""map_server + AMCL (omni motion model) on a saved map.

    ros2 launch g1_navigation localization.launch.py map:=/abs/path/map.yaml [use_sim_time:=true]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    share = get_package_share_directory('g1_navigation')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    map_yaml = LaunchConfiguration('map')

    params = RewrittenYaml(
        source_file=params_file,
        param_rewrites={'use_sim_time': use_sim_time, 'yaml_filename': map_yaml},
        convert_types=True)

    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '0'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map', description='Full path to the map yaml'),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(share, 'config', 'nav2_params.yaml')),
        Node(package='nav2_map_server', executable='map_server', name='map_server',
             output='screen', parameters=[params]),
        Node(package='nav2_amcl', executable='amcl', name='amcl',
             output='screen', parameters=[params]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_localization', output='screen',
             parameters=[{'use_sim_time': use_sim_time, 'autostart': True,
                          'node_names': ['map_server', 'amcl']}]),
    ])
