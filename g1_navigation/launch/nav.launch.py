"""Full navigation bringup for the G1.

  Localisation on a saved map (map_server + AMCL) + Nav2:
    ros2 launch g1_navigation nav.launch.py      # default map maps/hospital_full.yaml (world frame)
    ros2 launch g1_navigation nav.launch.py map:=$HG1_ROOT/ros2_ws/src/hospital-sim-ros2/g1_navigation/maps/<other>.yaml
  Nav2 on top of a running slam_toolbox (navigate while mapping):
    ros2 launch g1_navigation nav.launch.py slam:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_dir = os.path.join(get_package_share_directory('g1_navigation'), 'launch')
    share = get_package_share_directory('g1_navigation')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map', default_value=os.path.join(share, 'maps', 'hospital_full.yaml'),
                              description='Map yaml for map_server/AMCL (ignored when slam:=true); default: the '
                                          'lidar SLAM map of the whole hospital (world frame)'),
        DeclareLaunchArgument('slam', default_value='false',
                              description='true: map -> odom comes from a running slam_toolbox'),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(share, 'config', 'nav2_params.yaml')),
        DeclareLaunchArgument('map_topic', default_value='/map',
                              description='global costmap map topic; with slam:=true a growing map can be relayed '
                                          'on a fixed canvas (map_pad_relay.py -> /map_padded)'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'localization.launch.py')),
            condition=UnlessCondition(LaunchConfiguration('slam')),
            launch_arguments={'use_sim_time': use_sim_time, 'map': LaunchConfiguration('map'),
                              'params_file': params_file}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'navigation.launch.py')),
            launch_arguments={'use_sim_time': use_sim_time, 'params_file': params_file,
                              'map_topic': LaunchConfiguration('map_topic')}.items()),
    ])
