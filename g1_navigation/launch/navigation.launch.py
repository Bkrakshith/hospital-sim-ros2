"""Nav2 servers for the G1 (controller, planner, smoother, behaviours, BT navigator, waypoint
follower, velocity smoother). The final velocity command is published on /g1/cmd_vel_nav, which
g1_teleop_manager forwards to the controller only while the mode is NAV.

Needs map -> odom from either slam_toolbox (slam.launch.py) or AMCL (localization.launch.py).

    ros2 launch g1_navigation navigation.launch.py [use_sim_time:=true]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

NAV_CMD_VEL = '/g1/cmd_vel_nav'


def generate_launch_description():
    share = get_package_share_directory('g1_navigation')
    use_sim_time = LaunchConfiguration('use_sim_time')
    params_file = LaunchConfiguration('params_file')
    params = RewrittenYaml(source_file=params_file,
                           param_rewrites={'use_sim_time': use_sim_time,
                                           'map_topic': LaunchConfiguration('map_topic'),
                                           'yaml_filename': LaunchConfiguration('keepout_mask')},
                           convert_types=True)

    def server(pkg, exe, name=None, remaps=()):
        return Node(package=pkg, executable=exe, name=name or exe, output='screen',
                    parameters=[params], remappings=list(remaps))

    lifecycle_nodes = ['controller_server', 'smoother_server', 'planner_server',
                       'behavior_server', 'bt_navigator', 'waypoint_follower',
                       'velocity_smoother']
    return LaunchDescription([
        SetEnvironmentVariable('RCUTILS_LOGGING_BUFFERED_STREAM', '0'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('params_file',
                              default_value=os.path.join(share, 'config', 'nav2_params.yaml')),
        DeclareLaunchArgument('keepout_mask', default_value=os.path.join(share, 'maps', 'stairs_keepout.yaml'),
                              description='Nav2 KeepoutFilter mask (stairs + 1.25 m); the costmaps use it via '
                                          '/costmap_filter_info'),
        Node(package='nav2_map_server', executable='map_server', name='filter_mask_server', output='screen',
             parameters=[params]),
        Node(package='nav2_map_server', executable='costmap_filter_info_server', name='costmap_filter_info_server',
             output='screen', parameters=[params]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_filters',
             output='screen', parameters=[{'use_sim_time': use_sim_time, 'autostart': True,
                                           'node_names': ['filter_mask_server', 'costmap_filter_info_server']}]),
        DeclareLaunchArgument('map_topic', default_value='/map',
                              description='global costmap static layer map topic (/map_padded: fixed-size copy of '
                                          'a growing SLAM map, scripts/map_pad_relay.py)'),
        # controller -> cmd_vel_nav_raw -> velocity_smoother -> /g1/cmd_vel_nav
        server('nav2_controller', 'controller_server', remaps=[('cmd_vel', 'cmd_vel_nav_raw')]),
        server('nav2_smoother', 'smoother_server'),
        server('nav2_planner', 'planner_server'),
        # recovery behaviours (spin / back up) drive the robot directly, also through the arbiter
        server('nav2_behaviors', 'behavior_server', remaps=[('cmd_vel', NAV_CMD_VEL)]),
        server('nav2_bt_navigator', 'bt_navigator'),
        server('nav2_waypoint_follower', 'waypoint_follower'),
        server('nav2_velocity_smoother', 'velocity_smoother',
               remaps=[('cmd_vel', 'cmd_vel_nav_raw'), ('cmd_vel_smoothed', NAV_CMD_VEL)]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[{'use_sim_time': use_sim_time, 'autostart': True,
                          'node_names': lifecycle_nodes}]),
    ])
