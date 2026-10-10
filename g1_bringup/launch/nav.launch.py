"""Nav2 for the G1. Wrapper around g1_navigation/nav.launch.py.

    ros2 launch g1_bringup nav.launch.py                       # map_server + AMCL on the default map hospital_full
    ros2 launch g1_bringup nav.launch.py map:=<abs path or name in g1_navigation/maps>   # another saved map
    ros2 launch g1_bringup nav.launch.py slam:=true            # on a running SLAM (map:= ignored)
hospital_full is the lidar SLAM map of the whole Isaac hospital ground floor + surgical wing (world frame; see
g1_navigation/README.md "Maps"). Until 2026-10-09 an empty map meant "on SLAM"; use slam:=true for that now.
Nav2's velocity output is /g1/cmd_vel_nav; the mode manager forwards it only in NAV (key g from STAND).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _setup(context):
    nav_launch = os.path.join(get_package_share_directory('g1_navigation'), 'launch')
    root = os.environ.get('HG1_ROOT', '')
    m = LaunchConfiguration('map').perform(context).strip()
    if LaunchConfiguration('slam').perform(context).lower() in ('1', 'true', 'yes'):
        m = ''
    if m and not os.path.isabs(m):
        name = m if m.endswith('.yaml') else m + '.yaml'
        m = os.path.join(root, 'ros2_ws', 'src', 'hospital-sim-ros2', 'g1_navigation', 'maps', name)
    args = {'use_sim_time': LaunchConfiguration('use_sim_time').perform(context)}
    if m:
        args['map'] = m
    else:
        args['slam'] = 'true'
    return [IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(nav_launch, 'nav.launch.py')),
                                     launch_arguments=args.items())]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('map', default_value='hospital_full',
                              description='saved map (name in g1_navigation/maps or yaml path); empty = running SLAM'),
        DeclareLaunchArgument('slam', default_value='false',
                              description='true: Nav2 on a running slam_toolbox (no map_server/AMCL)'),
        OpaqueFunction(function=_setup),
    ])
