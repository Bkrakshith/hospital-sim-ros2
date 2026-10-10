"""G1 web dashboard (hospital-sim-ui/server: rclpy + aiohttp, one process) on http://localhost:8080.

    ros2 launch g1_bringup dashboard.launch.py [port:=8080] [use_sim_time:=true] [mock:=false]
mock:=true also starts hospital-sim-ui/tools/mock_robot.py (fake robot + emulated mode manager, no sim needed).
SLAM / Nav2 can then be started and stopped from the browser.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _setup(context):
    lc = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    # master repo root: $HG1_ROOT (env.sh), else from this file (ros2_ws/src/hospital-sim-ros2/g1_bringup/launch/)
    root = os.environ.get('HG1_ROOT') or os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', '..', '..'))
    py = os.path.join(root, '.venv', 'bin', 'python')
    if not os.path.exists(py):
        py = 'python3'
    cmd = [py, os.path.join(root, 'hospital-sim-ui', 'server'), '--port', lc('port'),
           '--use-sim-time', 'true' if lc('use_sim_time').lower() in ('1', 'true', 'yes') else 'false',
           '--exit-with-parent']
    if lc('mock').lower() in ('1', 'true', 'yes'):
        cmd.append('--mock')
    return [ExecuteProcess(cmd=cmd, name='g1_dashboard', output='screen', sigterm_timeout='15')]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='8080'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('mock', default_value='false'),
        OpaqueFunction(function=_setup),
    ])
