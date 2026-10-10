# g1_navigation: slam_toolbox and Nav2 for the G1

This package holds the slam_toolbox and Nav2 (Humble) configuration and launch files for the Unitree G1 humanoid in
the Hospital_G1 wing. Frames and topics follow `docs/interfaces.md`. The base frame is `pelvis`, and the simulator or
the odometry node publishes `odom -> pelvis`. The lidar topics are `/scan` (2-D slice of the head MID-360) and
`/livox/lidar` (3-D cloud). Nav2 writes to `/g1/cmd_vel_nav`, and the mode manager forwards it only in NAV and AUTO.

This package has no nodes of its own. It is configuration (`config/`), maps (`maps/`), launch files and survey
scripts. Build and install are in the repository README (`../README.md`).

## How it works

The simulator (or the drivers of the robot) publishes `/scan`, `/livox/lidar`, `/odom` and the TF `odom -> pelvis`.
The node `robot_state_publisher` (g1_description) adds the robot links from `/joint_states`. The launch file
`slam.launch.py` runs `async_slam_toolbox_node` with `config/slam_toolbox_online_async.yaml`. That node consumes
`/scan` and publishes `/map` every 2 s and the `map -> odom` transform. It adds a scan to its pose graph every 0.25 m
or 0.25 rad of travel. The launch file `localization.launch.py` is the alternative on a saved map: `map_server` serves
`maps/<name>.yaml`, and `amcl` (omni motion model) publishes `map -> odom`.

The launch file `navigation.launch.py` runs the Nav2 servers from `config/nav2_params.yaml`. The helper
`nav2_common.RewrittenYaml` injects `use_sim_time`, the map topic and the keep-out mask. The servers are:

- the controller server with MPPI (`motion_model: Omni`, 20 Hz, 1000 x 40-step samples)
- the Smac 2D planner
- the smoother
- the behaviour server
- the BT navigator
- the waypoint follower
- the velocity smoother
- a second `map_server` and a `costmap_filter_info_server` for the `maps/stairs_keepout.yaml` KeepoutFilter

The `cmd_vel` of the controller server is remapped to `cmd_vel_nav_raw`, and the velocity smoother turns it into
`/g1/cmd_vel_nav`. The behaviour server writes `/g1/cmd_vel_nav` directly. The mode manager `g1_teleop_manager`
forwards `/g1/cmd_vel_nav` to `/g1/cmd_vel` only in NAV or AUTO, so Nav2 never drives without the mode allowing it.
The launch file `nav.launch.py` combines localization (unless `slam:=true`) and navigation. The wrappers
`g1_bringup/launch/{slam,nav}.launch.py` resolve a map name to `maps/<name>.yaml`. Goals come from the dashboard
(`hospital-sim-ui`), RViz, `g1_mission` (`NavigateToPose`) or `scripts/survey_runner.py`.

## Dependencies

The table lists the dependencies of this package and the files that use them.

| Dependency | Version | Used by |
|---|---|---|
| `ros-humble-slam-toolbox` | 2.6.10 | `slam.launch.py` |
| `ros-humble-nav2-bringup`, `-common`, `-map-server`, `-amcl`, `-controller`, `-mppi-controller`, `-planner`, `-navfn-planner`, `-smac-planner`, `-smoother`, `-behaviors`, `-bt-navigator`, `-waypoint-follower`, `-velocity-smoother`, `-lifecycle-manager` | 1.1.20 | `localization.launch.py`, `navigation.launch.py`, `nav2_params.yaml` (`SmacPlanner2D`, `KeepoutFilter`) |
| `ros-humble-launch`, `ros-humble-launch-ros` | 1.0.15, 0.19.14 | launch files |
| `ros-humble-rviz2` | 11.2.29 | `rviz/slam_survey.rviz` |
| `ros-humble-rosbag2` | 0.15.17 | survey recordings (`scripts/camera_frame_saver.py` republishes a 1 Hz JPEG for the bag) |
| `rclpy`, `tf2_ros`, `nav_msgs`, `nav2_msgs`, `geometry_msgs`, `sensor_msgs`, `std_msgs`, `visualization_msgs` | Humble apt | `scripts/*.py` |
| `mujoco` 3.14.0, `numpy` 1.26.4, `PyYAML` 6.0.3 | .venv | `scripts/rasterize_world.py`, `survey_runner.py`, `compare_map.py`, `plot_survey_route.py` |
| `scipy` 1.8.0, `Pillow` 9.0.1, `matplotlib` 3.5.1, `PyQt5` 5.15.6 | Ubuntu apt (system site-packages) | `compare_map.py`, `plot_survey_route.py`, `camera_frame_saver.py`, `survey_caption_bar.py` |
| Sibling repository files | | `g1-mujoco-sim/hospital_g1.xml` (ground-truth map), `g1-isaac-sim/tools/results/collider_raster.npz` (survey plots), `hospital-sim-env/isaac/scene/layout.py` (`rasterize_world.py --layout`) |

## Launch files

The table lists the launch files of the package and what each one runs.

| Launch file | What it runs |
|---|---|
| `slam.launch.py [use_sim_time:=true] [base_frame:=pelvis]` | slam_toolbox `async_slam_toolbox_node` (online async mapping) -> `/map`, `map -> odom` |
| `localization.launch.py map:=<yaml>` | map_server + AMCL (`nav2_amcl::OmniMotionModel`) + lifecycle manager |
| `navigation.launch.py [keepout_mask:=<yaml>] [map_topic:=/map]` | controller (MPPI), planner (Smac 2D), smoother, behaviours, BT navigator, waypoint follower, velocity smoother + lifecycle manager |
| `nav.launch.py map:=<yaml>` | localization + navigation on a saved map |
| `nav.launch.py slam:=true` | navigation only, on top of a running SLAM (navigate while mapping) |

The top-level wrappers live in g1_bringup: `ros2 launch g1_bringup slam.launch.py` and
`ros2 launch g1_bringup nav.launch.py [map:=<name|path>]` (no map = on the running SLAM). The dashboard
(`hospital-sim-ui/`) normally runs and stops both.

```bash
# Activate the venv and ROS 2.
source env.sh
# Run the simulator in the hospital world, headless.
ros2 launch g1_bringup sim.launch.py world:=hospital viewer:=false
# Run SLAM, then drive in LOCO (keyboard node or dashboard).
ros2 launch g1_bringup slam.launch.py
# Save the map (or use "Save map" on the dashboard).
ros2 run nav2_map_server map_saver_cli -f $HG1_ROOT/ros2_ws/src/hospital-sim-ros2/g1_navigation/maps/my_map
# Stop SLAM, then run Nav2 on the saved map; set the initial pose (dashboard or RViz), press g, send a goal.
ros2 launch g1_bringup nav.launch.py map:=my_map
```

## Velocity path

The diagram shows how a Nav2 velocity command reaches the controller.

```
controller_server --cmd_vel_nav_raw--> velocity_smoother --/g1/cmd_vel_nav--> g1_teleop_manager (NAV only) --/g1/cmd_vel--> g1_controller
behavior_server (spin / back up) ------/g1/cmd_vel_nav-->
```

## Design choices

The footprint is circular, with `robot_radius: 0.25` (0.30 before the survey of 2026-10-09). The shoulder width of
the G1 is about 0.45 m with the arms in, and the swing of the legs stays inside. The inflation radius is 0.7 m (cost
scaling 10 local, 5 global). It keeps the robot in the middle of 2 m doors and the 4 m corridor.

The controller is MPPI with `motion_model: Omni`. The locomotion policy takes (vx, vy, wz), so the robot is
holonomic-capable. MPPI samples full (vx, vy, wz) trajectories and outputs a smooth, optimised sequence. A walking
policy with 0.3 s to 0.5 s of command-tracking lag follows that sequence better than the per-cycle best-sample
switching of DWB. DWB omni would also work, but it produces jerkier commands and needs many more critics tuned for vy.

Lateral stepping is slower and less stable than walking forward. The parameter `vy` is therefore capped at 0.2 m/s,
and `PreferForwardCritic` + `PathAngleCritic(forward_preference)` keep the robot facing its direction of travel. The
robot uses `vy` for small corrections (door alignment). A batch of 1000 x 40 steps (2 s horizon) at 20 Hz keeps the
CPU load moderate next to MuJoCo and the policy.

The speeds are conservative and inside the contract limits (vx [-0.5, 0.8], vy ±0.4, wz ±1.0). The configured
speeds are vx [-0.2, 0.35] m/s, vy ±0.2 m/s and wz ±0.6 rad/s. The smoother accelerations are 0.3 / 0.3 / 0.8 and the decelerations 0.5 / 0.5 / 1.2.
The placeholder policy walks forward at up to about 0.4 m/s, so Nav2 stays below that.

The progress checker expects 0.3 m within 20 s. A humanoid needs time to start stepping and to turn in place.

The local costmap is a rolling 4 m x 4 m voxel layer. Both `/scan` (slice at head height) and the MID-360 cloud
`/livox/lidar` (0.20 m to 1.80 m band) feed it, so that beds, carts and trolleys below the scan plane are seen. The
global costmap is the static map plus a `/scan` obstacle layer. The value `min_obstacle_height 0.20` is needed
because the swaying head lidar lifts distant floor returns to about 0.1 m in `odom` (measured in the simulator).

The planner is Smac 2D with `allow_unknown: true`, so that navigation also works on a growing SLAM map. NavFn with A*
was the planner before the survey (see "Hospital survey").

AMCL uses the omni motion model, `base_frame_id: pelvis`, 500 to 2000 particles and `set_initial_pose: false`. The
operator sets the initial pose in the dashboard.

slam_toolbox uses `base_frame: pelvis`, 5 cm cells, 15 m maximum range, and adds scans every 0.25 m or 0.25 rad.
The TurtleBot defaults of 0.5 m / 0.5 rad miss door openings at walking pace. The map is published every 2 s for the
live dashboard view. The value `min_laser_range 0.25` drops returns off the arms of the robot.

The argument `use_sim_time` is a launch argument everywhere (default `true`, the simulator publishes `/clock`).

## Maps (`maps/`)

The map `hospital_full.yaml/.pgm` is the default map (nav.launch.py, g1_bringup nav.launch.py). It is a lidar SLAM
map of the whole Isaac hospital ground floor and the surgical wing. The simulated G1 built it on 2026-10-09 and
2026-10-10 while it walked the survey route on its own (see "Hospital survey"). Its frame is the world frame (the
walk started at the world origin facing +x), so the `g1_mission` locations, the speed zones and the dashboard places
line up. The initial pose at the spawn is (0, 0, 0). The map has 5 cm cells and 1605x1225 cells (80 m x 61 m), and
nothing was painted in by hand.

Against the Isaac collider ground truth, the occupied cells are 2.5 cm (mean) and 10 cm (p95) from a true surface.
99.6 % of them are within 10 cm. 96.6 % of the reachable floor is mapped as free. The slam_toolbox pose graph could not be
serialized (see "Tested").

The file `stairs_keepout.yaml/.pgm` is a Nav2 KeepoutFilter mask, not a map. It covers both staircases plus 1.25 m
(plus the 0.25 m robot radius = 1.5 m). The launch file `navigation.launch.py` loads it (`keepout_mask:=`), and both
costmaps use it.

The image `survey_route.png` shows the survey route (config/survey_route.yaml) on the collider raster.

The map `hospital_wing_slam.yaml/.pgm` is a SLAM map built in the MuJoCo simulator (dashboard, LOCO driving,
slam_toolbox). Its `map` frame is the `odom` frame of the robot at the time mapping started (spawn `corridor_west`,
world (8.93, -12.45, 0)).

The map `hospital_gt.yaml/.pgm` is a ground-truth map, not a SLAM map. The script
`scripts/rasterize_world.py --mjcf g1-mujoco-sim/hospital_g1.xml` rasterised it from the MuJoCo world geometry. Every
non-robot geom whose height overlaps 0.08 m to 1.8 m is projected to the floor. Its frame is the world frame of the
simulator. For AMCL on it, set the initial pose in world coordinates (spawn corridor_west = (8.93, -12.45, 0)). It is
useful as a reference, or to test Nav2 without mapping first.

The files `<map>.waypoints.json` hold named waypoint routes ("tasks") saved from the dashboard.

Regenerate the ground-truth map after the simulator world changes:

```bash
# Rasterise the MuJoCo world into maps/hospital_gt.yaml and .pgm.
.venv/bin/python ros2_ws/src/hospital-sim-ros2/g1_navigation/scripts/rasterize_world.py \
    --mjcf g1-mujoco-sim/hospital_g1.xml --out ros2_ws/src/hospital-sim-ros2/g1_navigation/maps/hospital_gt
```

## Hospital survey (Isaac, 2026-10-09/10)

The route `config/survey_route.yaml` has 60 waypoints and about 530 m planned (A* on the Isaac collider raster). The
route is:

```
lobby -> central hall -> west wing -> north-west ward -> central hall -> stair hall (kept 1.5 m off) -> east wing
-> SE lobby -> link -> surgical corridor, sterile store, pre-op, scrub, theatre, PACU -> back to the start
```

The script `scripts/plot_survey_route.py` checks and plots it (`maps/survey_route.png`).

The script `scripts/survey_runner.py` sends each waypoint to Nav2 (NavigateToPose on the live SLAM map, mode NAV). It
retries (it clears the costmaps and runs a scripted turn-and-step "escape" when Nav2 aborts), skips, and optionally
retries the skipped waypoints once (`--second-pass`). It owns and relaunches Nav2 when a costmap stalls (`--nav-cmd`).
It publishes /survey/{trail,route,status,robot_pose} and logs every event (`survey_events.jsonl`).

The helpers are:

- `map_pad_relay.py`: a fixed-size copy of the growing /map for the global costmap
- `map_odom_keepalive.py`: re-stamps the map->odom of slam_toolbox
- `camera_frame_saver.py` and `survey_caption_bar.py`: recording overlay
- `compare_map.py`: SLAM against ground truth
- `rviz/slam_survey.rviz` (+ `slam_survey_dark.qss`): the RViz layout

The survey ran in Isaac, without a camera, at an RTF of about 0.85 to 1 (see g1-isaac-sim/README.md for the flags):

```bash
# Run the Isaac simulator, headless, at the world origin.
./g1-isaac-sim/run.sh --headless --spawn 0,0,0 --scan-level --base-footprint --topic-prefix /isaac
# Run the Isaac relay and the e-stop stub.
python3 g1-isaac-sim/tools/isaac_relay.py --cloud-stride 4 ; python3 g1-isaac-sim/tools/estop_stub.py
# Run the stack on the Isaac backend.
ros2 launch g1_bringup sim.launch.py backend:=isaac
# Run SLAM with base_footprint as the base frame.
ros2 launch g1_navigation slam.launch.py base_frame:=base_footprint
# Run the map relay and the map->odom keepalive.
python3 scripts/map_pad_relay.py --min-period 10 ; python3 scripts/map_odom_keepalive.py
# Run the survey; the runner relaunches Nav2 when a costmap stalls for 15 s.
python3 scripts/survey_runner.py --route config/survey_route.yaml --log-dir <dir> --second-pass \
    --nav-cmd "ros2 launch g1_navigation nav.launch.py slam:=true map_topic:=/map_padded" --stall-s 15
```

The survey needed these changes, and all of them are kept:

- planner NavFn -> Smac 2D (NavFn failed with "Failed to create a plan from potential" on the big growing map)
- robot_radius 0.30 -> 0.25
- the stairs keep-out filter
- the launch arguments `slam.launch.py base_frame:=`, `nav.launch.py map_topic:=` and `navigation.launch.py keepout_mask:=`

The result (run5, datasets/slam_survey/run5) was a 5.7 h simulated walk (78 s -> 20,470 s). The robot reached 50 of
60 waypoints plus the return to the start. It skipped hall_south_west, hall_south_mid, central_north_west,
west_corridor_back, west_corridor_sw, west_room_nw, east_corridor_north, east_room_w and east_north_room. Nav2
aborted next to walls and doorways, mostly with "start in lethal space" after the drifting gait brushed a wall; each
had one retry pass. The areas around them were still mapped (lidar range 15 m). There were no falls.

The 1.5 m keep-out at the stairs was not held. Before the Nav2 keep-out filter was added (simulated time about
9,100 s), the pelvis came within 0 m of the footprint of the central staircase (its bottom corner, t about 8,705 s).
The robot did not climb, and the pelvis height stayed at or below 0.80 m. It came within 0.28 m of the north-west
staircase (t about 5,474 s). With the filter (stairs + 1.25 m) the minimum was 1.04 m.

WARNING: Raise the mask margin (for example to 1.6 m) before you rely on the keep-out filter.

The watchdog of the runner had to relaunch Nav2 on SLAM 125 times (costmap stalls, cause not found). With map_server
and AMCL instead of SLAM, it relaunched 0 times in 25 min.

The Nav2 check on the saved map (datasets/slam_survey/navcheck, `config/nav_check_route.yaml`) used a fresh
simulator and AMCL seeded at the spawn (0, 0, 0). The goals were theatre_instrument_table 784 s (10 recoveries),
east_wing_corridor 423 s (0) and main_lobby 309 s (0). All succeeded, with 0 Nav2 restarts. An earlier order
(theatre -> PACU -> east wing) reached 2/3: the robot wedged itself on the PACU door jamb.

## Tested (MuJoCo simulator, placeholder policy, through the dashboard)

SLAM while driving about 60 m in LOCO produced the saved map `hospital_wing_slam`. AMCL ran on it. A NavigateToPose
goal 14 m down the corridor succeeded (154 s, 3 recoveries at the start while the robot turned round).
FollowWaypoints (link room and back) succeeded (110 s). With the mock robot, SLAM, AMCL, goals, waypoints and
pause/resume all worked.

Not tested: `nav.launch.py slam:=true` on the simulator (tested on the mock only), the real robot and the real
MID-360, and `all.launch.py`.

The slam_toolbox pose-graph serialization fails in this install (see hospital-sim-ui/README.md). It failed again for
hospital_full with "Failed to write file: Exception output stream error". The likely cause (not verified) is that the
scans carry `inf` for no-return beams (REP 117) and the text archive of boost cannot write non-finite floats.
