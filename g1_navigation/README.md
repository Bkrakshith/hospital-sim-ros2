# g1_navigation

slam_toolbox + Nav2 (Humble) configuration and launches for the Unitree G1 humanoid in the Hospital_G1
wing. Frames and topics follow `docs/interfaces.md`: base frame `pelvis`, `odom → pelvis` from the sim /
odometry node, `/scan` (2-D slice of the head MID-360), `/livox/lidar` (3-D cloud), Nav2 output on
`/g1/cmd_vel_nav` (the mode manager forwards it only in NAV).

## Launches

| launch | what |
|---|---|
| `slam.launch.py [use_sim_time:=true]` | slam_toolbox `async_slam_toolbox_node` (online async mapping) → `/map`, `map → odom` |
| `localization.launch.py map:=<yaml>` | map_server + AMCL (`nav2_amcl::OmniMotionModel`) + lifecycle manager |
| `navigation.launch.py` | controller (MPPI), planner (NavFn A*), smoother, behaviours, BT navigator, waypoint follower, velocity smoother + lifecycle manager |
| `nav.launch.py map:=<yaml>` | localization + navigation on a saved map |
| `nav.launch.py slam:=true` | navigation only, on top of a running SLAM (navigate while mapping) |

Top-level wrappers live in g1_bringup: `ros2 launch g1_bringup slam.launch.py`,
`ros2 launch g1_bringup nav.launch.py [map:=<name|path>]` (no map = on the running SLAM).
Normally both are started / stopped from the dashboard (`hospital-sim-ui/`).

```bash
source env.sh
ros2 launch g1_bringup sim.launch.py world:=hospital viewer:=false
ros2 launch g1_bringup slam.launch.py                  # drive in LOCO (keyboard node or dashboard)
ros2 run nav2_map_server map_saver_cli -f $HG1_ROOT/ros2_ws/src/hospital-sim-ros2/g1_navigation/maps/my_map   # or dashboard "Save map"
# stop SLAM, then
ros2 launch g1_bringup nav.launch.py map:=my_map       # set the initial pose (dashboard or RViz), key g, send a goal
```

## Velocity path

```
controller_server --cmd_vel_nav_raw--> velocity_smoother --/g1/cmd_vel_nav--> g1_teleop_manager (NAV only) --/g1/cmd_vel--> g1_controller
behavior_server (spin / back up) ------/g1/cmd_vel_nav-->
```

## Design choices

- **Footprint**: circular, `robot_radius: 0.30` (G1 shoulder width ≈ 0.45 m with arms in; the swing
  of the legs stays inside). Inflation radius 0.7 m (cost scaling 10 local / 5 global) keeps the robot
  in the middle of 2 m doors and the 4 m corridor.
- **Controller: MPPI, `motion_model: Omni`**. The locomotion policy takes (vx, vy, wz), so the robot
  is holonomic-capable; MPPI samples full (vx, vy, wz) trajectories and outputs a smooth, optimised
  sequence, which a walking policy with 0.3–0.5 s command-tracking lag follows better than DWB's
  per-cycle best-sample switching (DWB omni would also work but produces jerkier commands and needs
  many more critics tuned for vy). Lateral stepping is slower and less stable than walking forward, so
  `vy` is capped at 0.2 m/s and `PreferForwardCritic` + `PathAngleCritic(forward_preference)` keep the
  robot facing its direction of travel; `vy` is used for small corrections (door alignment). Batch size
  1000 × 40 steps (2 s horizon) at 20 Hz keeps the CPU load moderate next to MuJoCo + the policy.
- **Speeds** (conservative, inside the contract limits vx [-0.5, 0.8], vy ±0.4, wz ±1.0):
  vx [-0.2, 0.35] m/s, vy ±0.2 m/s, wz ±0.6 rad/s; smoother accel 0.3 / 0.3 / 0.8, decel 0.5 / 0.5 / 1.2.
  The placeholder policy walks forward at up to ≈0.4 m/s, so Nav2 stays below that.
- **Progress checker**: 0.3 m within 20 s (a humanoid needs time to start stepping and to turn in place).
- **Costmaps**: local = rolling 4×4 m voxel layer fed by `/scan` (slice at head height) **and** the MID-360
  cloud `/livox/lidar` (0.20–1.80 m band) so beds, carts and trolleys below the scan plane are seen;
  global = static map + `/scan` obstacle layer. `min_obstacle_height 0.20` because the swaying head
  lidar lifts distant floor returns to ~0.1 m in `odom` (measured in the sim).
- **Planner**: NavFn with A*, `allow_unknown: true` (so navigation also works on a growing SLAM map).
- **AMCL**: omni motion model, `base_frame_id: pelvis`, 500–2000 particles, `set_initial_pose: false`
  (the operator sets it in the dashboard).
- **slam_toolbox**: `base_frame: pelvis`, 5 cm, 15 m max range, scans added every 0.25 m / 0.25 rad
  (the TurtleBot defaults of 0.5 m / 0.5 rad miss door openings at walking pace), map published every 2 s
  for the live dashboard view, `min_laser_range 0.25` to drop returns off the robot's own arms.
- `use_sim_time` is a launch argument everywhere (default `true`, the sim publishes `/clock`).

## Maps (`maps/`)

- `hospital_full.yaml/.pgm`: **DEFAULT MAP** (nav.launch.py, g1_bringup nav.launch.py). Lidar SLAM map of the
  whole Isaac hospital ground floor + surgical wing, built 2026-10-09/10 by the simulated G1 walking the survey route
  on its own (below). **World frame** (the walk started at the world origin facing +x), so `g1_mission`
  locations, speed zones and the dashboard places line up; initial pose at the spawn = (0, 0, 0). 5 cm cells,
  1605x1225 cells (80 x 61 m), nothing painted in by hand. Against the Isaac collider ground truth: occupied cells mean 2.5 cm /
  p95 10 cm from a true surface, 99.6 % within 10 cm; 96.6 % of the reachable floor is mapped as free. The
  slam_toolbox pose graph could not be serialized (see Known issues).
- `stairs_keepout.yaml/.pgm`: Nav2 KeepoutFilter mask (not a map): both staircases + 1.25 m (+ 0.25 m robot
  radius = 1.5 m). Loaded by `navigation.launch.py` (`keepout_mask:=`), used by both costmaps.
- `survey_route.png`: the survey route (config/survey_route.yaml) on the collider raster.

- `hospital_wing_slam.yaml/.pgm`: **SLAM map** built in the MuJoCo sim (dashboard, LOCO driving,
  slam_toolbox). Its `map` frame is the robot's `odom` frame at the time mapping started (spawn
  `corridor_west`, world (8.93, -12.45, 0)).
- `hospital_gt.yaml/.pgm`: **GROUND-TRUTH map**, not a SLAM map: rasterised from the MuJoCo world
  geometry (`scripts/rasterize_world.py --mjcf g1-mujoco-sim/hospital_g1.xml`; every non-robot geom whose
  height overlaps 0.08–1.8 m is projected to the floor). Its frame is the **sim world frame**, so for
  AMCL on it set the initial pose in world coordinates (spawn corridor_west = (8.93, -12.45, 0)).
  Useful as a reference or to test Nav2 without mapping first.
- `<map>.waypoints.json`: named waypoint routes ("tasks") saved from the dashboard.

Regenerate the ground-truth map after the sim world changes:
```bash
.venv/bin/python ros2_ws/src/hospital-sim-ros2/g1_navigation/scripts/rasterize_world.py \
    --mjcf g1-mujoco-sim/hospital_g1.xml --out ros2_ws/src/hospital-sim-ros2/g1_navigation/maps/hospital_gt
```

## Hospital survey (Isaac, 2026-10-09/10)

`config/survey_route.yaml`: 60 waypoints, ~530 m planned (A* on the Isaac collider raster), lobby -> central hall
-> west wing -> north-west ward -> central hall -> stair hall (kept 1.5 m off) -> east wing -> SE lobby -> link ->
surgical corridor, sterile store, pre-op, scrub, theatre, PACU -> back to the start. `scripts/plot_survey_route.py`
checks/plots it (`maps/survey_route.png`). `scripts/survey_runner.py` sends each waypoint to Nav2 (NavigateToPose on
the live SLAM map, mode NAV), retries (clear costmaps, a scripted turn-and-step "escape" when Nav2 aborts), skips,
optionally retries skipped ones once (`--second-pass`), owns/relaunches Nav2 when a costmap stalls (`--nav-cmd`),
publishes /survey/{trail,route,status,robot_pose} and logs every event (`survey_events.jsonl`). Helpers:
`map_pad_relay.py` (fixed-size copy of the growing /map for the global costmap), `map_odom_keepalive.py`
(re-stamps slam_toolbox's map->odom), `camera_frame_saver.py`, `survey_caption_bar.py` (recording overlay),
`compare_map.py` (SLAM vs ground truth). RViz layout: `rviz/slam_survey.rviz` (+ `slam_survey_dark.qss`).

How it was run (Isaac, no camera, RTF ~0.85-1; see g1-isaac-sim/README.md for the flags):
```bash
./g1-isaac-sim/run.sh --headless --spawn 0,0,0 --scan-level --base-footprint --topic-prefix /isaac
python3 g1-isaac-sim/tools/isaac_relay.py --cloud-stride 4 ; python3 g1-isaac-sim/tools/estop_stub.py
ros2 launch g1_bringup sim.launch.py backend:=isaac
ros2 launch g1_navigation slam.launch.py base_frame:=base_footprint
python3 scripts/map_pad_relay.py --min-period 10 ; python3 scripts/map_odom_keepalive.py
python3 scripts/survey_runner.py --route config/survey_route.yaml --log-dir <dir> --second-pass \
    --nav-cmd "ros2 launch g1_navigation nav.launch.py slam:=true map_topic:=/map_padded" --stall-s 15
```
Changes this needed (all kept): planner NavFn -> Smac 2D (NavFn failed with "Failed to create a plan from potential"
on the big growing map), robot_radius 0.30 -> 0.25, stairs keep-out filter, `slam.launch.py base_frame:=`,
`nav.launch.py map_topic:=`, `navigation.launch.py keepout_mask:=`.

Result (run5, datasets/slam_survey/run5): 5.7 h simulated walk (78 s -> 20,470 s), 50 of 60 waypoints reached
plus the return to the start; skipped: hall_south_west, hall_south_mid, central_north_west, west_corridor_back,
west_corridor_sw, west_room_nw, east_corridor_north, east_room_w, east_north_room (Nav2 aborts next to walls /
doorways, mostly "start in lethal space" after the drifting gait brushed a wall; one retry pass each). The areas
around them were still mapped (lidar range 15 m). No falls. Stairs: the 1.5 m keep-out was NOT held. Before the Nav2 keep-out filter was added (sim ~9,100 s) the
pelvis came within 0 m of the central staircase's footprint (its bottom corner, t ~8,705 s; the robot did not climb,
pelvis height stayed <= 0.80 m) and 0.28 m of the north-west one (t ~5,474 s); with the filter (stairs + 1.25 m)
the minimum was 1.04 m. Raise the mask margin (e.g. 1.6 m) before relying on it.
Nav2 on SLAM had to be relaunched 125 times by the runner's watchdog (costmap stalls, cause not found; with
map_server + AMCL instead of SLAM: 0 in 25 min).

Nav2 check on the saved map (datasets/slam_survey/navcheck, `config/nav_check_route.yaml`): fresh sim, AMCL seeded
at the spawn (0, 0, 0), goals theatre_instrument_table 784 s (10 recoveries), east_wing_corridor 423 s (0),
main_lobby 309 s (0), all succeeded, 0 Nav2 restarts. An earlier order (theatre -> PACU -> east wing) reached 2/3:
the robot wedged itself on the PACU door jamb.

## Tested (MuJoCo sim, placeholder policy, via the dashboard)

SLAM while driving ~60 m in LOCO → saved `hospital_wing_slam`; AMCL on it; NavigateToPose 14 m down
the corridor succeeded (154 s, 3 recoveries at the start while the robot turned round); FollowWaypoints
(link room and back) succeeded (110 s). With the mock robot: SLAM, AMCL, goals, waypoints, pause/resume.

Not tested: `nav.launch.py slam:=true` on the sim (tested on the mock only), the real robot / real
MID-360, `all.launch.py`. slam_toolbox pose-graph serialization fails in this install (see hospital-sim-ui/README.md):
"Failed to write file: Exception output stream error" again for hospital_full. Likely cause (not verified): the
scans carry `inf` for no-return beams (REP 117) and boost's text archive cannot write non-finite floats.
