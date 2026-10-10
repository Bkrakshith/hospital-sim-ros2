# g1_mission: scrub-nurse mission and skill library

This package is step 1 of the autonomous scrub nurse (`docs/interfaces.md` section 6). The G1 walks from the
corridor to the operating theatre and lines up at the instrument table. It waits for an instrument request, picks the
instrument out of the tool rack, and carries it to the surgeon. With the `hospital-sim-env/objects` layout, the
surgeon stands 2.6 m away at the operating table. The robot hands the instrument over handle first, walks back to the
table and returns to its ready pose. The operator can override at any time: `p` returns to STAND, `x` damps the
motors.

Object poses come from the ground truth of the simulator for now. Each of these cheats sits behind an interface. The
section [Cheats and what replaces them](#cheats-and-what-replaces-them) says what replaces each one. The build,
install and stack-level run commands are in the repository README (`../README.md`). This file describes the mission
package in detail.

## How it works

The file `mission_node.py` is one rclpy node. On start it loads `config/mission.yaml`, `config/locations.yaml` and
the instrument catalogue. The catalogue is `config/instruments.yaml` merged with `config/grasp_overrides.yaml`. The
node builds the `scrub_nurse` py_trees tree from `tree.py` and waits. A `RequestInstrument` service call, or `start`
on `/g1/mission/command`, puts an instrument in the queue. The tree then asks the mode manager for AUTO (source
`mission`).

When the robot is not at the station, the tree navigates with a Nav2 `NavigateToPose` goal. It aligns the last
centimetres with `/g1/cmd_vel_nav` and then runs the arm skills. A timer ticks the tree at `rate` (50 Hz,
`mission.yaml`). Every tick, the active skill produces one command through
`SkillContext.robot.command(upper, left_hand, right_hand, cmd_vel)`. The class `ros_io.RosRobot` publishes that
command as `/g1/auto/upper_body_target` (17 joints: waist 3 + arms 14), `/g1/auto/hand/{left,right}/cmd` (6 values
0..1) and `/g1/cmd_vel_nav`. The mode manager forwards those topics only in AUTO.

Arm targets are Cartesian or joint segments from `skills/base.ArmMotion`. The solver `kinematics.UpperBodyIK`
(pinocchio on the `_ik` URDF, pelvis frame) solves them every tick. Object and surgeon poses arrive through the
sources in `interfaces.py`. The default source is the `/sim/objects` ground truth. With `object_source:=perception`
and `surgeon_source:=perception` the sources are `/g1/perception/detections` and `/g1/perception/surgeon_hand`.
Status goes out on `/g1/mission/status` (JSON, 5 Hz and on change).

If `/g1/mode` leaves AUTO, the node stops the tree. HOLD pauses the tree. The module `offline.py` runs the same tree
and skills on an in-process MuJoCo model without ROS. The tests and `scripts/offline_trials.py` use it.

The mission commands below need a running stack (see the repository README):

```bash
# Request an instrument; this starts the mission when it is idle.
ros2 service call /g1/mission/request_instrument g1_interfaces/srv/RequestInstrument \
    "{instrument: scalpel_handle, requester: cli}"
# Send a mission command: start | abort | pause | resume | goto:<loc>.
ros2 topic pub --once /g1/mission/command std_msgs/String "{data: abort}"
```

The Mission page of the dashboard does the same (`dashboard:=true`, http://localhost:8080/#/mission).

## Dependencies

The table lists the dependencies of this package and the files that use them.

| Dependency | Version | Used by |
|---|---|---|
| ROS 2 Humble: `rclpy`, `std_msgs`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `nav2_msgs`, `action_msgs`, `tf2_ros`, `g1_interfaces` | apt / this repository | `mission_node.py`, `ros_io.py` (`NavigateToPose` action client) |
| `py_trees` | 2.6.0 (.venv) | `tree.py`, `mission_node.py` |
| `numpy` | 1.26.4 (.venv) | everywhere |
| `pin` (pinocchio) | 3.9.0 (.venv) | `kinematics.py` IK on `g1-brainco-description/g1_29dof_brainco/urdf/g1_29dof_brainco_ik.urdf` |
| `PyYAML` | 6.0.3 (.venv) | `config.py`, `scripts/make_locations.py` |
| `mujoco` | 3.14.0 (.venv) | `offline.py`, `scripts/check_extraction.py`, tests (with `g1_mujoco_sim.scene_objects`, `model_map`) |
| `pyarrow`, `imageio` (+ `imageio-ffmpeg`), `opencv-contrib-python` | 25.0.1, 2.38.0 (0.6.0), 4.10.0.84 (.venv) | `recorder.py` (LeRobot v2.1 parquet + h264 mp4), `scripts/offline_trials.py` |
| `torch`, `lerobot` | 2.3.0+cpu in .venv; lerobot not installed | `skills/policy.py` only; the scripted fallback runs without them |
| `pytest` | 6.2.5 (Ubuntu apt) | `test/` |
| Sibling repository files | | `g1-mujoco-sim/worlds/theatre_setup.json` + `hospital-sim-env/isaac/scene/layout.py` (input of `scripts/make_locations.py`), `hospital-sim-env/objects/catalog.json`, `g1-mujoco-sim/hospital_g1.xml` (offline harness) |

The pins are in `../requirements.txt`.

## Behaviour tree (`g1_mission/tree.py`, py_trees)

```
scrub_nurse (Sequence)
|-- EnsureAuto                        SetMode AUTO (source "mission"); the robot must be in STAND
|-- ReachStation (Selector)
|   |-- AtStation?                    within at_station_tolerance (0.30 m) of theatre_instrument_table
|   `-- GoTo(theatre_instrument_table)    Nav2 NavigateToPose (map frame, approach_back 0.25 m short), arms at walk_pose
|-- ToReady                           ReturnToReady skill: arms to the ready pose (forearms up, hands open)
|-- FineAlign                         last centimetres with cmd_vel (StationEstimator); fixed base: glide
|                                     (walking: FineAlign runs before ToReady, see below)
`-- Serve (Repeat forever; repeat:=false = once + ReleaseAuto)
    `-- ServeOrRecover (Selector)
        |-- ServeOne (Sequence)
        |   |-- WaitForRequest        queue filled by /g1/mission/request_instrument
        |   |-- LocateInstrument      ObjectSource: in reach, not already with the surgeon
        |   |-- Pick                  skill
        |   |-- ToSurgeon (Selector)  only when handover_station != station (hospital-sim-env/objects layout)
        |   |   |-- SurgeonInReach?   surgeon hand within handover_reach (0.70 m)
        |   |   `-- CarryToSurgeon    Carry skill (instrument close to the body), then
        |   |       `-- GoToSurgeon   At(theatre_surgeon_side)? | GoTo + FineAlign   (fixed base: glide)
        |   |-- Handover              skill
        |   |-- Retract               skill
        |   |-- ReturnToReady         skill
        |   |-- BackToTable           At(station)? | GoTo + FineAlign                (only with a handover_station)
        |   `-- Done                  log success
        `-- Recover (Sequence)        a failed request is logged, the hand opened, the arm back at ready, the
            |-- LogFailure            robot back at the table, and the next request is served
            |-- RecoverToReady
            `-- RecoverToTable
```

The setting `handover_station: auto` (mission.yaml) takes `handover_station` from `config/locations.yaml`. That is
`theatre_surgeon_side` in the hospital-sim-env/objects layout, and the table itself with the stand-ins.

The command `goto:<location>` runs `EnsureAuto -> GoTo(location) -> [ToReady, FineAlign if the location has
fine_align] -> ReleaseAuto`.

The arm pose and the alignment order follow these rules:

- The arms are at `walk_pose` (the policy default) while the robot walks. Raised arms made the placeholder drift
  backwards.
- In fixed-base mode the arms go to the ready pose before the final glide in. At the policy default the hands would
  sit in the rack.
- When the robot walks, FineAlign runs first and the arms rise afterwards.

The mission node (`mission_node.py`) behaves as follows:

- It ticks the tree at 50 Hz and keeps `/g1/auto/*` alive. The controller drops arm targets older than 0.5 s.
- Operator override: if `/g1/mode` leaves AUTO while a mission runs (`p`, `x`, a fall), the tree stops at once. The
  running skill stops, the node cancels the Nav2 goal and the state becomes `aborted`. The mode manager blends the
  arms back to default. The mission does not request AUTO again by itself.
- Stop (HOLD, contract 3.2): AUTO -> HOLD pauses the mission instead of aborting it. The status shows
  `paused_by_hold`, the skills freeze, the node cancels the Nav2 goal, and the controller freezes the arms where they
  are. The command `resume` then requests HOLD -> AUTO (source `mission`) and the tree continues. If the robot leaves
  HOLD any other way (resume to STAND = cancel, power off), the node aborts the mission.
- The command `pause` freezes the tree. Arm targets hold, the node cancels the Nav2 goal, and FineAlign sends zero
  velocity. The command `resume` shifts the timeline of every skill by the pause length and sends the Nav2 goal again.
- The command `abort` stops the tree and requests STAND.
- The status goes on `/g1/mission/status` as the contract JSON `{"state","step","instrument","tree":[{name,status}],
  "message"}` plus these extras:
  - tree entries also carry `depth`, `type` and `message`
  - `queue`, `instruments`, `instrument_info`, `locations`, `mode`, `station_error`, `history`, `counts`,
    `recording`, `sources`

The table lists the ROS parameters of the mission node.

| Parameter | Default | Values |
|---|---|---|
| `object_source` | `ground_truth` | `ground_truth` (`/sim/objects`) or `perception` (`/g1/perception/detections`) |
| `surgeon_source` | `ground_truth` | `ground_truth` (`/sim/objects`) or `perception` (`/g1/perception/surgeon_hand`) |
| `base_source` | `ground_truth` | `ground_truth` (`/sim/ground_truth`) or `tf` (`map -> pelvis`) |
| `use_nav` | `true` | creates the Nav2 `NavigateToPose` client |
| `initial_pose` | `""` | `""`, `ground_truth` or `"x,y,yaw"`; publishes `/initialpose` for AMCL |
| `start_on_request` | `true` | an instrument request starts the mission when it is idle |
| `autostart` | `false` | starts the mission as soon as the robot is in STAND |
| `repeat` | `true` | serves requests forever; `false` serves once and releases AUTO |
| `record_dir` | `""` | folder of the LeRobot dataset; `""` turns recording off |
| `record_fps` | 15 | frame rate of the recording |
| `config` | `""` | path of `mission.yaml`; `""` uses the package `config/` |

## Skills (`g1_mission/skills/`)

Every skill implements `skills.base.Skill`:

```python
class Skill:
    name = "pick"
    def start(self, ctx: SkillContext, **goal): ...     # once
    def tick(self, ctx) -> Status: ...                  # RUNNING / SUCCESS / FAILURE, at 50 Hz
    def stop(self, ctx): ...                            # abort / pause
    phase: str; message: str                            # shown in the tree view
```

A skill acts only through `ctx.robot.command(upper=17 joint targets, left_hand=6, right_hand=6, cmd_vel=3)`. These
are the channels that teleop and a learned policy also use. A skill perceives only through `ctx.robot` (joint and
hand state, base pose, head image), `ctx.objects`, `ctx.surgeon` and `ctx.station` (`interfaces.py`).

The table lists the skills and what each one does.

| Skill | File | What |
|---|---|---|
| `locate` | `locate.py` | instrument pose from the ObjectSource; it fails if the instrument is missing, out of reach (> 0.75 m) or with the surgeon |
| `pick` | `pick.py` | scripted power grasp. Steps: preshape (thumb opposed), above (turn the hand away from the rack), pre-grasp 10 cm back along the approach, approach, close, lift 12 cm, retreat 8 cm. It then verifies that the instrument moved with the hand and stores the measured grasp transform. The approach is the instrument axis that faces the robot (`approach_axis`: x for hospital-sim-env/objects, whose rings and jaws point at the robot; y for the stand-ins) |
| `handover` | `handover.py` | searches the presentation: handle end 1 cm short of the hand of the surgeon, axis direction over yaw offsets from the shoulder-to-hand direction and 0/-20 degree tilt, 12 rolls. It needs a reachable present pose and pre-pose, scored by IK residual, joint-limit margin (0.25 rad), waist use and joint travel. The search runs incrementally, about 12 ms per tick, so the arm stream never stalls. Then: joint move to the pre-pose, straight in, wait until the surgeon holds it, release |
| `carry` | `retract.py` | before the robot walks, the held instrument comes close to the body (TCP at 0.22 m forward, 0.20 m up) so that it cannot sweep into people. Without this the instrument hit the arm of the surgeon on arrival |
| `retract` | `retract.py` | the empty hand backs away against the palm normal, then along the handover axis. It fails if the instrument is still in the hand |
| `return_to_ready` | `retract.py` | joint-space move to the ready pose with the hands open. A low hand first rises clear of the rack |
| `fine_align` | `fine_align.py` | proportional body-frame `cmd_vel` on `/g1/cmd_vel_nav` (forwarded in AUTO) until within 5 cm / 0.06 rad for 1 s. `vx_forward_min` is a forward floor for policies with a command offset. It aborts if the error grows 25 cm above its best (divergence guard). In fixed base it glides exactly instead |
| (policy) | `policy.py` | `LearnedPolicySkill`: runs a LeRobot policy in place of any of the above |

Per-instrument grasps: the hand-maintained file `config/grasp_overrides.yaml` is merged over the generated
`instruments.yaml` by `config.load_catalog`. It can set `grasp_z`, `hand_close` (6 Revo2 closures), `lift` and
`straighten`. It also records a `note` per instrument with its known issues. Pick reads the merged values.

The skills share two pieces of machinery:

- `kinematics.UpperBodyIK` runs pinocchio on `g1-brainco-description/g1_29dof_brainco/urdf/g1_29dof_brainco_ik.urdf`,
  all in the pelvis frame. It is a weighted damped least-squares solver (waist weighted 3x) with a posture term in the
  task nullspace. The TCP `right_grasp` is at `wrist_yaw_link` + (0.125, 0.040, 0). That is the centre of the Revo2
  power-grasp pocket, with the object axis along TCP z. It was measured from the MJCF finger kinematics.
- `skills.base.ArmMotion` generates Cartesian segments (IK every tick) and joint segments with smoothstep timing and
  ramped hand commands. It also runs a slow integral term per arm joint (ki 3/s, clamp 0.15 rad). That term removes
  the steady-state sag of the PD arm of the controller (placeholder kp 20-100) under gravity and payload. Without it
  the sag would be 1 cm to 3 cm at the hand.

### Swapping a scripted skill for a learned policy (ACT / Diffusion Policy through LeRobot)

1. Record demonstrations with `mission.launch.py record_dir:=/data/g1_scrub`. Each `pick` and `handover` execution
   becomes one LeRobot v2.1 episode (`g1_mission/recorder.py`, format below). The scripted skills are the first
   demonstrator. Camera-teleop demonstrations in MANIP use the same channels, and the same conversion applies.
2. Train with LeRobot (not done here, and no policy has been trained). For example:
   `lerobot/scripts/train.py --dataset.root=/data/g1_scrub --policy.type=act`.
3. Point the skill at the checkpoint in `config/mission.yaml`:
   `handover: {impl: policy, policy_path: /path/to/pretrained_model, fallback: scripted, max_time: 30}`.

The class `LearnedPolicySkill` builds `observation.state` (29) and `observation.images.head` from `ctx.robot`. It
calls `policy.select_action` at the mission rate and splits the 32-D action back into `command(...)`. It needs
`pip install lerobot` in `.venv` (not installed yet). Until a policy loads, it logs the reason and runs the fallback.
The success checks are the scripted ones. The instrument moving with the hand verifies a pick;
`SurgeonHandSource.holding()` verifies a handover.

### Recording format (LeRobot v2.1, off by default)

The vector `observation.state` (29, float32) uses GR00T-style groups:

| group | slice | contents |
|---|---|---|
| `waist` | 0:3 | waist yaw, roll, pitch (rad, measured) |
| `left_arm` | 3:10 | shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw |
| `right_arm` | 10:17 | same order |
| `left_hand` | 17:23 | thumb, thumb_aux, index, middle, ring, pinky (0 open .. 1 closed, measured) |
| `right_hand` | 23:29 | same order |

The vector `action` (32, float32) is what the skill commanded. It has the same groups (joint and hand targets) plus
`base_velocity` 29:32 (vx, vy, wz = cmd_vel). The image `observation.images.head` is the head RGB, 640x360 (like
Isaac), 15 fps, h264.

The final fixed-base ROS trial session (`record_dir:=`) verified the format: 60 episodes (pick + handover for 30
requests), 8 626 frames, 61 MB. The frames were real head-camera frames, and the state and action ranges looked
right. That run was still at 640x480; the camera has been 640x360 since.

There is no "31-D" state: 3 + 14 + 12 = 29. The base velocity is only in the action (the arm skills run standing).
Add `observation.base_velocity` if a walking skill is ever learned.

The recorder writes these files:

- `meta/info.json`: features, fps, and `g1_modality` with the group slices
- `meta/episodes.jsonl`, `meta/tasks.jsonl` (task = "pick needle_driver"), `meta/episodes_stats.jsonl`
- `data/chunk-000/episode_%06d.parquet`: also `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`,
  `next.success`
- `videos/chunk-000/observation.images.head/episode_%06d.mp4`

## Cheats and what replaces them

The table lists each cheat of step 1, where it lives, and the replacement behind the interface that is already in
place.

| Cheat (step 1) | Where | Replacement (interface already in place) |
|---|---|---|
| Instrument poses from `/sim/objects` | `interfaces.GroundTruthObjects` | `DetectionObjects` on `/g1/perception/detections` (`object_source:=perception`); poses in other frames go through TF |
| Surgeon hand from `/sim/objects` | `GroundTruthSurgeonHand` | `PerceptionSurgeonHand` on `/g1/perception/surgeon_hand` (`surgeon_source:=perception`) |
| Surgeon "grip" = simulator weld after proximity + dwell | `g1_mujoco_sim/scene_objects.py`, `SurgeonHandSource.holding()` | tug or force detection in the hand (Revo2 current), or perception. `holding()` returning None already makes the handover release after `blind_wait` |
| The surgeon stand-in puts the held instrument back in its rack slot when the next one is offered (out of the rack, within 0.40 m of the hand) | `scene_objects.update_grip` / `_return_to_rack` | a human (or a used-instrument tray) |
| Pelvis pose from `/sim/ground_truth` (arm targets, FineAlign) | `RosRobot.base_T` (`base_source`), `GroundTruthStation` | TF map->pelvis from AMCL for navigation; for manipulation, object poses measured in the robot frame (head camera) make the world pose unnecessary; FineAlign: a table marker (ArUco/AprilTag) or a table-edge estimate that implements `StationEstimator` |
| AMCL initial pose from ground truth | `mission_node.send_initial_pose` (`initial_pose:=ground_truth`) | the operator sets it on the dashboard, or a known docking pose |
| Grasp transform measured from ground truth after the pick | `ScriptedPick._verify` | nominal grasp (already the fallback), or in-hand pose from the head camera |
| Fixed base (pelvis welded) for the arm part | `g1_mujoco_sim fixed_base:=true` | the whole-body (Stage B) policy with arm reaching |
| Walking between the table and the surgeon in fixed-base mode = straight-line glide of the welded base (no collision checks) | `/sim/fixed_base_goal`, `ros_io.GlideNav`, the glide branch of FineAlign | Nav2 + the walking policy (already the path when `fixed_base:=false`) |
| The hand and forearm of the surgeon mannequin are contact-free (a rigid hand cannot yield) | `scene_objects._soften_surgeon_hand` | a person (or a compliant hand model) |

## Results (MuJoCo, this machine, 2026-09-30)

These results come from `scripts/run_trials.py` (full ROS stack: simulator + controller + mode manager + mission
node) and `scripts/offline_trials.py` (the same tree and skills, in-process). The rack offsets are uniform in ±2 cm
x/y and ±0.08 rad (4.6 degrees) yaw, applied to the rack and its instruments.

### hospital-sim-env/objects layout

These trials used the assets of the perception agent. Each trial is: pick at the table stance, carry, fixed-base
glide 2.6 m to the surgeon side, handover, walk back, ready. One trial takes about 45 s of simulated time, and the
RTF stays at 1.0. The table lists the runs and their results.

| Run | Result |
|---|---|
| ROS, final hospital-sim-env/objects build, LeRobot recording on | scalpel_handle 9/10, adson_forceps 10/10; extra runs of veress_needle 5/5 and trocar 5/5. The scalpel failure happened after the handover: the glide back to the table was judged "failed" from the 1 Hz status of the simulator 1 s after it started. The glide now tracks arrival from ground truth (fixed after this run) |
| ROS, earlier hospital-sim-env/objects build (4 mm rack clearance), before the handover servo step | scalpel_handle 8/10 (2 handle-not-at-hand), scissors 10/10 |
| offline, current assets + code (3 per instrument) | scalpel_handle 3/3, adson_forceps 3/3, veress_needle 3/3, trocar 3/3, needle_driver 2/3 (once stuck on the fingers), angled_tweezers 0/3, scissors 0/3 |

The scissors and the angled tweezers fail against the hospital-sim-env/objects rebuild of 01:43 (3 mm rack
clearance). The cause is geometric, not the skill. The script `scripts/check_extraction.py` translates each instrument
straight up out of its slot. The scissors start to interfere with the rack at 2.5 cm of lift and reach 8.8 mm of
interference at 4 cm. The tweezers reach 4.2 mm at 3.5 cm. The holes are sized to the section at guide height, but
the instrument is wider above it. With the earlier 4 mm build, the scissors succeeded 10/10 through ROS.

### Stand-in layout

This layout comes from `build_world.py --theatre standin`. It has 4 box instruments in a free rack, and the hand of
the surgeon is reachable from the table. The friction cones were pyramidal at the time. The table lists the runs and
their results.

| Run | Result |
|---|---|
| ROS | 40/40: scalpel_handle, forceps, needle_driver, scissors, 10 each. About 21 s per trial |
| ROS, earlier ready pose | 19/20. The one failure was the hand sweeping an instrument off the rack; that failure motivated the raised ready pose and the "above" step |

### Walking

The walking test used `fixed_base:=false`, the placeholder unitree_rl_gym policy, Nav2 MPPI + AMCL on `hospital_gt`,
and AMCL seeded from ground truth. It ran with `scripts/walk_test.py`, `goto:theatre_instrument_table` from
`corridor_west`.

- Nav2: success. The robot walked from the corridor spawn (8.97, -12.51) into the theatre in 151 s of simulated
  time. It stopped at (25.81, -19.81), 0.31 m / 0.26 m / 3 degrees from the station (the goal is 0.25 m short of
  it). There was no fall.
- FineAlign: did not finish. In 60 s it brought the error from 31/26 cm to 6.1/4.3 cm and 1.3 degrees, against a
  5 cm tolerance, then timed out.
  - The placeholder does not follow small commands. Its calibrated map is policy vx = 0.4 * vx - 0.13, so it needs
    about 0.34 m/s commanded to creep forward. It then overshoots.
  - FineAlign therefore uses a forward floor (`vx_forward_min`) and a divergence guard.
- Arm part: not possible with this policy. Raising the arms to the ready pose made the robot walk backwards (0.35 m
  in about 3 s, and 12 m in an earlier run where FineAlign kept pushing). The arm skills need the whole-body
  (Stage B) policy, so they are verified in fixed-base mode only.
- Found and fixed on the way: the `/clock` of the simulator went back 1 s every few seconds. The function `_stamp()`
  turned t = N.999999999 into N-1.0. Every tf2 buffer in Nav2 and AMCL then logged "jump back in time" and cleared
  itself (about 600 times in 150 s), and robot_state_publisher warned "Moved backwards in time". Nav2 aborted the
  first walk attempt (arms raised while walking, clock bug present) after the robot had drifted 2.8 m backwards.

## Files

The table lists the files and folders of the package.

| Path | What |
|---|---|
| `g1_mission/mission_node.py` | ROS node (tree runner, commands, requests, status, override handling, AMCL seed) |
| `g1_mission/tree.py` | behaviour tree + `MissionCore` (ROS-free) |
| `g1_mission/skills/*.py` | skill library |
| `g1_mission/interfaces.py` | Robot / Object / SurgeonHand / Station interfaces + ground-truth and perception implementations |
| `g1_mission/ros_io.py` | `RosRobot`, `NavClient` |
| `g1_mission/kinematics.py`, `geometry.py` | pinocchio IK, transforms |
| `g1_mission/recorder.py` | LeRobot v2.1 recorder |
| `g1_mission/offline.py` | the same skills on an in-process MuJoCo model (no ROS): fast iteration, tests, batch trials |
| `config/locations.yaml`, `config/instruments.yaml` | generated by `scripts/make_locations.py` from `g1-mujoco-sim/worlds/theatre_setup.json` |
| `config/mission.yaml` | rate, station, ready pose, skill implementations and parameters |
| `scripts/run_trials.py` | trials through the ROS stack (random rack offsets through the `object_offset` of the simulator + `/sim/reset_objects`) |
| `scripts/offline_trials.py` | the same trials offline (optionally with an MP4 from a fixed camera) |
| `scripts/walk_test.py` | walking check: Nav2 to the table, FineAlign, optional arm request, with a per-step log |
| `scripts/check_extraction.py` | checks whether every instrument can be pulled straight out of the rack (geometry only) |
| `config/grasp_overrides.yaml` | per-instrument grasp parameters and known issues (hand-maintained) |
| `test/` | unit tests, offline mission tests (real tree + physics), no ROS graph needed |

Run the tests with `colcon test --packages-select g1_mission` from `ros2_ws` (13 tests). The full command and the
counts are in the repository README under "Test / check".
