# hospital-sim-ros2: G1 teleoperation and autonomy stack (ROS 2 Humble)

The ROS 2 side of the hospital G1 project: the `g1_*` packages (MuJoCo sim node, policy controller, mode
manager, keyboard teleop, SLAM and Nav2 configuration, perception nodes, the scrub-nurse mission and the
bringup launches) plus the vendored Unitree messages under `third_party/`. It implements the contract in the
master repo's `docs/interfaces.md`: topics, modes, key bindings, policy sidecar. The sim speaks the real G1's
low-level topics (`/lowcmd`, `/lowstate`, `unitree_hg`), so the same controller, manager and teleop can later
point at the real robot.

At build and run time the packages reach into most of the other repos. `g1_description` installs the model
from `g1-brainco-description`. The sim node loads scenes from `g1-mujoco-sim`. The controller loads trained
policies from `g1-whole-body-rl/exports`. The mission and navigation read `objects/catalog.json` and
`isaac/scene/layout.py` from `hospital-sim-env`. Perception loads ONNX models from `hospital-sim-perception`,
the dashboard launch starts the server in `hospital-sim-ui`, teleop finds its hand landmarker in
`g1-webcam-teleop`, and the Isaac backend is `g1-isaac-sim`. In a `hospital-sim` checkout this repo sits at
`ros2_ws/src/hospital-sim-ros2`, `ros2_ws/` is the colcon workspace, and `env.sh` in the master repo sets
`HG1_ROOT`, which is how the packages find the siblings. Paths in this README are relative to this repo unless
they start with `ros2_ws/` or a sibling repo name.

## Build

```bash
source env.sh                     # in the master checkout: clean LD path + ROS 2 Humble + .venv + this overlay
cd ros2_ws && colcon build --symlink-install
colcon test --packages-select g1_teleop_manager g1_controller && colcon test-result --all   # 143 + 12 tests
colcon test --packages-select g1_mission g1_mujoco_sim && colcon test-result --all          # skills, tree, sim objects
```

`colcon` has to run from the venv python. `env.sh` activates `.venv`, and `.venv/bin/colcon` is a thin wrapper
around `colcon_core`, so plain `colcon build` then works. Without it, ament_python console scripts get a
`#!/usr/bin/python3` shebang and cannot import `mujoco` or `onnxruntime`. If you see
`ModuleNotFoundError: mujoco` from `ros2 run`, rebuild after `source env.sh`, or use `python -m colcon build`.

## Packages

| Package | What |
|---|---|
| `third_party/unitree_ros2/{unitree_hg,unitree_go,unitree_api}` | Vendored Unitree messages (see `SOURCE.md`: commit, BSD-3, one CMake patch) |
| `g1_interfaces` | `msg/Mode` (incl. `AUTO=5`, `HOLD=6`), `srv/SetMode`, `srv/RequestInstrument`, `srv/ControlLease` (contract 3.2, 3.4, 6) |
| `g1_description` | URDF + `robot_state_publisher` launch (robot part of the stack) |
| `g1_mujoco_sim` | MuJoCo sim node: `/lowcmd` -> PD motor model -> `/lowstate` at 500 Hz, `/clock`, `/joint_states`, `/odom` + TF `odom->pelvis`, `/sim/ground_truth`, hands, `/livox/lidar`, `/scan`, `/sim/status`, `/sim/reset`, `/sim/harness`; head RGB-D `/g1/head_camera/*` (on demand, separate render process), `/sim/objects` (ground truth), `/sim/reset_objects`, `fixed_base`, `/sim/fixed_base_goal` (see `g1-mujoco-sim/README.md`) |
| `g1_controller` | Policy runner (ONNX + JSON sidecar) -> `/lowcmd`: damp / stand-up / policy, rate-limited cmd_vel and upper body, fall -> DAMP, `/g1/controller/status`. Ships the placeholder policy in `policies/unitree_g1_12dof_placeholder/` |
| `g1_teleop_manager` | Mode FSM + command arbiter, `/g1/set_mode`, `/g1/mode` (transient-local), key handling from `config/keys.yaml`. AUTO (contract 6): STAND→AUTO only from source `mission`/`ui`, `p`/SetMode → STAND (arms blend back), any → DAMP; in AUTO it forwards `/g1/cmd_vel_nav`, `/g1/auto/upper_body_target`, `/g1/auto/hand/*/cmd` and ignores teleop inputs (`fsm.FORWARD`). HOLD (Stop, `o`): nothing forwarded, arms frozen by the controller, resume only explicitly. Control lease `/g1/control/lease`, UI heartbeat watchdog, hardware e-stop interlock (`/g1/estop/heartbeat`), speed zones (`config/zones.yaml`) in `safety.py` (contract 3.4) |
| `g1_mission` | Scrub-nurse mission: py_trees behaviour tree + skill library (pick, handover, carry, retract, fine align; scripted, swappable for LeRobot policies), LeRobot recorder, offline MuJoCo harness. See `g1_mission/README.md` |
| `g1_teleop_keyboard` | Terminal raw-key node -> `/g1/teleop/key`, with a live status line |
| `g1_navigation` | slam_toolbox + Nav2 (UI/nav part of the stack) |
| `g1_bringup` | `sim.launch.py` (+ `fixed_base:=`, `camera:=`), `teleop.launch.py`, `mission.launch.py` (sim + theatre + [Nav2 + AMCL on hospital_gt] + mission [+ dashboard]), `all.launch.py` (`mission:=true` adds the mission node) (+ slam/nav/dashboard launches added by the UI part) |

## Run (simulation)

```bash
# terminal 1: sim + robot_state_publisher + controller + teleop manager (headless hospital)
source env.sh && ros2 launch g1_bringup sim.launch.py
#   args: world:=hospital|flat  viewer:=true  spawn:=corridor_west|link|lobby|"x,y,yaw"
#         policy:=<g1-whole-body-rl/exports name | g1_controller/policies name | dir>  effort_limits:=auto|model|ultra
#         harness:=true  lidar:=true  record:=/tmp/run.npz  compute_crc:=false

# terminal 2: keyboard
source env.sh && ros2 run g1_teleop_keyboard keyboard_node
#   r stand up (DAMP->STAND; the harness lets go ~2.5 s later) | l locomotion | hold w/s a/d q/e | space stop
#   p pause (->STAND) | m manip (STAND->MANIP) | n leave manip (arms blend back ~1 s) | g nav
#   o STOP (-> HOLD: arms frozen, task paused; r resumes to STAND) | x power off (-> DAMP, the robot sinks)

# everything incl. SLAM / Nav2 / dashboard when those launch files exist:
ros2 launch g1_bringup all.launch.py

# scrub-nurse mission (fixed base at the instrument table; fixed_base:=false walks from corridor_west with Nav2)
ros2 launch g1_bringup mission.launch.py [dashboard:=true]
ros2 service call /g1/mission/request_instrument g1_interfaces/srv/RequestInstrument "{instrument: scissors}"
```

Keys: `r` stand, `p` override (AUTO → STAND, aborts the mission), `o` stop (AUTO → HOLD pauses the mission; the UI's
"resume task" continues it), `x` power off.

Motion (LOCO/MANIP/NAV/AUTO) needs the hardware e-stop heartbeat `/g1/estop/heartbeat`; the sim publishes a stub
(`ros2 service call /sim/estop std_srvs/srv/SetBool "{data: true}"` presses it). Without a sim (teleop.launch.py on a
robot without the e-stop receiver) motion stays locked; DEV override: `manager_node --ros-args -p
DEV_ONLY_ignore_estop_heartbeat:=true`. The controller passes `cmd_vel` in
LOCO, NAV and **AUTO**.

Useful: `ros2 topic echo /g1/controller/status`, `/sim/status` (RTF, step time),
`/g1/teleop/manager_status`, `ros2 service call /sim/reset std_srvs/srv/Trigger`.

End-to-end check (drives the running stack through the real topics):
`python3 ros2_ws/src/hospital-sim-ros2/g1_bringup/scripts/e2e_sim_check.py --json /tmp/e2e.json`.

**Use your own `ROS_DOMAIN_ID`** when several people or agents run stacks on this machine. Two sims on one
domain double `/lowstate` and `/clock`, and another agent's camera-teleop node will drive your arms.

## Real robot (not yet tested on hardware)

Run the controller and manager without the sim, with CRC on, on the robot's DDS network (CycloneDDS, same
topics):

```bash
ros2 run g1_controller controller_node --ros-args -p compute_crc:=true -p policy:=<trained policy>
ros2 launch g1_bringup teleop.launch.py
```

- The placeholder policy is **sim-only**: it was calibrated in MuJoCo and is marginal with the 35 Nm model limits.
- The robot must be in low-level (debug) mode, with its sport service released, before `/lowcmd` is obeyed.

## Trained policies (g1-whole-body-rl/exports)

`g1_controller` reads the `g1-whole-body-rl/exports` sidecar extensions:

- flat ONNX history input `(1, H*frame)`, oldest frame first, all rows filled with the first frame on (re)start
- `onnx.file`, `onnx.input` and `onnx.output`
- obs term `cmd_base_height` (commanded pelvis height minus `home_pelvis_height`), always 0 unless
  `command_details.base_height.enabled`. The command comes from param `base_height` or from topic
  `/g1/cmd_base_height` (std_msgs/Float32) and defaults to the home height.
- `joint_limits`: targets are clipped after default + action * action_scale

With `effort_limits:=auto`, `sim.launch.py` follows the sidecar's `robot.effort_source`: `ultra` means 50 Nm,
`canonical` means the robot model's 35 Nm.

## Placeholder policy

`g1_controller/policies/unitree_g1_12dof_placeholder/` holds Unitree's unitree_rl_gym pre-trained G1 policy
(12 leg DoF). It was converted by `g1_controller/scripts/convert_placeholder.py`, which must run with the conda
`unitree-rl` torch. The policy is an LSTM: the ONNX file takes and returns `h`/`c`, and the sidecar lists the
pair under `onnx.recurrent`. The ONNX rollout matches the TorchScript one exactly (unit test).
`"placeholder": true`. Waist, arms and hands are PD-held at the default pose or at `/g1/upper_body_target`.

Calibration (optional sidecar keys, not in the contract):

- `cmd_vel_scale` [0.4, 1, 1] and `cmd_vel_offset` [-0.13, 0, 0].
- Without the offset the policy creeps forward about 0.15 m/s at zero command on this robot model.
- Forward inputs above about 0.2 drive it into a fast crouched gait that falls.

Limits:

- It does not know about the arms. Large two-arm forward reaches in MANIP (for example shoulder pitch -0.9 with
  elbow 1.0) make it walk off and fall.
- Moderate poses ("hold a tray") are fine (about 0.03 m/s drift).
- The real ULTRA-style whole-body policy comes from `g1-whole-body-rl/exports/<name>/` via `policy:=<name>`, and
  `effort_limits:=auto` then uses the robot model's limits.
