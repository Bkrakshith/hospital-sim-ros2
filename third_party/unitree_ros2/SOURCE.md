# Vendored: unitree_ros2 message packages

- Upstream: https://github.com/unitreerobotics/unitree_ros2
- Commit: `668d1ec5a05d1c38d3306bdca7d59f2ba3581a88` (2026-07-02)
- Path copied: `cyclonedds_ws/src/unitree/{unitree_hg,unitree_go,unitree_api}` (unmodified)
- Licence: BSD 3-Clause, Copyright (c) 2016-2024 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics") — see `LICENSE`.

Only the message packages are vendored (no examples, no SDK). These give the same
`/lowcmd` / `/lowstate` types as the real G1 (`unitree_hg/msg/LowCmd`, `LowState`, 35 motor slots;
slots 0..28 are the 29 body joints in docs/interfaces.md section 2 order).

## Local modification

The three `CMakeLists.txt` files are patched in one place each: `find_package(rosidl_generator_dds_idl)`
is `QUIET` instead of `REQUIRED`, and the `rosidl_generate_dds_interfaces(... dds_connext ...)` block is
wrapped in `if(rosidl_generator_dds_idl_FOUND)`. That generator only produces RTI Connext IDL, is not
installed on this machine (no apt), and is irrelevant for CycloneDDS / Fast DDS. Message definitions,
names and wire format are unchanged.
