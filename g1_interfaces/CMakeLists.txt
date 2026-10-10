cmake_minimum_required(VERSION 3.8)
project(g1_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Mode.msg"
  "srv/SetMode.srv"
  "srv/RequestInstrument.srv"
  "srv/ControlLease.srv"
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
