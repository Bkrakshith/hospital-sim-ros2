cmake_minimum_required(VERSION 3.8)
project(g1_bringup)
find_package(ament_cmake REQUIRED)
# Installs every file in launch/ (other agents add slam/nav/dashboard launch files here; rebuild to pick
# them up) and config/ if present.
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
  install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
endif()
ament_package()
