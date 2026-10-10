cmake_minimum_required(VERSION 3.8)
project(g1_navigation)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch maps DESTINATION share/${PROJECT_NAME})
install(DIRECTORY rviz DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/rasterize_world.py scripts/survey_runner.py scripts/survey_caption_bar.py
                 scripts/camera_frame_saver.py scripts/map_pad_relay.py scripts/map_odom_keepalive.py
                 scripts/plot_survey_route.py scripts/compare_map.py DESTINATION lib/${PROJECT_NAME})

ament_package()
