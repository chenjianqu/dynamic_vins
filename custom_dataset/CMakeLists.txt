cmake_minimum_required(VERSION 3.0.2)
project(custom_dataset)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  std_msgs
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories( ${EIGEN3_INCLUDE_DIR})

find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV DIR:  ${OpenCV_DIR}")
message("OpenCV_INCLUDE_DIRS:  ${OpenCV_INCLUDE_DIRS}")
message("OpenCV OpenCV_VERSION:  ${OpenCV_VERSION}")

find_package(spdlog REQUIRED)
include_directories(${spdlog_INCLUDE_DIRS})
message("spdlog version:${spdlog_VERSION}")
message("spdlog dir:${spdlog_DIR}")




catkin_package(
)

include_directories(
        # include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)


add_executable(sub_write_images src/sub_write_images.cpp src/io_utils.cpp)
target_link_libraries(sub_write_images ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES} )

add_executable(read_stereo_cam src/read_stereo_cam.cpp src/io_utils.cpp)
target_link_libraries(read_stereo_cam ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES} )

add_executable(pub_stereo_cam src/pub_stereo_cam.cpp src/io_utils.cpp)
target_link_libraries(pub_stereo_cam ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES} )

add_executable(capture_single_image src/capture_single_image.cpp src/io_utils.cpp)
target_link_libraries(capture_single_image ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES} )

add_executable(calib_camera src/calib_camera.cpp src/io_utils.cpp)
target_link_libraries(calib_camera ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES} )

add_executable(write_rosbag src/write_rosbag.cpp src/io_utils.cpp)
target_link_libraries(write_rosbag ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES} )

add_executable(sub_write_vicon src/sub_write_vicon.cpp src/io_utils.cpp)
target_link_libraries(sub_write_vicon ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES} )

