cmake_minimum_required(VERSION 3.0.2)
project(kitti_pub)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        cv_bridge
        image_transport
)



find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV DIR:  ${OpenCV_DIR}")
message("OpenCV_INCLUDE_DIRS:  ${OpenCV_INCLUDE_DIRS}")
message("OpenCV OpenCV_VERSION:  ${OpenCV_VERSION}")


catkin_package(
        #  INCLUDE_DIRS include
        #  LIBRARIES opencv_test
        #  CATKIN_DEPENDS cv_bridge roscpp rospy
        #  DEPENDS system_lib
)

include_directories(
        # include
        ${catkin_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
)


aux_source_directory(src src_SOURCES)
message("${src_SOURCES}")


add_executable(kitti_pub src/kitti_pub.cpp)
target_link_libraries(kitti_pub ${catkin_LIBRARIES} ${OpenCV_LIBS})


