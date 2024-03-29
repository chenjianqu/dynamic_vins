cmake_minimum_required(VERSION 3.0.2)
project(dynamic_vins_eval)

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
        camera_models
        )

find_package(Eigen3 REQUIRED)
include_directories(${Eigen3_INCLUDE_DIRS})


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


add_executable(pub_oxts src/pub_oxts.cpp src/oxts_parser.cpp)
target_link_libraries(pub_oxts ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(save_oxts src/save_oxts.cpp src/oxts_parser.cpp)
target_link_libraries(save_oxts ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(visualize_3d_box src/visualize_3d_box.cpp src/call_back.cpp)
target_link_libraries(visualize_3d_box ${catkin_LIBRARIES} ${OpenCV_LIBS})

#add_executable(visualize_predict src/visualize_predict_3d_boxes.cpp src/call_back.cpp src/dy_utils.cpp src/box3d.cpp)
#target_link_libraries(visualize_predict ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(convert_tracking_to_object src/convert_tracking_to_object.cpp)
target_link_libraries(convert_tracking_to_object ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(viode_generate_odometry src/viode_generate_odometry.cpp)
target_link_libraries(viode_generate_odometry ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(split_mot_to_tum src/split_mot_to_tum.cpp)
target_link_libraries(split_mot_to_tum ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(split_mot_to_single src/split_mot_to_single.cpp)
target_link_libraries(split_mot_to_single ${catkin_LIBRARIES} ${OpenCV_LIBS})


