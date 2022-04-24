cmake_minimum_required(VERSION 2.8.3)
project(dynamic_vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)


#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -no-pie -fno-pie -fPIC")
set(CMAKE_CXX_FLAGS "-O3 -Wall -g -no-pie -fno-pie -fPIC")

message("PROJECT_SOURCE_DIR : ${PROJECT_SOURCE_DIR}")


find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport)

include_directories(${catkin_INCLUDE_DIRS})

find_package(spdlog REQUIRED)
include_directories(${spdlog_INCLUDE_DIRS})
message("spdlog version:${spdlog_VERSION}")
message("spdlog dir:${spdlog_DIR}")


find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV DIR:  ${OpenCV_DIR}")
message("OpenCV OpenCV_VERSION:  ${OpenCV_VERSION}")

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

#set(VTK_DIR usr/local/lib)
find_package(PCL 1.8 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
message("${PCL_LIBRARIES}")
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

set(PCL_LIB
        /usr/lib/x86_64-linux-gnu/libpcl_common.so
        /usr/lib/x86_64-linux-gnu/libpcl_io.so
        /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
        )


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories( ${EIGEN3_INCLUDE_DIR})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})


#Libtorch
set(CMAKE_PREFIX_PATH "/usr/local/lib;/home/chen/app/libtorch")

find_package(TorchVision REQUIRED)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

#CUDA
find_package(CUDA 10.2 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


set(TensorRT_LIBS nvinfer nvonnxparser nvinfer_plugin)
include_directories(${TensorRT_INCLUDE_DIRS})


catkin_package()

#[[

set(Basic_Sources src/parameters.cpp src/utils.cpp src/utility/ViodeUtils.cpp)
add_library(parameter ${Basic_Sources})
link_libraries(parameter ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES})


aux_source_directory(src/InstanceSegment InstanceSegmentation_SOURCES)
add_library(instance_segmentation_lib ${InstanceSegmentation_SOURCES})
link_libraries(instance_segmentation_lib ${CUDA_LIBRARIES}  ${TORCH_LIBRARIES} TorchVision::TorchVision
        ${TensorRT_LIBS} parameter)


aux_source_directory(src/InstanceTracking InstanceTracking_SOURCES)
add_library(instance_tracking_lib ${InstanceTracking_SOURCES})
link_libraries(instance_tracking_lib  ${TORCH_LIBRARIES} TorchVision::TorchVision parameter)


aux_source_directory(src/featureTracker featureTracker_SOURCES)
add_library(feature_tracker_lib ${featureTracker_SOURCES})
link_libraries(feature_tracker_lib parameter instance_segmentation_lib instance_tracking_lib)


aux_source_directory(src/initial initial_SOURCES)
aux_source_directory(src/utility utility_SOURCES)
aux_source_directory(src/factor factor_SOURCES)
aux_source_directory(src/e estimator_SOURCES)

add_library(dynamic_vins_lib ${estimator_SOURCES} ${factor_SOURCES} ${utility_SOURCES} ${initial_SOURCES})
target_link_libraries(dynamic_vins_lib  ${CERES_LIBRARIES} ${PCL_LIB} ${Sophus_LIBRARIES}  feature_tracker_lib )


add_executable(dynamic_vins src/main.cpp src/featureTracker/SegmentImage.cpp)
target_link_libraries(dynamic_vins dynamic_vins_lib)

add_executable(front_end src/front_end.cpp)
target_link_libraries(front_end parameter feature_tracker_lib)

add_executable(tracking src/tracking.cpp)
target_link_libraries(tracking parameter instance_tracking_lib)

add_executable(build_model src/build_model.cpp src/InstanceSegment/logger.cpp)
target_link_libraries(build_model  ${TensorRT_LIBS} parameter)]]

include_directories(src)

set(Basic_Sources src/parameters.cpp src/utils.cpp src/utility/viode_utils.cpp)
aux_source_directory(src/InstanceSegment InstanceSegmentation_SOURCES)
aux_source_directory(src/InstanceTracking InstanceTracking_SOURCES)
#aux_source_directory(src/FlowEstimating OpticalFlowEstimating_SOURCES)
aux_source_directory(src/TensorRT TensorRT_SOURCES)
aux_source_directory(src/featureTracker featureTracker_SOURCES)
aux_source_directory(src/initial initial_SOURCES)
aux_source_directory(src/utility utility_SOURCES)
aux_source_directory(src/factor factor_SOURCES)
aux_source_directory(src/estimator estimator_SOURCES)

message("estimator_SOURCES: ${estimator_SOURCES} ")

add_executable(dynamic_vins src/main.cpp ${Basic_Sources} ${InstanceSegmentation_SOURCES} ${InstanceTracking_SOURCES}
        ${featureTracker_SOURCES} ${initial_SOURCES} ${utility_SOURCES} ${factor_SOURCES} ${estimator_SOURCES}
        # ${OpticalFlowEstimating_SOURCES}
        ${TensorRT_SOURCES})
target_link_libraries(dynamic_vins ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES}  ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES} TorchVision::TorchVision ${TensorRT_LIBS} ${CERES_LIBRARIES} ${PCL_LIB} ${Sophus_LIBRARIES})

add_executable(build_raft src/build_raft.cpp ${TensorRT_SOURCES} src/parameters.cpp)
target_link_libraries(build_raft  ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES}  ${TORCH_LIBRARIES}
        TorchVision::TorchVision ${TensorRT_LIBS})

add_executable(build_solo src/build_solo.cpp ${TensorRT_SOURCES} src/parameters.cpp)
target_link_libraries(build_solo  ${CUDA_LIBRARIES} ${OpenCV_LIBRARIES}  ${TORCH_LIBRARIES}
        TorchVision::TorchVision ${TensorRT_LIBS})