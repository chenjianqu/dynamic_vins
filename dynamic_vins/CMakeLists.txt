cmake_minimum_required(VERSION 2.8.3)
project(dynamic_vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)


#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -no-pie -fno-pie -fPIC")
set(CMAKE_CXX_FLAGS "-O3 -Wall -g -no-pie -fno-pie -fPIC")

message("PROJECT_SOURCE_DIR : ${PROJECT_SOURCE_DIR}")

#set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
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


include_directories(src)

aux_source_directory(src/utils utils_sources)
aux_source_directory(src/utils/dataset dataset_sources)
aux_source_directory(src/utils/tensorrt tensorrt_sources)
aux_source_directory(src/utils/io io_sources)
aux_source_directory(src/det2d det2d_sources)
aux_source_directory(src/det3d det3d_sources)
aux_source_directory(src/mot mot_sources)
aux_source_directory(src/flow flow_sources)
aux_source_directory(src/front_end front_end_sources)
aux_source_directory(src/initial initial_sources)
aux_source_directory(src/factor factor_sources)
aux_source_directory(src/estimator estimator_sources)

message("estimator: ${estimator} ")

add_executable(dynamic_vins src/system/main.cpp ${utils_sources} ${dataset_sources} ${tensorrt_sources}
        ${det2d_sources} ${det3d_sources} ${io_sources}
        ${mot_sources} ${flow_sources} ${front_end_sources} ${initial_sources}  ${factor_sources} ${estimator_sources})
target_link_libraries(dynamic_vins ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES}  ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES} TorchVision::TorchVision ${TensorRT_LIBS} ${CERES_LIBRARIES} ${PCL_LIB} ${Sophus_LIBRARIES})

add_executable(build_raft src/system/build_raft.cpp src/flow/flow_parameter.cpp ${tensorrt_sources} src/utils/log_utils.cpp)
target_link_libraries(build_raft ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES}  ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES} TorchVision::TorchVision ${TensorRT_LIBS} ${CERES_LIBRARIES} ${PCL_LIB} ${Sophus_LIBRARIES})


add_executable(build_solo src/system/build_solo.cpp ${tensorrt_sources} src/utils/log_utils.cpp src/det2d/det2d_parameter.cpp)
target_link_libraries(build_solo ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES}  ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES} TorchVision::TorchVision ${TensorRT_LIBS} ${CERES_LIBRARIES} ${PCL_LIB} ${Sophus_LIBRARIES})


add_executable(pub_object3d src/system/eval/pub_object3d.cpp src/utils/log_utils.cpp src/utils/parameters.cpp
        src/utils/box3d.cpp src/utils/io/io_parameters.cpp src/utils/io/io_utils.cpp src/utils/camera_model.cpp
        src/utils/dataset/kitti_utils.cpp src/det3d/det3d_parameter.cpp src/utils/io/build_markers.cpp
        src/utils/io/camera_pose_visualization.cpp)
target_link_libraries(pub_object3d ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES})
