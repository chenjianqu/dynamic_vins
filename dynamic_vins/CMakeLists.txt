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
    image_transport
    camera_models
	)

include_directories(${catkin_INCLUDE_DIRS})

find_package(spdlog REQUIRED)
include_directories(${spdlog_INCLUDE_DIRS})
message("spdlog version:${spdlog_VERSION}")
message("spdlog dir:${spdlog_DIR}")

#set(OpenCV_DIR /home/chen/app/opencv-3.4.16/build)
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
        /usr/lib/x86_64-linux-gnu/libpcl_filters.so
        /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
        /usr/lib/x86_64-linux-gnu/libpcl_search.so
		/usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
		/usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
		/usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.3.so.6.3
		/usr/lib/x86_64-linux-gnu/libvtkRenderingFreeTypeFontConfig-6.3.so.6.3
		/usr/lib/x86_64-linux-gnu/libvtkRenderingMatplotlib-6.3.so.6.3
		/usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL-6.3.so.6.3
		/usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-6.3.so.6.3
		/usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-6.3.so.6.3
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
include_directories(thirdparty)
include_directories(thirdparty/line_descriptor/include)

aux_source_directory(src/utils utils_sources)
aux_source_directory(src/utils/dataset dataset_sources)
aux_source_directory(src/utils/tensorrt tensorrt_sources)
aux_source_directory(src/utils/test test_sources)
aux_source_directory(src/utils/io io_sources)
aux_source_directory(src/basic basic_sources)
aux_source_directory(src/det2d det2d_sources)
aux_source_directory(src/det3d det3d_sources)
aux_source_directory(src/mot mot_sources)
#aux_source_directory(src/flow flow_sources)
aux_source_directory(src/stereo stereo_sources)
aux_source_directory(src/image_process image_process_source)
aux_source_directory(src/front_end front_end_sources)
aux_source_directory(src/estimator/initial initial_sources)
aux_source_directory(src/estimator/factor factor_sources)
aux_source_directory(src/estimator/imu imu_sources)
aux_source_directory(src/estimator estimator_sources)
aux_source_directory(thirdparty/line_descriptor/src line_descriptor_sources)
aux_source_directory(src/line_detector line_detector_sources)

message("estimator: ${estimator} ")

add_executable(dynamic_vins src/system/main.cpp ${utils_sources} ${dataset_sources} ${tensorrt_sources} ${test_sources}
        ${det2d_sources} ${det3d_sources} ${image_process_source} ${stereo_sources} ${line_detector_sources}
		${line_descriptor_sources} ${io_sources} ${basic_sources}
        ${mot_sources}  ${front_end_sources} ${initial_sources}  ${factor_sources} ${imu_sources} ${estimator_sources})
target_link_libraries(dynamic_vins ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES}  ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES} TorchVision::TorchVision ${TensorRT_LIBS} ${CERES_LIBRARIES} ${PCL_LIB} ${Sophus_LIBRARIES}
		${PROJECT_SOURCE_DIR}/thirdparty/line_descriptor/lib/liblinedesc.so
        )

add_executable(build_raft src/system/build_tools/build_raft.cpp src/flow/flow_parameter.cpp ${tensorrt_sources}
		src/utils/log_utils.cpp)
target_link_libraries(build_raft ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES}  ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES} TorchVision::TorchVision ${TensorRT_LIBS} ${CERES_LIBRARIES} ${PCL_LIB} ${Sophus_LIBRARIES})


add_executable(build_solo src/system/build_tools/build_solo.cpp ${tensorrt_sources} src/utils/log_utils.cpp
		src/det2d/det2d_parameter.cpp)
target_link_libraries(build_solo ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES}  ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES} TorchVision::TorchVision ${TensorRT_LIBS} ${CERES_LIBRARIES} ${PCL_LIB} ${Sophus_LIBRARIES})


add_executable(pub_object3d src/system/eval/pub_object3d.cpp  ${utils_sources} ${dataset_sources} ${tensorrt_sources}
        ${det2d_sources} ${det3d_sources} ${io_sources} ${image_process_source} ${stereo_sources} ${basic_sources}
		${line_detector_sources} ${line_descriptor_sources}
		${mot_sources} ${front_end_sources} ${initial_sources}  ${factor_sources} ${estimator_sources})
target_link_libraries(pub_object3d ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES}  ${CUDA_LIBRARIES}
        ${TORCH_LIBRARIES} TorchVision::TorchVision ${TensorRT_LIBS} ${CERES_LIBRARIES} ${PCL_LIB} ${Sophus_LIBRARIES})


add_executable(pub_inst_pointcloud src/system/eval/pub_inst_pointcloud.cpp  ${utils_sources} ${dataset_sources} ${tensorrt_sources}
		${det2d_sources} ${det3d_sources} ${io_sources} ${image_process_source} ${stereo_sources} ${basic_sources}
		${line_detector_sources} ${line_descriptor_sources}
		${mot_sources} ${front_end_sources} ${initial_sources}  ${factor_sources} ${estimator_sources})
target_link_libraries(pub_inst_pointcloud ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES}  ${CUDA_LIBRARIES}
		${TORCH_LIBRARIES} TorchVision::TorchVision ${TensorRT_LIBS} ${CERES_LIBRARIES} ${PCL_LIB} ${Sophus_LIBRARIES})



add_executable(corners_detection src/system/exp_tools/corners_detections.cpp)
target_link_libraries(corners_detection ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES})


add_executable(lk_flow src/system/exp_tools/lk_flow.cpp)
target_link_libraries(lk_flow ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES})

add_executable(instance_lk_flow src/system/exp_tools/instance_lk_flow.cpp)
target_link_libraries(instance_lk_flow ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES})

aux_source_directory(src/utils/orb orb_sources)

add_executable(orb_demo src/system/exp_tools/orb_demo.cpp ${orb_sources})
target_link_libraries(orb_demo ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES})


add_executable(read_pub_pointcloud src/system/exp_tools/read_pub_pointcloud.cpp src/utils/file_utils.cpp)
target_link_libraries(read_pub_pointcloud ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES} ${PCL_LIB})

add_executable(ransac_test src/system/exp_tools/ransac_test.cpp)
target_link_libraries(ransac_test ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES} ${PCL_LIB})


add_executable(stereo_test src/system/exp_tools/stereo_test.cpp src/utils/file_utils.cpp)
target_link_libraries(stereo_test ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES} ${PCL_LIB})

add_executable(object_pointcloud_test src/system/exp_tools/object_pointcloud_test.cpp src/utils/file_utils.cpp)
target_link_libraries(object_pointcloud_test ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES} ${PCL_LIB})



add_executable(line_tracking src/system/exp_tools/line_tracking.cpp src/utils/log_utils.cpp src/basic/frame_lines.cpp
		${line_detector_sources} ${line_descriptor_sources})
target_link_libraries(line_tracking ${catkin_LIBRARIES} ${OpenCV_LIBS} ${spdlog_LIBRARIES}
		${PROJECT_SOURCE_DIR}/thirdparty/line_descriptor/lib/liblinedesc.so)

