# Dynamic VINS









## 依赖

* CMake >=3.12  这是为了兼容torch-vision的编译
* gcc >= 9, 为了使用C++17的新特性。注意在安装cuda、tensorrt的时候，需要低版本(比如gcc7)来编译安装。
* CUDA10.2
* cudnn>=8
* TensorRT = 8.0.1.6
* Eigen>=3.3
* opencv3.4.16 with cuda
* spdlog
* ceres1.14.0
* pcl-1.8.1
* Sophus
* Libtorch1.8 + TorchVision0.9.1



## Install

clone project:

```shell
mkdir dynamic_ws/src
cd dynamic_ws/src

git clone https://github.com/chenjianqu/dynamic_vins.git
```



build line_descriptor:

```shell
cd ./dynamic_vins/src/utils/line_detector/line_descriptor
mkdir build && cd build
cmake ..
make
```



build `dynamic_vins` from github:

```shell
cd ..

catkin_make -j4
```

​	Then set project path:

```shell
echo "dynamic_vins_root=/home/chen/ws/dynamic_ws" >> ~/.bashrc

source ~/.bashrc
```





* use clion

```shell
source devel/setup.bash

clion_Path=/home/chen/app/clion-2021.2

#launch clion
sh ${clion_Path}/bin/clion.sh

#after catkin_make
${clion_Path}/bin/cmake/linux/bin/cmake --build ${dynamic_vins_root}/src/dynamic_vins/cmake-build-debug --target dynamic_vins -- -j4
```






## Demo

* launch ros core
```shell
roscore
```

* launch rviz
```shell
rosrun rviz rviz -d ${dynamic_vins_root}/src/dynamic_vins/config/rviz/rviz.rviz
```



* launch dynamic_vins

```shell
#config_file=${dynamic_vins_root}/src/dynamic_vins/config/viode/viode.yaml 
#config_file=${dynamic_vins_root}/src/dynamic_vins/config/kitti/kitti_09_30/kitti_09_30_config.yaml
#config_file=${dynamic_vins_root}/src/dynamic_vins/config/kitti/kitti_10_03/kitti_10_03_config.yaml
config_file=/home/chen/ws/dynamic_ws/src/dynamic_vins/config/kitti/kitti_tracking/kitti_tracking.yaml
#config_file=/home/chen/ws/dynamic_ws/src/dynamic_vins/config/kitti/kitti_tracking/kitti_tracking_raw_line.yaml
#config_file=/home/chen/ws/dynamic_ws/src/dynamic_vins/config/euroc2/euroc.yaml

source  ${dynamic_vins_root}/devel/setup.bash && rosrun dynamic_vins dynamic_vins ${config_file}
```



* play dataset

```shell
#viode
rosbag play /home/chen/datasets/VIODE/bag/city_night/0_none.bag

#kitti
rosbag play /home/chen/Datasets/kitti/odometry_color_07.bag

#euroc
rosbag play /home/chen/datasets/Euroc/MH_01_easy.bag
```



* or read dataset directly

```shell
rosrun kitti_pub kitti_pub left_image_path right_image_path [time_delta [is_show]] 
```

​	such as:

```shell
seq=0003
rosrun kitti_pub kitti_pub /home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/${seq}  /home/chen/datasets/kitti/tracking/data_tracking_image_3/training/image_03/${seq} 100 1
```



* Shutdown DynamicVINS with ROS

```shell
rostopic pub -1 /vins_terminal std_msgs/Bool -- '1'
```







## Visualization

see [visualization.md](./dynamic_vins/docs/visualization.md) .



## Evaluation

see [evaluate.md](./dynamic_vins/docs/evaluate.md) .



















