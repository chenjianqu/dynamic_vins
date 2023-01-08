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
cd ./dynamic_vins/src/thirdparty/line_descriptor
mkdir build && cd build
cmake ..
make
```

**note**: if you changed OpenCV version, please recompile the line_descriptor.



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



## Features

### Support datasets

* Kitti tracking

* VIODE
* EuRoC
* Custom dataset

see [custom_dataset](custom_dataset/README.md)



### Support mode

**sensor mode**:

* Vison-only
* VIO

**line-point mode**:

* PointOnly
* LinePoint

**dynamic mode**:

* raw
* naive
* dynamic




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
#config_file=/home/chen/ws/dynamic_ws/src/dynamic_vins/config/kitti/kitti_tracking/kitti_tracking.yaml
#config_file=/home/chen/ws/dynamic_ws/src/dynamic_vins/config/kitti/kitti_tracking/kitti_tracking_raw_line.yaml

#config_file=/home/chen/ws/dynamic_ws/src/dynamic_vins/config/euroc/euroc.yaml

#config_file=/home/chen/ws/dynamic_ws/src/dynamic_vins/config/custom/stereo_1920x1080/custom.yaml

config_file=${dynamic_vins_root}/src/dynamic_vins/config/custom/zed_1280x720_vision_only/custom.yaml
#config_file=${dynamic_vins_root}/src/dynamic_vins/config/custom/zed_1280x720/custom.yaml

#config_file=${dynamic_vins_root}/src/dynamic_vins/config/custom/mynteye/custom.yaml
#config_file=${dynamic_vins_root}/src/dynamic_vins/config/custom/mynteye_vison_only/custom.yaml
 
source  ${dynamic_vins_root}/devel/setup.bash && rosrun dynamic_vins dynamic_vins ${config_file} ${seq_name} 0000
```



* play dataset

```shell
#viode
rosbag play /home/chen/datasets/VIODE/bag/city_night/0_none.bag

#custom
rosbag play corridor_dynamic_1/data.bag -r 0.5
```



* Shutdown DynamicVINS with ROS

```shell
rostopic pub -1 /vins_terminal std_msgs/Bool -- '1'
```







## Visualization

see [visualization.md](./dynamic_vins/docs/visualization.md) .



## Evaluation

see [evaluate.md](./dynamic_vins/docs/evaluate.md) .



















