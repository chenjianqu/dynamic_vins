# Dynamic VINS

## 依赖

**CMake >=3.12**  

这是为了兼容torch-vision的编译

**gcc >= 9**

为了使用C++17的新特性。注意在安装cuda、tensorrt的时候，需要低版本(比如gcc7)来编译安装。

**Cuda10.2**

**cudnn>=8**

**TensorRT = 8.0.1.6.Linux.x86_64-gnu.cuda-10.2.cudnn8.2**

**Eigen>=3.3**

**opencv3.4.16 cuda**

需要启用cuda编译opencv

**spdlog**

**ceres1.14.0**

**pcl-1.8.1**

**Sophus**

**Libtorch1.8 + TorchVision0.9.1**


## Demo
### Compile

* use catkin
```shell
cd dynamic_ws
catkin_make -j10
```

* use clion
```shell
#after catkin_make
/home/chen/app/clion-2021.2/bin/cmake/linux/bin/cmake --build /home/chen/ws/dynamic_ws/src/dynamic_vins/cmake-build-debug --target dynamic_vins -- -j10
```

### Run

* launch ros core:
```shell
roscore
```

* launch rviz
```shell
rosrun rviz rviz -d (dynamic_vins_dir)/config/rviz.rviz
# such as:
rosrun rviz rviz -d /home/chen/ws/dynamic_ws/src/dynamic_vins/config/rviz.rviz
```

* launch  
loop_fusion
```shell
#loop_fusion
source ~/ws/vio_ws/devel/setup.bash && \ 
rosrun loop_fusion loop_fusion_node ~/ws/vio_ws/src/dynamic_vins/config/viode/calibration.yaml 

```

dynamic_vins
```shell
#VINS
source ~/ws/vio_ws/devel/setup.bash && \ 
rosrun dynamic_vins dynamic_vins ~/ws/vio_ws/src/dynamic_vins/config/viode/calibration.yaml

kitti参数
/home/chen/ws/dynamic_ws/src/dynamic_vins/config/kitti/kitti_09_30/kitti_09_30_config.yaml
/home/chen/ws/dynamic_ws/src/dynamic_vins/config/kitti/kitti_10_03/kitti_10_03_config.yaml
/home/chen/ws/dynamic_ws/src/dynamic_vins/config/kitti/kitti_tracking/kitti_tracking.yaml

viode参数：
/home/chen/ws/dynamic_ws/src/dynamic_vins/config/viode/calibration.yaml

```

* play dataset
```shell
#viode
rosbag play  /media/chen/EC4A17F64A17BBF0/datasets/viode/city_day/3_high.bag
rosbag play /home/chen/Datasets/viode/3_high.bag

#kitti
rosbag play /media/chen/EC4A17F64A17BBF0/datasets/kitti/odometry/colors/odometry_color_07.bag
rosbag play /media/chen/EC4A17F64A17BBF0/datasets/kitti/odometry/colors/odometry_color_04.bag
rosbag play /home/chen/Datasets/kitti/odometry_color_07.bag
```


直接读取数据集
```shell

```


















