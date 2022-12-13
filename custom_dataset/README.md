
# Custom Datasets



## Generate Dataset with ZED camera

### ZED启动

**方式一：使用ZED SDK**

* 设置分辨率**：修改`zed_ws/src/zed-ros-wrapper/zed_wrapper/params/common.yaml`中的general中的resolution参数， **`0`: HD2K, `1`: HD1080, `2`: HD720, `3`: VGA**

* launch ZED camera

```shell
roslaunch zed_wrapper zed.launch
```

* 查看图像话题

```shell
rostopic list
```

* 查看话题的分辨率

```shell
rostopic echo /zed/zed_node/stereo_raw/image_raw_color --noarr
```



**方式二：当做UVC相机使用**

```shell
rosrun pub_zed pub_zed
```





### 记录传感器信息



**录制为ROS bag**

这里录制的话题包括：

```shell
/imu/data
/imu/mag
/odom
/odometry/filtered
/scan
/zed_cam/cam0
/zed_cam/cam1
```

执行：

```shell
rosbag record /imu/data \
/imu/mag \
/odom \
/odometry/filtered \
/scan \
/zed_cam/cam0 \
/zed_cam/cam1 \
-o room.bag
```



* rviz查看录制成功后的bag

```shell
#将bag从机器人移动到本地
scp -r shansu@192.168.99.30:~/room_2022-12-08-23-05-00.bag   /home/chen/datasets/MyData/bags

rosbag play room_2022-12-7.bag

rviz -d  ${dynamic_vins_root}/src/custom_dataset/config/rviz/custom_dataset.rviz
```



### 记录VICON轨迹

* 启动Vicon的接口

```shell
roslaunch vicon_bridge vicon.launch 
```

* 记录轨迹

```shell
rosrun sub_vicon sub_vicon "room"
```



* 可视化vicon轨迹

```shell
startconda && conda activate py36

evo_traj tum vicon.txt -p
```



### 记录Gmapping轨迹

#### 建图

* 运行gampping

```shell
#在机器人上运行
roslaunch ir100_description load.launch

roslaunch ir100_navigation gmapping_demo.launch
```

* 运行rviz

```shell
#在本地主机上运行
rviz -d /home/chen/ws/robot_exp_ws/src/gmapping_demo/rviz/gmapping.rviz
```

* 保存地图

```shell
rosrun map_server map_saver -f map_426b
```



#### 导航

* 启动导航，加载之前保存的地图

```shell
roslaunch ir100_navigation ir100_navigation.launch
```

* 开始保存机器人的轨迹

```shell
rosrun tf_test write_tf_stamp \odom \base_link  gmapping.txt
```

* 查看机器人保存的轨迹

```shell
startconda && conda activate py36
evo_traj tum gmapping.txt -p

或
evo_traj tum --align  -p gmapping.txt --ref vicon.txt -s
```





### 轨迹评估

* 评估VICON和Gmapping的轨迹

```shell
pose_path=gmapping.txt
vicon_path=vicon.txt

evo_ape tum --align  ${pose_path} ${vicon_path} && evo_rpe tum --align  ${pose_path} ${vicon_path} -r trans_part && evo_rpe tum --align  ${pose_path} ${vicon_path} -r rot_part
```





### 录制步骤总结

* 设备启动：机器人、本地PC、Vicon，且均连接到机器人的局域网下

* [PC]启动Vicon的接口

```
roslaunch vicon_bridge vicon.launch 
```

* [R]启动GMapping定位
* [R]启动ZED相机

```shell
rosrun pub_zed pub_zed
```

* [PC]同时录制机器人轨迹和vicon的轨迹

```shell
#roslaunch sub_vicon record.launch

rosrun sub_vicon sub_vicon "room" true

rosrun tf_test write_tf_stamp \odom \base_link  gmapping.txt
```

* [R]录制传感器信息

```shell
rosbag record /imu/data \
/imu/mag \
/odom \
/odometry/filtered \
/scan \
/zed_cam/cam0 \
/zed_cam/cam1 \
-o room.bag
```

上面的PC表示本地主机，R表示机器人。



### 传感器标定

自定义



## Offline process Zed dataset

### Convert bag to images

**将图像话题保存为图像文件**

* 修改`custom_dataset`中的`sub_write_images.cpp`

* 编译`custom_dataset`

* 记录bag中左相机的图像

```shell
source  ${dynamic_vins_root}/devel/setup.bash  

rosrun custom_dataset sub_write_images /home/chen/datasets/MyData/ZED_data/corridor_dynamic_1/cam0
```

* 播放数据集

```shell
rosbag play -r 0.5 ./room_dynamic_1/data.bag
```



### Instance Segmentation

* 进行实例分割

```shell
python solov2_det2d_zed.py
```

















 





