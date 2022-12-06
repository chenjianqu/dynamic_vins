
# Custom Datasets



## Generate Dataset with ZED camera

### ZED camera

**设置分辨率**：修改`zed_ws/src/zed-ros-wrapper/zed_wrapper/params/common.yaml`中的general中的resolution参数， **`0`: HD2K, `1`: HD1080, `2`: HD720, `3`: VGA**



* launch ZED camera

```shell
roslaunch zed_wrapper zed.launch
```

* 查看图像话题

```shell
rostopic list
```

> /clicked_point
> /diagnostics
> /rosout
> /rosout_agg
> /tf
> /tf_static
> /zed/joint_states
> /zed/zed_node/confidence/confidence_map
> /zed/zed_node/depth/camera_info
> /zed/zed_node/depth/depth_registered
> /zed/zed_node/depth/depth_registered/compressed
> /zed/zed_node/depth/depth_registered/compressed/parameter_descriptions
> /zed/zed_node/depth/depth_registered/compressed/parameter_updates
> /zed/zed_node/depth/depth_registered/compressedDepth
> /zed/zed_node/depth/depth_registered/compressedDepth/parameter_descriptions
> /zed/zed_node/depth/depth_registered/compressedDepth/parameter_updates
> /zed/zed_node/depth/depth_registered/theora
> /zed/zed_node/depth/depth_registered/theora/parameter_descriptions
> /zed/zed_node/depth/depth_registered/theora/parameter_updates
> /zed/zed_node/disparity/disparity_image
> /zed/zed_node/left/camera_info
> /zed/zed_node/left/image_rect_color
> /zed/zed_node/left/image_rect_color/compressed
> /zed/zed_node/left/image_rect_color/compressed/parameter_descriptions
> /zed/zed_node/left/image_rect_color/compressed/parameter_updates
> /zed/zed_node/left/image_rect_color/compressedDepth
> /zed/zed_node/left/image_rect_color/compressedDepth/parameter_descriptions
> /zed/zed_node/left/image_rect_color/compressedDepth/parameter_updates
> /zed/zed_node/left/image_rect_color/theora
> /zed/zed_node/left/image_rect_color/theora/parameter_descriptions
> /zed/zed_node/left/image_rect_color/theora/parameter_updates
> /zed/zed_node/left/image_rect_gray
> /zed/zed_node/left/image_rect_gray/compressed
> /zed/zed_node/left/image_rect_gray/compressed/parameter_descriptions
> /zed/zed_node/left/image_rect_gray/compressed/parameter_updates
> /zed/zed_node/left/image_rect_gray/compressedDepth
> /zed/zed_node/left/image_rect_gray/compressedDepth/parameter_descriptions
> /zed/zed_node/left/image_rect_gray/compressedDepth/parameter_updates
> /zed/zed_node/left/image_rect_gray/theora
> /zed/zed_node/left/image_rect_gray/theora/parameter_descriptions
> /zed/zed_node/left/image_rect_gray/theora/parameter_updates
> /zed/zed_node/left_raw/camera_info
> /zed/zed_node/left_raw/image_raw_color
> /zed/zed_node/left_raw/image_raw_color/compressed
> /zed/zed_node/left_raw/image_raw_color/compressed/parameter_descriptions
> /zed/zed_node/left_raw/image_raw_color/compressed/parameter_updates
> /zed/zed_node/left_raw/image_raw_color/compressedDepth
> /zed/zed_node/left_raw/image_raw_color/compressedDepth/parameter_descriptions
> /zed/zed_node/left_raw/image_raw_color/compressedDepth/parameter_updates
> /zed/zed_node/left_raw/image_raw_color/theora
> /zed/zed_node/left_raw/image_raw_color/theora/parameter_descriptions
> /zed/zed_node/left_raw/image_raw_color/theora/parameter_updates
> /zed/zed_node/left_raw/image_raw_gray
> /zed/zed_node/left_raw/image_raw_gray/compressed
> /zed/zed_node/left_raw/image_raw_gray/compressed/parameter_descriptions
> /zed/zed_node/left_raw/image_raw_gray/compressed/parameter_updates
> /zed/zed_node/left_raw/image_raw_gray/compressedDepth
> /zed/zed_node/left_raw/image_raw_gray/compressedDepth/parameter_descriptions
> /zed/zed_node/left_raw/image_raw_gray/compressedDepth/parameter_updates
> /zed/zed_node/left_raw/image_raw_gray/theora
> /zed/zed_node/left_raw/image_raw_gray/theora/parameter_descriptions
> /zed/zed_node/left_raw/image_raw_gray/theora/parameter_updates
> /zed/zed_node/odom
> /zed/zed_node/parameter_descriptions
> /zed/zed_node/parameter_updates
> /zed/zed_node/path_map
> /zed/zed_node/path_odom
> /zed/zed_node/plane
> /zed/zed_node/plane_marker
> /zed/zed_node/point_cloud/cloud_registered
> /zed/zed_node/pose
> /zed/zed_node/pose_with_covariance
> /zed/zed_node/rgb/camera_info
> /zed/zed_node/rgb/image_rect_color
> /zed/zed_node/rgb/image_rect_color/compressed
> /zed/zed_node/rgb/image_rect_color/compressed/parameter_descriptions
> /zed/zed_node/rgb/image_rect_color/compressed/parameter_updates
> /zed/zed_node/rgb/image_rect_color/compressedDepth
> /zed/zed_node/rgb/image_rect_color/compressedDepth/parameter_descriptions
> /zed/zed_node/rgb/image_rect_color/compressedDepth/parameter_updates
> /zed/zed_node/rgb/image_rect_color/theora
> /zed/zed_node/rgb/image_rect_color/theora/parameter_descriptions
> /zed/zed_node/rgb/image_rect_color/theora/parameter_updates
> /zed/zed_node/rgb/image_rect_gray
> /zed/zed_node/rgb/image_rect_gray/compressed
> /zed/zed_node/rgb/image_rect_gray/compressed/parameter_descriptions
> /zed/zed_node/rgb/image_rect_gray/compressed/parameter_updates
> /zed/zed_node/rgb/image_rect_gray/compressedDepth
> /zed/zed_node/rgb/image_rect_gray/compressedDepth/parameter_descriptions
> /zed/zed_node/rgb/image_rect_gray/compressedDepth/parameter_updates
> /zed/zed_node/rgb/image_rect_gray/theora
> /zed/zed_node/rgb/image_rect_gray/theora/parameter_descriptions
> /zed/zed_node/rgb/image_rect_gray/theora/parameter_updates
> /zed/zed_node/rgb_raw/camera_info
> /zed/zed_node/rgb_raw/image_raw_color
> /zed/zed_node/rgb_raw/image_raw_color/compressed
> /zed/zed_node/rgb_raw/image_raw_color/compressed/parameter_descriptions
> /zed/zed_node/rgb_raw/image_raw_color/compressed/parameter_updates
> /zed/zed_node/rgb_raw/image_raw_color/compressedDepth
> /zed/zed_node/rgb_raw/image_raw_color/compressedDepth/parameter_descriptions
> /zed/zed_node/rgb_raw/image_raw_color/compressedDepth/parameter_updates
> /zed/zed_node/rgb_raw/image_raw_color/theora
> /zed/zed_node/rgb_raw/image_raw_color/theora/parameter_descriptions
> /zed/zed_node/rgb_raw/image_raw_color/theora/parameter_updates
> /zed/zed_node/rgb_raw/image_raw_gray
> /zed/zed_node/rgb_raw/image_raw_gray/compressed
> /zed/zed_node/rgb_raw/image_raw_gray/compressed/parameter_descriptions
> /zed/zed_node/rgb_raw/image_raw_gray/compressed/parameter_updates
> /zed/zed_node/rgb_raw/image_raw_gray/compressedDepth
> /zed/zed_node/rgb_raw/image_raw_gray/compressedDepth/parameter_descriptions
> /zed/zed_node/rgb_raw/image_raw_gray/compressedDepth/parameter_updates
> /zed/zed_node/rgb_raw/image_raw_gray/theora
> /zed/zed_node/rgb_raw/image_raw_gray/theora/parameter_descriptions
> /zed/zed_node/rgb_raw/image_raw_gray/theora/parameter_updates
> /zed/zed_node/right/camera_info
> /zed/zed_node/right/image_rect_color
> /zed/zed_node/right/image_rect_color/compressed
> /zed/zed_node/right/image_rect_color/compressed/parameter_descriptions
> /zed/zed_node/right/image_rect_color/compressed/parameter_updates
> /zed/zed_node/right/image_rect_color/compressedDepth
> /zed/zed_node/right/image_rect_color/compressedDepth/parameter_descriptions
> /zed/zed_node/right/image_rect_color/compressedDepth/parameter_updates
> /zed/zed_node/right/image_rect_color/theora
> /zed/zed_node/right/image_rect_color/theora/parameter_descriptions
> /zed/zed_node/right/image_rect_color/theora/parameter_updates
> /zed/zed_node/right/image_rect_gray
> /zed/zed_node/right/image_rect_gray/compressed
> /zed/zed_node/right/image_rect_gray/compressed/parameter_descriptions
> /zed/zed_node/right/image_rect_gray/compressed/parameter_updates
> /zed/zed_node/right/image_rect_gray/compressedDepth
> /zed/zed_node/right/image_rect_gray/compressedDepth/parameter_descriptions
> /zed/zed_node/right/image_rect_gray/compressedDepth/parameter_updates
> /zed/zed_node/right/image_rect_gray/theora
> /zed/zed_node/right/image_rect_gray/theora/parameter_descriptions
> /zed/zed_node/right/image_rect_gray/theora/parameter_updates
> /zed/zed_node/right_raw/camera_info
> /zed/zed_node/right_raw/image_raw_color
> /zed/zed_node/right_raw/image_raw_color/compressed
> /zed/zed_node/right_raw/image_raw_color/compressed/parameter_descriptions
> /zed/zed_node/right_raw/image_raw_color/compressed/parameter_updates
> /zed/zed_node/right_raw/image_raw_color/compressedDepth
> /zed/zed_node/right_raw/image_raw_color/compressedDepth/parameter_descriptions
> /zed/zed_node/right_raw/image_raw_color/compressedDepth/parameter_updates
> /zed/zed_node/right_raw/image_raw_color/theora
> /zed/zed_node/right_raw/image_raw_color/theora/parameter_descriptions
> /zed/zed_node/right_raw/image_raw_color/theora/parameter_updates
> /zed/zed_node/right_raw/image_raw_gray
> /zed/zed_node/right_raw/image_raw_gray/compressed
> /zed/zed_node/right_raw/image_raw_gray/compressed/parameter_descriptions
> /zed/zed_node/right_raw/image_raw_gray/compressed/parameter_updates
> /zed/zed_node/right_raw/image_raw_gray/compressedDepth
> /zed/zed_node/right_raw/image_raw_gray/compressedDepth/parameter_descriptions
> /zed/zed_node/right_raw/image_raw_gray/compressedDepth/parameter_updates
> /zed/zed_node/right_raw/image_raw_gray/theora
> /zed/zed_node/right_raw/image_raw_gray/theora/parameter_descriptions
> /zed/zed_node/right_raw/image_raw_gray/theora/parameter_updates
> /zed/zed_node/stereo/image_rect_color
> /zed/zed_node/stereo/image_rect_color/compressed
> /zed/zed_node/stereo/image_rect_color/compressed/parameter_descriptions
> /zed/zed_node/stereo/image_rect_color/compressed/parameter_updates
> /zed/zed_node/stereo/image_rect_color/compressedDepth
> /zed/zed_node/stereo/image_rect_color/compressedDepth/parameter_descriptions
> /zed/zed_node/stereo/image_rect_color/compressedDepth/parameter_updates
> /zed/zed_node/stereo/image_rect_color/theora
> /zed/zed_node/stereo/image_rect_color/theora/parameter_descriptions
> /zed/zed_node/stereo/image_rect_color/theora/parameter_updates
> /zed/zed_node/stereo_raw/image_raw_color
> /zed/zed_node/stereo_raw/image_raw_color/compressed
> /zed/zed_node/stereo_raw/image_raw_color/compressed/parameter_descriptions
> /zed/zed_node/stereo_raw/image_raw_color/compressed/parameter_updates
> /zed/zed_node/stereo_raw/image_raw_color/compressedDepth
> /zed/zed_node/stereo_raw/image_raw_color/compressedDepth/parameter_descriptions
> /zed/zed_node/stereo_raw/image_raw_color/compressedDepth/parameter_updates
> /zed/zed_node/stereo_raw/image_raw_color/theora
> /zed/zed_node/stereo_raw/image_raw_color/theora/parameter_descriptions
> /zed/zed_node/stereo_raw/image_raw_color/theora/parameter_updates



* 查看话题的分辨率

```shell
rostopic echo /zed/zed_node/stereo_raw/image_raw_color --noarr
```







### Record Image with ZED driver

* 修改`custom_dataset`中的`sub_write_images.cpp`



* 编译`custom_dataset`



* 录制图像

```shell
source  ${dynamic_vins_root}/devel/setup.bash && rosrun custom_dataset sub_write_images room
```



### Calib Camera

* record sevaral images

```shell
source  ${dynamic_vins_root}/devel/setup.bash && rosrun custom_dataset capture_single_image
```

* run calib

```shell
source  ${dynamic_vins_root}/devel/setup.bash && rosrun custom_dataset calib_camera
```

* (Optional) use matlab to calibrate camera



最终得到ZED相机的左右内参为：

```shell
size:[720 1280]

cam0
fx: 684.4439
fy: 683.1740
cx: 662.8685
cy: 365.4014
k1: -0.1822
k2: 0.0308
p1: 0.0
p2: 0.0

cam1
fx: 687.8644
fy: 687.6804
cx: 677.6340
cy: 380.4417
k1: -0.1691
k2: 0.0235
p1: 0.0
p2: 0.0

```



* 测试标定得到的内参





 





