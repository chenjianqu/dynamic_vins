

# visualization



## Visualize 3D box



### Whit Image



* Visualize GT 3D box

```shell
dataset_root_path=/home/chen/datasets/kitti/tracking/

seq=0018

tracking_result=${dataset_root_path}/data_tracking_label_2/training/label_02/${seq}.txt
image_dir=${dataset_root_path}/data_tracking_image_2/training/image_02/${seq}/

rosrun dynamic_vins_eval visualize_3d_box ${tracking_result} ${image_dir}
```



* Visualize estimate 3D box

```shell
dataset_root_path=/home/chen/datasets/kitti/tracking/

seq=0018

tracking_result=${dynamic_vins_root}/src/dynamic_vins/data/output/${seq}.txt
image_dir=${dataset_root_path}/data_tracking_image_2/training/image_02/${seq}/

rosrun dynamic_vins_eval visualize_3d_box ${tracking_result} ${image_dir}
```





### With Rviz

* run rviz

```shell
rviz -d ${dynamic_vins_root}/src/custom_dataset/config/rviz/box3d.rviz
```



* run pub_object3d

```shell
config_file=${dynamic_vins_root}/src/dynamic_vins/config/kitti/kitti_tracking/kitti_tracking.yaml

source ${dynamic_vins_root}/devel/setup.bash

rosrun dynamic_vins pub_object3d ${config_file} "prediction"
```



## Visualize Instance Pointcloud

### with Rviz

* run rviz

```shell
rviz -d ${dynamic_vins_root}/src/custom_dataset/config/rviz/box3d.rviz
```



* run pub_inst_pointcloud

```shell
source  ${dynamic_vins_root}/devel/setup.bash

config_file=${dynamic_vins_root}/src/dynamic_vins/config/custom/zed_1280x720_vision_only/dynamic.yaml

rosrun dynamic_vins pub_inst_pointcloud ${config_file} road_2
```



