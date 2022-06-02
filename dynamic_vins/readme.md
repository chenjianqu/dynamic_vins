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

Install `dynamic_vins` from github:

```shell
mkdir dynamic_ws/src
cd dynamic_ws/src

git clone https://github.com/chenjianqu/dynamic_vins.git

cd ..

catkin_make -j4
```



* use clion

```shell
source devel/setup.bash

#launch clion
sh {CLion Path}/bin/clion.sh

#after catkin_make
/home/chen/app/clion-2021.2/bin/cmake/linux/bin/cmake --build /home/chen/ws/dynamic_ws/src/dynamic_vins/cmake-build-debug --target dynamic_vins -- -j4
```






## Demo



### Run

* launch ros core:
```shell
roscore
```

* launch rviz
```shell
rosrun rviz rviz -d {dynamic_vins_dir}/config/rviz/rviz.rviz
# such as:
rosrun rviz rviz -d /home/chen/ws/dynamic_ws/src/dynamic_vins/config/rviz/rviz.rviz
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
/home/chen/ws/dynamic_ws/src/dynamic_vins/config/viode/viode.yaml

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
rosrun kitti_pub kitti_pub left_image_path right_image_path [time_delta [is_show]] 
```

such as:

```shell
seq="0003"
rosrun kitti_pub kitti_pub /home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/${seq}  /home/chen/datasets/kitti/tracking/data_tracking_image_3/training/image_03/${seq} 100 1
```



## Visualization

### Visualize output 3D box

```shell
seq="0003"
rosrun dynamic_vins_eval visualize_box  /home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/${seq}_object.txt  /home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/${seq}/
```







## Evaluate



### Evaluate ego-motion with EVO

#### Install EVO

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate

sudo apt install python-pip 
pip install evo --upgrade --no-binary evo 
```





#### Prepare ground-truth KITTI

* Convert KITTI's oxts data to 'TUM' pose

```shell
source ~/ws/dynamic_ws/devel/setup.bash

oxts_path="/home/chen/datasets/kitti/tracking/data_tracking_oxts/training/oxts/"
save_path="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/ground_truth/kitti_tracking/"
sequence="0001" && rosrun dynamic_vins_eval save_oxts ${oxts_path}/${sequence}.txt ${save_path}/${sequence}.txt
```



* To visualize the ground-truth trajectory, you run:

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

save_path="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/ground_truth/kitti_tracking/"
sequence="0003"

evo_traj tum ${save_path}/${sequence}.txt -p
```



#### Evaluate ego-motion



* To visualize the estimate ego-motion,run:

```shell
estimate_path="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/"
sequence="0003"

evo_traj tum ${estimate_path}/${sequence}_ego-motion.txt -p
```

or
```shell
gt_path="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/ground_truth/kitti_tracking/"
estimate_path="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/"
sequence="0002"
evo_traj tum ${estimate_path}/${sequence}_ego-motion.txt --ref=${gt_path}/${sequence}.txt --align -p
```



* To evaluate the estimate trajectory,run

```shell
gt_path="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/ground_truth/kitti_tracking/"
estimate_path="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/"
sequence="0003"
evo_ape tum --align -p ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt 
```



### Evaluate mot with TrackEval

#### Install TrackEval

Here we use TrackEval to evaluate our mot estimation result.

```shell
git clone https://github.com/JonathonLuiten/TrackEval.git

export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda create -n track_evel python=3.8

conda activate track_evel

pip3 install -r requirements.txt
pip3 install numpy=1.20.1 #requirements中的numpy版本会报错
```

Download the demo ,https://omnomnom.vision.rwth-aachen.de/data/TrackEval/data.zip

Unzip in TrackEval's root path, then run the demo:

```shell
python ./scripts/run_kitti.py

python ./scripts/run_kitti_mots.py
```



#### Evaluate mots

Put our mot result to TrackEval's path:

```shell
TrackEval=/home/chen/PycharmProjects/TrackEval-master
kitti_mode=kitti_mots_val

mkdir ${TrackEval}/data/trackers/${kitti_mode}/dynamic_vins/data

cp /home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/0002_object.txt ${TrackEval}/data/trackers/kitti/${kitti_mode}/dynamic_vins/data

mv ${TrackEval}/data/trackers/kitti/${kitti_mode}/dynamic_vins/data/0002_object.txt ${TrackEval}/data/trackers/kitti/${kitti_mode}/dynamic_vins/data/0002.txt
```



set the eval seq:

```shell
TrackEval=/home/chen/PycharmProjects/TrackEval-master
gedit ${TrackEval}/data/gt/kitti/kitti_mots_val/evaluate_mots.seqmap.val

#set evel seq:

0002 empty 000000 000233
0006 empty 000000 000270
0007 empty 000000 000800
0008 empty 000000 000390
0010 empty 000000 000294
0013 empty 000000 000340
0014 empty 000000 000106
0016 empty 000000 000209
0018 empty 000000 000339

#or training:
0000 empty 000000 000154
0001 empty 000000 000447
0002 empty 000000 000233
0003 empty 000000 000144
0004 empty 000000 000314
0005 empty 000000 000297
0006 empty 000000 000270
0007 empty 000000 000800
0008 empty 000000 000390
0009 empty 000000 000803
0010 empty 000000 000294
0011 empty 000000 000373
0012 empty 000000 000078
0013 empty 000000 000340
0014 empty 000000 000106
0015 empty 000000 000376
0016 empty 000000 000209
0017 empty 000000 000145
0018 empty 000000 000339
0019 empty 000000 001059
0020 empty 000000 000837
```



Eval:

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate track_evel


TrackEval=/home/chen/PycharmProjects/TrackEval-master

python  ${TrackEval}/scripts/run_kitti_mots.py --GT_FOLDER  ${TrackEval}/data/gt/kitti/kitti_mots_val   --TRACKERS_FOLDER  ${TrackEval}/data/trackers/kitti/kitti_mots_val --CLASSES_TO_EVAL car --METRICS CLEAR --TRACKERS_TO_EVAL dynamic_vins

或
python  ${TrackEval}/scripts/run_kitti.py --GT_FOLDER  ${TrackEval}/data/gt/kitti/kitti_2d_box_train   --TRACKERS_FOLDER  ${TrackEval}/data/trackers/kitti/kitti_2d_box_train --CLASSES_TO_EVAL car --METRICS CLEAR --TRACKERS_TO_EVAL dynamic_vins
```

the output path is `${TrackEval}/data/trackers/kitti/kitti_mots_val `.





### Evaluate mot with KITTI devkit_tracking

















