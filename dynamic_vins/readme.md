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



### Run

#### ROS Prepare

* launch ros core
```shell
roscore
```

* launch rviz
```shell
rosrun rviz rviz -d ${dynamic_vins_root}/src/dynamic_vins/config/rviz/rviz.rviz
```



#### Launch dynamic_vins



```shell
#VINS
dynamic_vins_root=/home/chen/ws/dynamic_ws
config_file=${dynamic_vins_root}/src/dynamic_vins/config/viode/viode.yaml 
#config_file=${dynamic_vins_root}/src/dynamic_vins/config/kitti/kitti_09_30/kitti_09_30_config.yaml
#config_file=${dynamic_vins_root}/src/dynamic_vins/config/kitti/kitti_10_03/kitti_10_03_config.yaml
#config_file=/home/chen/ws/dynamic_ws/src/dynamic_vins/config/kitti/kitti_tracking/kitti_tracking.yaml

source  ${dynamic_vins_root}/devel/setup.bash && rosrun dynamic_vins dynamic_vins ${config_file}
```



#### Pub dataset



* play dataset

```shell
#viode
rosbag play /home/chen/datasets/VIODE/bag/city_day/0_none.bag

#kitti
rosbag play /home/chen/Datasets/kitti/odometry_color_07.bag
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



#### Others

* Shutdown DynamicVINS with ROS

```shell
rostopic pub -1 /vins_terminal std_msgs/Bool -- '1'
```







## Visualization

### Visualize 3D box

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









## Evaluate



### Evaluate ego-motion with EVO

#### Install EVO

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

sudo apt install python-pip 
pip install evo --upgrade --no-binary evo 
```





**Prepare VIODE ground-truth trajectory**

```shell
cd dynamic_ws
source devel/setup.bash

save_dir=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/viode_egomotion/
save_name=city_night_3_high

rosrun dynamic_vins_eval viode_generate_odometry /odometry ${save_dir}/${save_name}.txt

#########################################

viode_dir=/home/chen/datasets/VIODE/bag/
sequence=city_night/3_high

rosbag play ${viode_dir}/${sequence}.bag
```





#### Prepare ground-truth KITTI

* Convert KITTI's oxts data to TUM format pose

```shell
oxts_path=/home/chen/datasets/kitti/tracking/data_tracking_oxts/training/oxts/
save_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/
sequence=0018

source ~/ws/dynamic_ws/devel/setup.bash
rosrun dynamic_vins_eval save_oxts ${oxts_path}/${sequence}.txt ${save_path}/${sequence}.txt
```



* To visualize the ground-truth trajectory, you run:

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

save_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/

sequence=0003

evo_traj tum ${save_path}/${sequence}.txt -p
```



#### Evaluate ego-motion



* To visualize the estimate ego-motion,run:

```shell
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-6-7/kitti_tracking_dynamic/
sequence=0000

evo_traj tum ${estimate_path}/${sequence}_ego-motion.txt -p
```

or
```shell
gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-6-7/kitti_tracking_dynamic/

evo_traj tum ${estimate_path}/${sequence}_ego-motion.txt --ref=${gt_path}/${sequence}.txt --align -p
```



* To evaluate the ATE of estimate trajectory,run

```shell
gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-6-7/kitti_tracking_dynamic/
sequence="0000"

#evo_ape tum --align -p ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt
evo_ape tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt
```



* To evaluate the RPE_t of estimate trajectory,run

```shell
gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-6-7/kitti_tracking_raw/
sequence="0000"

evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r trans_part
```



* To evaluate the RPE_R of estimate trajectory,run

```shell
gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-6-7/kitti_tracking_raw/
sequence="0000"

evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r rot_part
```



* Evaluate 3 metrics simultaneously

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-6-7/kitti_tracking_naive/
sequence=0000

evo_ape tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt && evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r trans_part && evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r rot_part

```

或

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/viode_egomotion/
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-6-7/viode_raw/

sequence=city_day_0_none

evo_ape tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt && evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r trans_part && evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r rot_part

```



### Evaluate object trajectory with EVO

#### Prepare

* Split object tracking label to `TUM` format

```shell
gt_id=3
sequence=0018

gt_file=/home/chen/datasets/kitti/tracking/data_tracking_label_2/training/label_02/${sequence}.txt

cam_pose_file=${dynamic_vins_root}/src/dynamic_vins/data/output/${sequence}_ego-motion.txt

save_to_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_tum/${sequence}_${gt_id}.txt

rosrun dynamic_vins_eval split_mot_to_tum ${gt_id} ${gt_file} ${cam_pose_file} ${save_to_path}
```



* Visualize the **estimate** object trajectory and **ground-truth** trajectory

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

object_id=8
gt_id=2
sequence=0018

ref_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_tum/${sequence}_${gt_id}.txt

estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/output/${sequence}_tum/${object_id}.txt

evo_traj tum ${estimate_path} --ref=${ref_path} -p
```





#### Estimate object trajectory



```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

object_id=9
gt_id=3
sequence=0018

estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/output/${sequence}_tum/${object_id}.txt

ref_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_tum/${sequence}_${gt_id}.txt

evo_ape tum --align  ${estimate_path} ${ref_path}  && evo_rpe tum --align  ${estimate_path} ${ref_path} -r trans_part && evo_rpe tum --align  ${estimate_path} ${ref_path} -r rot_part
```





### Evaluate mot with KITTI devkit_tracking



#### Install devkit_tracking

Download the devkit： [The KITTI Vision Benchmark Suite (cvlibs.net)](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) .

Run environment: `Python2`, dependency:

* pip install munkres



#### Prepare

split the single object from GT and estimation



```shell
# split the single object from **ground-truth** labels:

object_id=9
gt_id=3
sequence=0018

gt_file=/home/chen/datasets/kitti/tracking/data_tracking_label_2/training/label_02/${sequence}.txt

save_to_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_single/${sequence}_${gt_id}.txt

rosrun dynamic_vins_eval split_mot_to_single ${gt_id} ${gt_file} ${save_to_path}


#split the single object from **estimate results**:

estimate_file=${dynamic_vins_root}/src/dynamic_vins/data/output/${sequence}.txt

save_to_path=${dynamic_vins_root}/src/dynamic_vins/data/output/kitti_tracking_single/${sequence}_${object_id}.txt

rosrun dynamic_vins_eval split_mot_to_single ${object_id} ${estimate_file} ${save_to_path}

```



把评估的数据放到适当的位置：

```shell
save_to_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_single/${sequence}_${gt_id}.txt

cp_to_path=${dynamic_vins_root}/src/dynamic_vins_eval/eval_tools/devkit_tracking/devkit/python/data/tracking/label_02/${sequence}.txt

cp ${save_to_path} ${cp_to_path}

#将预测结果放到适当的位置，并改名：
save_to_path=${dynamic_vins_root}/src/dynamic_vins/data/output/kitti_tracking_single/${sequence}_${object_id}.txt

cp_to_path=${dynamic_vins_root}/src/dynamic_vins_eval/eval_tools/devkit_tracking/devkit/python/results/dynamic_vins/data/${sequence}.txt

cp ${save_to_path} ${cp_to_path}

```



修改.segmap文件：

```shell
#修改.segmap文件来设置要评估的序列

echo "${sequence} empty 000000 000339" > ${dynamic_vins_root}/src/dynamic_vins_eval/eval_tools/devkit_tracking/devkit/python/data/tracking/evaluate_tracking.seqmap

填入：序列对应的帧区间：
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







#### Evaluate



​		执行评估:

```shell
cd ${dynamic_vins_root}/src/dynamic_vins_eval/eval_tools/devkit_tracking/devkit/python

python ./evaluate_tracking.py dynamic_vins
```

​	这是一个python脚本，其中dynamic_vins是`./result`下保存模型输出结果的地方。

​	需要注意的是，该kit只评估Car（van）类别和Pedestrian类别的目标，因此输出文件中应该只包含这两个类别的目标跟踪结果，且不应该包含Don't care类别。输出结果中的每行中的`truncated`和`occluded` 项设置为默认值-1.

​	为了设置评估的序列，在./data/tracking/下面的`evaluate_tracking.seqmap`设置要评估的序列。若想更改映射的文件，可在`./evaluate_tracking.py`中的代码 `filename_test_mapping = "./data/tracking/my_test.seqmap"` 设置文件名。







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

cp ${dynamic_vins_root}/src/dynamic_vins/data/output/0002_object.txt ${TrackEval}/data/trackers/kitti/${kitti_mode}/dynamic_vins/data

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
```



Eval:

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate track_evel

TrackEval=/home/chen/PycharmProjects/MOT_Eval/TrackEval-master

python  ${TrackEval}/scripts/run_kitti_mots.py --GT_FOLDER  ${TrackEval}/data/gt/kitti/kitti_mots_val   --TRACKERS_FOLDER  ${TrackEval}/data/trackers/kitti/kitti_mots_val --CLASSES_TO_EVAL car --METRICS CLEAR --TRACKERS_TO_EVAL dynamic_vins

或 根据2D包围框评估 
python  ${TrackEval}/scripts/run_kitti.py --GT_FOLDER  ${TrackEval}/data/gt/kitti/kitti_2d_box_train   --TRACKERS_FOLDER  ${TrackEval}/data/trackers/kitti/kitti_2d_box_train --CLASSES_TO_EVAL car --METRICS CLEAR --TRACKERS_TO_EVAL dynamic_vins
```

the output path is `${TrackEval}/data/trackers/kitti/kitti_mots_val `.







### Evaluate mot with KITTI devkit_object

#### Intall devkit_object





#### Prepare ground-truth

由于devkit_object使用的标签格式与KITTI Tracking的格式不同,因此通过下面的程序将`tracking_label_path`标签文件转换为kitti object数据集的格式文件.

```shell
rosrun dynamic_vins_eval convert_tracking_to_object tracking_label_path save_object_label_path

such as:
rosrun dynamic_vins_eval convert_tracking_to_object /home/chen/datasets/kitti/tracking/data_tracking_label_2/training/label_02/0003.txt ${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_object/0003
```





#### Evaluate

```shell
label=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_object/0003

./evaluate_object ${label} results
```

















