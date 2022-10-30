


# Evaluate






## Evaluate ego-motion with EVO

### Install EVO

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

sudo apt install python-pip 
pip install evo --upgrade --no-binary evo 
```



### Evaluate Kitti-Tracking ego-motion

* Evaluate

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

estimate_dir=data/exp/2022-6-7/kitti_tracking_naive/
sequence=0004

source ${dynamic_vins_root}/src/dynamic_vins/scripts/eval_kitti_tracking_traj.sh ${estimate_dir} ${sequence} 
```



* To visualize the estimate ego-motion,run:

```shell
# visual kitti estimate trajectory
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-6-7/kitti_tracking_dynamic/
evo_traj tum ${estimate_path}/${sequence}_ego-motion.txt -p

# visual kitti gt trajectory
save_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/
evo_traj tum ${save_path}/${sequence}.txt -p
```



### Evaluate VIODE ego-motion

* Prepare VIODE ground-truth trajectory

```shell
cd dynamic_ws
source devel/setup.bash

save_dir=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/viode_egomotion/
save_name=city_night_3_high

rosrun dynamic_vins_eval viode_generate_odometry /odometry ${save_dir}/${save_name}.txt

#########################################Another window

viode_dir=/home/chen/datasets/VIODE/bag/
sequence=city_night/3_high

rosbag play ${viode_dir}/${sequence}.bag
```



* Visualize Trajectory

```shell
gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-6-7/kitti_tracking_dynamic/

evo_traj tum ${estimate_path}/${sequence}_ego-motion.txt --ref=${gt_path}/${sequence}.txt --align -p
```



* Evaluate

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/viode_egomotion/
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-6-7/viode_raw/

sequence=city_day_0_none

evo_ape tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt && evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r trans_part && evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r rot_part

```





### Evaluate EuRoc ego-motion

* Prepare EuRoc dataset ground-truth

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

evo_traj euroc --help

dataset_dir=/home/chen/datasets/Euroc
sequence=MH_01_easy

raw_gt_path=${dataset_dir}/${sequence}/mav0/state_groundtruth_estimate0/data.csv
save_gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/euroc_egomotion/${sequence}.txt

mkdir ${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/euroc_egomotion

evo_traj euroc ${raw_gt_path} --save_as_tum #得到data.tum文件

mv ./data.tum ${save_gt_path} #移动文件
```



* Visualize Trajectory

```shell
gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/euroc_egomotion
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/output

evo_traj tum ${estimate_path}/${sequence}_ego-motion.txt --ref=${gt_path}/${sequence}.txt --align -p
```



* Evaluate

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/euroc_egomotion
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/output

sequence=MH_01_easy

evo_ape tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt && evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r trans_part && evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r rot_part
```















## Evaluate object trajectory with EVO



* Split object tracking label to `TUM` format

```shell
export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

cd ${dynamic_vins_root}
source devel/setup.bash

sequence=0004
gt_id=2
object_id=1

${dynamic_vins_root}/src/dynamic_vins/scripts/eval_object_traj.sh ${sequence} ${gt_id} ${object_id}
```



* Visualize the **estimate** object trajectory and **ground-truth** trajectory

```shell
ref_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_tum/${sequence}_${gt_id}.txt
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/output/${sequence}_tum/${object_id}.txt

evo_traj tum ${estimate_path} --ref=${ref_path} -p
```





## Evaluate mot with KITTI devkit_tracking



#### Install devkit_tracking

Download the devkit： [The KITTI Vision Benchmark Suite (cvlibs.net)](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) .

Run environment: `Python2`, dependency:

* pip install munkres





#### Evaluate

​		执行评估:

```shell
sequence=0004
gt_id=2
object_id=1

source ${dynamic_vins_root}/src/dynamic_vins/scripts/eval_mot_kitti_tracking.sh ${sequence} ${gt_id} ${object_id}
```

​	这是一个python脚本，其中dynamic_vins是`./result`下保存模型输出结果的地方。

​	需要注意的是，该kit只评估Car（van）类别和Pedestrian类别的目标，因此输出文件中应该只包含这两个类别的目标跟踪结果，且不应该包含Don't care类别。输出结果中的每行中的`truncated`和`occluded` 项设置为默认值-1.

​	为了设置评估的序列，在./data/tracking/下面的`evaluate_tracking.seqmap`设置要评估的序列。若想更改映射的文件，可在`./evaluate_tracking.py`中的代码 `filename_test_mapping = "./data/tracking/my_test.seqmap"` 设置文件名。







## Evaluate MOT with TrackEval

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







## Evaluate MOT with KITTI devkit_object

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