


# Evaluate

* set alias

```shell
vim ~/.bashrc
```

write down:

```shell
alias clion='sh /home/chen/app/clion-2021.2/bin/clion.sh'
alias sourced='source devel/setup.bash'
alias sourceb='source ~/.bashrc'
alias startconda='export PATH="/home/chen/anaconda3/bin:$PATH" && source activate'
```






## Evaluate odometry with EVO

### Install EVO

```shell
startconda && conda activate py36

sudo apt install python-pip 
pip install evo --upgrade --no-binary evo 
```



### Evaluate Kitti-Tracking odometry

#### Manually

* Prepare ground-truth

```shell
#project_root
sequence=0004

dataset_root=/home/chen/datasets/kitti/tracking

####Convert KITTI's oxts data to TUM format pose
oxts_path=${dataset_root}/data_tracking_oxts/training/oxts/
save_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/

# shellcheck disable=SC2039
source ${dynamic_vins_root}/devel/setup.bash
rosrun dynamic_vins_eval save_oxts ${oxts_path}/${sequence}.txt ${save_path}/${sequence}.txt
```



* Evaluation

```shell
startconda && conda activate py36

sequence=0003

estimate_file=0003_VO_dynamic_LinePoint_Odometry.txt

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/output


evo_ape tum --align  ${estimate_path}/${estimate_file} ${gt_path}/${sequence}.txt \
&& evo_rpe tum --align  ${estimate_path}/${estimate_file} ${gt_path}/${sequence}.txt -r trans_part \
&& evo_rpe tum --align  ${estimate_path}/${estimate_file} ${gt_path}/${sequence}.txt -r rot_part
```

输出：

```shell
APE w.r.t. translation part (m)
(with SE(3) Umeyama alignment)

       max	12.310783
      mean	1.788885
    median	1.551813
       min	0.217193
      rmse	2.563968
       sse	940.072483
       std	1.836797

RPE w.r.t. translation part (m)
for delta = 1 (frames) using consecutive pairs
(with SE(3) Umeyama alignment)

       max	11.254441
      mean	1.784602
    median	1.674964
       min	1.159199
      rmse	1.971327
       sse	551.830494
       std	0.837452

RPE w.r.t. rotation part (unit-less)
for delta = 1 (frames) using consecutive pairs
(with SE(3) Umeyama alignment)

       max	0.025461
      mean	0.006245
    median	0.005226
       min	0.000699
      rmse	0.007423
       sse	0.007824
       std	0.004013

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





#### Use shell script

* Evaluate

```shell
startconda && conda activate py36

estimate_dir=data/exp/2022-6-7/kitti_tracking_naive/
sequence=0004

source ${dynamic_vins_root}/src/dynamic_vins/scripts/eval_kitti_tracking_traj.sh ${estimate_dir} ${sequence} 
```





### Evaluate VIODE odometry



* Generate VIODE ground-truth trajectory

```shell
cd dynamic_ws
sourced

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
startconda && conda activate py36

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/viode_egomotion/
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-12-05/viode_odometry/point_line/raw

sequence=city_day_0_none

evo_traj tum ${estimate_path}/${sequence}_ego-motion.txt --ref=${gt_path}/${sequence}.txt --align -p
```



* Evaluate

```shell
startconda && conda activate py36

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/viode_egomotion/
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-12-05/viode_odometry/

sequence=city_day_0_none
mode=raw
use_line=LinePoint

file_name=${estimate_path}/${sequence}_${mode}_${use_line}_Odometry.txt
evo_ape tum --align  ${file_name} ${gt_path}/${sequence}.txt && evo_rpe tum --align  ${file_name} ${gt_path}/${sequence}.txt -r trans_part && evo_rpe tum --align  ${file_name} ${gt_path}/${sequence}.txt -r rot_part

```



或者使用脚本直接评估: 修改`eval_viode_odometry.sh`：

```shell
sh ${dynamic_vins_root}/src/dynamic_vins/scripts/eval_viode_odometry.sh
```





### Evaluate EuRoc odometry

* Prepare EuRoc dataset ground-truth

```shell
startconda && conda activate py36

evo_traj euroc --help

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/euroc_egomotion/

mkdir ${gt_path}

dataset_dir=/home/chen/datasets/EuRoc


sequence=V2_03_difficult
raw_gt_path=${dataset_dir}/${sequence}/mav0/state_groundtruth_estimate0/data.csv
save_gt_path=${gt_path}/${sequence}.txt

evo_traj euroc ${raw_gt_path} --save_as_tum #得到data.tum文件
mv ./data.tum ${save_gt_path} #移动文件

```



* Visualize Trajectory

```shell
gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/euroc_egomotion/MH_01_easy.txt

estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-12-13/EuRoc/MH_01_easy_VIO_raw_LinePoint_Odometry.txt

evo_traj tum ${estimate_path} --ref=${gt_path} --align -p
```



仅可视化xy轴

```shell
sequence=V2_03_difficult

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/EuRoc/${sequence}_gt.txt

estimate_path_1=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-12-13/EuRoc_evaluate/${sequence}_PL.txt

estimate_path_2=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-12-13/EuRoc_evaluate/${sequence}_PO.txt

evo_traj tum ${estimate_path_1} ${estimate_path_2} --ref=${gt_path} --align -p --plot_mode xy

或
evo_traj tum \
${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/EuRoc/MH_02_easy_gt.txt \
${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/EuRoc/MH_03_medium_gt.txt \
${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/EuRoc/MH_04_difficult_gt.txt \
${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/EuRoc/MH_05_difficult_gt.txt \
--ref=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/EuRoc/MH_01_easy_gt.txt \
 -p --plot_mode xy
```







* Evaluate

```shell
startconda && conda activate py36

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/euroc_egomotion
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/output

sequence=MH_01_easy

evo_ape tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt && evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r trans_part && evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r rot_part
```

或者使用脚本直接评估: 修改`eval_euroc_odometry.sh`：

```shell
sh ${dynamic_vins_root}/src/dynamic_vins/scripts/eval_euroc_odometry.sh
```







### Evaluate Custom dataset odometry

```shell
startconda && conda activate py36



gt_name=${gt_path}/room_static_2/vicon.txt
estimate_name=${estimate_path}/room_static_2_VO_raw_PointLine_Odometry.txt

evo_ape tum --align  ${estimate_name} ${gt_name} && evo_rpe tum --align  ${estimate_name} ${gt_name} -r trans_part && evo_rpe tum --align  ${estimate_name} ${gt_name} -r rot_part
```

and draw the plot:

```shell
evo_traj tum ${estimate_name}  --ref=${gt_name}  --align -p
```



* Evaluate and write to csv

将输出的评估结果绘制为表格

```shell
startconda && conda activate py36

gt_name=${gt_path}/room_static_3/vicon.txt

estimate_file=room_static_3_VO_raw_LinePoint_Odometry.txt

echo ${estimate_file}  > ape.txt
evo_ape tum --align  ${estimate_path}/${estimate_file} ${gt_name} >> ape.txt

echo ${estimate_file}  > rpe_t.txt
evo_rpe tum --align  ${estimate_path}/${estimate_file} ${gt_name}  -r trans_part  >> rpe_t.txt

echo ${estimate_file}  > rpe_r.txt
evo_rpe tum --align  ${estimate_path}/${estimate_file} ${gt_name}  -r rot_part  >> rpe_r.txt

python ${dynamic_vins_root}/src/dynamic_vins/scripts/python/write_summary.py ape.txt ape.csv
python ${dynamic_vins_root}/src/dynamic_vins/scripts/python/write_summary.py rpe_t.txt rpe_t.csv
python  ${dynamic_vins_root}/src/dynamic_vins/scripts/python/write_summary.py rpe_r.txt rpe_r.csv

```



或者，修改 `${dynamic_vins_root}/src/dynamic_vins/scripts/eval_custom_odometry.sh`，直接执行脚本即可：

```shell
sh ${dynamic_vins_root}/src/dynamic_vins/scripts/eval_custom_odometry.sh
```





## Evaluate odometry with TUM scripts

ref:https://blog.csdn.net/weixin_50508111/article/details/120687985

### 使用

在对Kinect的摄像机轨迹进行估计并保存到文件后，我们需要通过与地面真实值的比较来评估估计轨迹中的误差。有不同的误差度量。两种比较突出的方法是绝对轨迹误差法(ATE)和相对位姿误差法(RPE)。
该ATE非常适合于测量视觉SLAM系统的性能。
相比之下，RPE非常适合于测量视觉里程计系统的漂移，例如每秒的漂移。

#### evaluate_ate.py 

绝对轨迹误差直接测量真实轨迹和估计轨迹之间的点差。作为预处理步骤，我们使用时间戳将估计的姿态与地面真实姿态关联起来。在此基础上，我们利用奇异值分解对真实轨迹和估计轨迹进行对齐。最后，我们计算每对姿势之间的差值，并输出这些差值的平均值/中值/标准差。可选地，脚本可以将两条轨迹绘制为png或pdf文件。

```shell
usage: evaluate_ate.py [-h] [--offset OFFSET] [--scale SCALE]
                       [--max_difference MAX_DIFFERENCE] [--save SAVE]
                       [--save_associations SAVE_ASSOCIATIONS] [--plot PLOT]
                       [--verbose]
                       first_file second_file

This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.

positional arguments:
  first_file            first text file (format: timestamp tx ty tz qx qy qz
                        qw)
  second_file           second text file (format: timestamp tx ty tz qx qy qz
                        qw)

optional arguments:
  -h, --help            show this help message and exit
  --offset OFFSET       time offset added to the timestamps of the second file
                        (default: 0.0)
  --scale SCALE         scaling factor for the second trajectory (default:
                        1.0)
  --max_difference MAX_DIFFERENCE
                        maximally allowed time difference for matching entries
                        (default: 0.02)
  --save SAVE           save aligned second trajectory to disk (format: stamp2
                        x2 y2 z2)
  --save_associations SAVE_ASSOCIATIONS
                        save associated first and aligned second trajectory to
                        disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)
  --plot PLOT           plot the first and the aligned second trajectory to an
                        image (format: png)
  --verbose             print all evaluation data (otherwise, only the RMSE
                        absolute translational error in meters after alignment
                        will be printed)

```

如：

```shell
python evaluate_ate.py groundtruth.txt pose_output.txt --plot result.png
```





#### evaluate_rpe.py

为了计算相对姿态误差，我们提供了一个脚本“evaluate_rpe.py”。这个脚本计算时间戳对之间相对运动的误差。默认情况下，脚本会计算估计轨迹文件中所有时间戳对之间的误差。由于估计轨迹中时间戳对的数量是轨迹长度的二次元，因此将该集合向下采样到一个固定的数字(-max_pairs)是有意义的。或者，也可以选择使用固定的窗口大小(-fixed_delta)。在这种情况下，估计轨迹中的每个姿态都会根据窗口大小(-delta)和单位(-delta_unit)与之后的姿态相关联。这种评价技术对估计漂移是有用的。

```shell
usage: evaluate_rpe.py [-h] [--max_pairs MAX_PAIRS] [--fixed_delta]
                       [--delta DELTA] [--delta_unit DELTA_UNIT]
                       [--offset OFFSET] [--scale SCALE] [--save SAVE]
                       [--plot PLOT] [--verbose]
                       groundtruth_file estimated_file

This script computes the relative pose error from the ground truth trajectory
and the estimated trajectory.

positional arguments:
  groundtruth_file      ground-truth trajectory file (format: "timestamp tx ty
                        tz qx qy qz qw")
  estimated_file        estimated trajectory file (format: "timestamp tx ty tz
                        qx qy qz qw")

optional arguments:
  -h, --help            show this help message and exit
  --max_pairs MAX_PAIRS
                        maximum number of pose comparisons (default: 10000,
                        set to zero to disable downsampling)
  --fixed_delta         only consider pose pairs that have a distance of delta
                        delta_unit (e.g., for evaluating the drift per
                        second/meter/radian)
  --delta DELTA         delta for evaluation (default: 1.0)
  --delta_unit DELTA_UNIT
                        unit of delta (options: 's' for seconds, 'm' for
                        meters, 'rad' for radians, 'f' for frames; default:
                        's')
  --offset OFFSET       time offset between ground-truth and estimated
                        trajectory (default: 0.0)
  --scale SCALE         scaling factor for the estimated trajectory (default:
                        1.0)
  --save SAVE           text file to which the evaluation will be saved
                        (format: stamp_est0 stamp_est1 stamp_gt0 stamp_gt1
                        trans_error rot_error)
  --plot PLOT           plot the result to a file (requires --fixed_delta,
                        output format: png)
  --verbose             print all evaluation data (otherwise, only the mean
                        translational error measured in meters will be
                        printed)
```





### 评估EuRoc

use python2

```shell
estimate_file=${dynamic_vins_root}/src/dynamic_vins/data/exp/2022-12-13/EuRoc/MH_01_easy_VIO_raw_LinePoint_Odometry.txt

gt_file=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/euroc_egomotion/MH_01_easy.txt

python evaluate_ate.py ${gt_file} ${estimate_file} --plot result.png
```









## Evaluate object trajectory with EVO

* Visualize ground-turth boxes and its IDs

```shell
sequence=0018

gt_file=/home/chen/datasets/kitti/tracking/data_tracking_label_2/training/label_02/${sequence}.txt
image_dir=/home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/${sequence}

rosrun dynamic_vins_eval visualize_3d_box ${gt_file} ${image_dir}
```

根据该程序，可以看到每个物体的gt id。

* Split object tracking label to `TUM` format

```shell
startconda && conda activate py36

sequence=0018
gt_id=3
object_id=9
#object_id=1 - 0
#object_id=9 - 3
pose_file=0018_VO_dynamic_PointOnly_Odometry.txt

${dynamic_vins_root}/src/dynamic_vins/scripts/eval_object_traj.sh ${sequence} ${gt_id} ${object_id} ${pose_file}
```



* Visualize the **estimate** object trajectory and **ground-truth** trajectory

```shell
ref_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_tum/${sequence}_${gt_id}.txt
estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/output/${sequence}/${sequence}_tum/${object_id}.txt

evo_traj tum ${estimate_path} --ref=${ref_path} -p
```









## Evaluate MOT with KITTI devkit_tracking



### Install devkit_tracking

Download the devkit： [The KITTI Vision Benchmark Suite (cvlibs.net)](http://www.cvlibs.net/datasets/kitti/eval_tracking.php) .

Run environment: `Python2`, dependency:

* pip install munkres

### Evaluate

执行评估:

```shell
sequence=0018
gt_object_id=3
estimate_object_id=9

source ${dynamic_vins_root}/src/dynamic_vins/scripts/eval_mot_kitti_tracking.sh ${sequence} ${gt_object_id} ${estimate_object_id}
```

​	这是一个python脚本，其中dynamic_vins是`./result`下保存模型输出结果的地方。

​	需要注意的是，该kit只评估Car（van）类别和Pedestrian类别的目标，因此输出文件中应该只包含这两个类别的目标跟踪结果，且不应该包含Don't care类别。输出结果中的每行中的`truncated`和`occluded` 项设置为默认值-1.

​	为了设置评估的序列，在./data/tracking/下面的`evaluate_tracking.seqmap`设置要评估的序列。若想更改映射的文件，可在`./evaluate_tracking.py`中的代码 `filename_test_mapping = "./data/tracking/my_test.seqmap"` 设置文件名。







## Evaluate MOT with TrackEval

#### Install TrackEval

Here we use TrackEval to evaluate our mot estimation result.

```shell
git clone https://github.com/JonathonLuiten/TrackEval.git

startconda
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
startconda
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
