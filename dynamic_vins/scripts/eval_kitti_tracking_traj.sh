#!/bin/sh

estimate_dir=$1
sequence=$2

#KITTI TRACKING dataset
dataset_root=/home/chen/datasets/kitti/tracking

#project_root
dynamic_vins_root=/home/chen/ws/dynamic_ws


####Convert KITTI's oxts data to TUM format pose
oxts_path=${dataset_root}/data_tracking_oxts/training/oxts/
save_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/

# shellcheck disable=SC2039
source ${dynamic_vins_root}/devel/setup.bash
rosrun dynamic_vins_eval save_oxts ${oxts_path}/${sequence}.txt ${save_path}/${sequence}.txt


#### Eval

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_egomotion/
estimate_path=${dynamic_vins_root}/src/dynamic_vins/${estimate_dir}


evo_ape tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt \
&& evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r trans_part \
&& evo_rpe tum --align  ${estimate_path}/${sequence}_ego-motion.txt ${gt_path}/${sequence}.txt -r rot_part


