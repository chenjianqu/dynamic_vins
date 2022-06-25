#!/bin/sh

sequence=$1
gt_id=$2
object_id=$3

#KITTI TRACKING dataset
dataset_root=/home/chen/datasets/kitti/tracking

#project_root
dynamic_vins_root=/home/chen/ws/dynamic_ws


### Prepare GT trajectory


gt_file=${dataset_root}/data_tracking_label_2/training/label_02/${sequence}.txt

cam_pose_file=${dynamic_vins_root}/src/dynamic_vins/data/output/${sequence}_ego-motion.txt

save_to_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_tum/${sequence}_${gt_id}.txt

# shellcheck disable=SC2039
source ${dynamic_vins_root}/devel/setup.bash
rosrun dynamic_vins_eval split_mot_to_tum ${gt_id} ${gt_file} ${cam_pose_file} ${save_to_path}


### Eval

estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/output/${sequence}_tum/${object_id}.txt

ref_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/kitti_tracking_tum/${sequence}_${gt_id}.txt

evo_ape tum --align  ${estimate_path} ${ref_path} \
 && evo_rpe tum --align  ${estimate_path} ${ref_path} -r trans_part \
 && evo_rpe tum --align  ${estimate_path} ${ref_path} -r rot_part


