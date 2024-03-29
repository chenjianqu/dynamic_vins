#!/bin/sh

sequence=$1
mot_estimate_file=$2

#KITTI TRACKING dataset
dataset_root=/home/chen/datasets/kitti/tracking

#project_root
dynamic_vins_root=/home/chen/ws/dynamic_ws


gt_file=${dataset_root}/data_tracking_label_2/training/label_02/${sequence}.txt

# 把gt到适当的位置
cp_to_gt_path=${dynamic_vins_root}/src/dynamic_vins_eval/eval_tools/devkit_tracking/devkit/python/data/tracking/label_02/${sequence}.txt
cp ${gt_file} ${cp_to_gt_path}


#将预测结果放到适当的位置
cp_to_path=${dynamic_vins_root}/src/dynamic_vins_eval/eval_tools/devkit_tracking/devkit/python/results/dynamic_vins/data/${sequence}.txt
cp ${mot_estimate_file} ${cp_to_path}



### set evaluate_tracking.seqmap

sequence_len=100

# shellcheck disable=SC2039
# shellcheck disable=SC2080
if ((${sequence}=="0000")); then
    sequence_len=000154
elif ((${sequence}=="0001")); then
    sequence_len=000447
elif ((${sequence}=="0002")); then
    sequence_len=000233
elif ((${sequence}=="0003")); then
    sequence_len=000144
elif ((${sequence}=="0004")); then
    sequence_len=000314
elif ((${sequence}=="0005")); then
    sequence_len=000297
elif ((${sequence}=="0006")); then
    sequence_len=000270
elif ((${sequence}=="0007")); then
    sequence_len=000800
elif ((${sequence}=="0008")); then
    sequence_len=000390
elif ((${sequence}=="0009")); then
    sequence_len=000803
elif ((${sequence}=="0010")); then
    sequence_len=000294
elif ((${sequence}=="0011")); then
    sequence_len=000373
elif ((${sequence}=="0012")); then
    sequence_len=000078
elif ((${sequence}=="0013")); then
    sequence_len=000340
elif ((${sequence}=="0014")); then
    sequence_len=000106
elif ((${sequence}=="0015")); then
    sequence_len=000376
elif ((${sequence}=="0016")); then
    sequence_len=000209
elif ((${sequence}=="0017")); then
    sequence_len=000145
elif ((${sequence}=="0018")); then
    sequence_len=000339
elif ((${sequence}=="0019")); then
    sequence_len=001059
elif ((${sequence}=="0020")); then
    sequence_len=000837
else
    sequence_len=001000
fi

# shellcheck disable=SC2034
str_to_write=${sequence}" empty 000000 "${sequence_len}
echo ${str_to_write} > ${dynamic_vins_root}/src/dynamic_vins_eval/eval_tools/devkit_tracking/devkit/python/data/tracking/evaluate_tracking.seqmap


### EVAL
# shellcheck disable=SC2164
cd ${dynamic_vins_root}/src/dynamic_vins_eval/eval_tools/devkit_tracking/devkit/python

python ./evaluate_tracking.py dynamic_vins
