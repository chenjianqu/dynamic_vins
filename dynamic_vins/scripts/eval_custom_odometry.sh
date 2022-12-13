#!/bin/bash

source ~/.bashrc

export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

dynamic_vins_root=/home/chen/ws/dynamic_ws

gt_path=/home/chen/datasets/MyData/ZED_data


estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/output


estimate_array=( \
"room_static_1_VO_raw_PointOnly_Odometry.txt"  \
"room_static_1_VO_raw_LinePoint_Odometry.txt"  \
"room_static_2_VO_raw_PointOnly_Odometry.txt"  \
"room_static_2_VO_raw_LinePoint_Odometry.txt"  \
"room_static_3_VO_raw_PointOnly_Odometry.txt"  \
"room_static_3_VO_raw_LinePoint_Odometry.txt"  \
"room_dynamic_1_VO_raw_PointOnly_Odometry.txt"  \
"room_dynamic_1_VO_raw_LinePoint_Odometry.txt"  \
"room_dynamic_2_VO_raw_PointOnly_Odometry.txt" \
"room_dynamic_2_VO_raw_LinePoint_Odometry.txt"  \
"room_dynamic_3_VO_raw_PointOnly_Odometry.txt"  \
"room_dynamic_3_VO_raw_LinePoint_Odometry.txt"   \
"room_dynamic_1_VO_naive_PointOnly_Odometry.txt"  \
"room_dynamic_1_VO_naive_LinePoint_Odometry.txt"  \
"room_dynamic_2_VO_naive_PointOnly_Odometry.txt" \
"room_dynamic_2_VO_naive_LinePoint_Odometry.txt"  \
"room_dynamic_3_VO_naive_PointOnly_Odometry.txt"  \
"room_dynamic_3_VO_naive_LinePoint_Odometry.txt"   \
)

gt_array=( \
"room_static_1"  \
"room_static_1"   \
"room_static_2"  \
"room_static_2"  \
"room_static_3"  \
"room_static_3"  \
"room_dynamic_1"  \
"room_dynamic_1"  \
"room_dynamic_2"  \
"room_dynamic_2"  \
"room_dynamic_3"  \
"room_dynamic_3"  \
"room_dynamic_1"  \
"room_dynamic_1"  \
"room_dynamic_2"  \
"room_dynamic_2"  \
"room_dynamic_3"  \
"room_dynamic_3"  \
)


if [ "${#estimate_array[*]}" -ne "${#gt_array[*]}" ] ; then
	echo "estimate_array's size not equal gt_array's size"
	exit 1
fi


arr_length="${#estimate_array[*]}"

for ((i=0; i<$arr_length; i++))
do

	gt_name=${gt_path}/${gt_array[$i]}/vicon.txt

	estimate_file=${estimate_array[$i]}
	
	echo ${estimate_file} 

	echo ${estimate_file}  > ape.txt
	evo_ape tum --align  ${estimate_path}/${estimate_file} ${gt_name} >> ape.txt

	echo ${estimate_file}  > rpe_t.txt
	evo_rpe tum --align  ${estimate_path}/${estimate_file} ${gt_name}  -r trans_part  >> rpe_t.txt

	echo ${estimate_file}  > rpe_r.txt
	evo_rpe tum --align  ${estimate_path}/${estimate_file} ${gt_name}  -r rot_part  >> rpe_r.txt

	python ${dynamic_vins_root}/src/dynamic_vins/scripts/python/write_summary.py ape.txt ape.csv
	python ${dynamic_vins_root}/src/dynamic_vins/scripts/python/write_summary.py rpe_t.txt rpe_t.csv
	python  ${dynamic_vins_root}/src/dynamic_vins/scripts/python/write_summary.py rpe_r.txt rpe_r.csv

done

