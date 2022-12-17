#!/bin/bash

source ~/.bashrc

export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

dynamic_vins_root=/home/chen/ws/dynamic_ws

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/euroc_egomotion


estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/output


estimate_array=( \
"MH_01_easy_VIO_raw_PointOnly_Odometry.txt"  \
"MH_01_easy_VIO_raw_LinePoint_Odometry.txt"  \
"MH_02_easy_VIO_raw_PointOnly_Odometry.txt"  \
"MH_02_easy_VIO_raw_LinePoint_Odometry.txt"  \
"MH_03_medium_VIO_raw_PointOnly_Odometry.txt"  \
"MH_03_medium_VIO_raw_LinePoint_Odometry.txt"  \
"MH_04_difficult_VIO_raw_PointOnly_Odometry.txt"  \
"MH_04_difficult_VIO_raw_LinePoint_Odometry.txt"  \
"MH_05_difficult_VIO_raw_PointOnly_Odometry.txt"  \
"MH_05_difficult_VIO_raw_LinePoint_Odometry.txt"  \
"V1_01_easy_VIO_raw_PointOnly_Odometry.txt"  \
"V1_01_easy_VIO_raw_LinePoint_Odometry.txt"  \
"V1_02_medium_VIO_raw_PointOnly_Odometry.txt"  \
"V1_02_medium_VIO_raw_LinePoint_Odometry.txt"  \
"V1_03_difficult_VIO_raw_PointOnly_Odometry.txt"  \
"V1_03_difficult_VIO_raw_LinePoint_Odometry.txt"  \
"V2_01_easy_VIO_raw_PointOnly_Odometry.txt"  \
"V2_01_easy_VIO_raw_LinePoint_Odometry.txt"  \
"V2_02_medium_VIO_raw_PointOnly_Odometry.txt"  \
"V2_02_medium_VIO_raw_LinePoint_Odometry.txt"  \
"V2_03_difficult_VIO_raw_PointOnly_Odometry.txt"  \
"V2_03_difficult_VIO_raw_LinePoint_Odometry.txt"  \
)

gt_array=( \
"MH_01_easy"  \
"MH_01_easy"   \
"MH_02_easy"  \
"MH_02_easy"  \
"MH_03_medium"  \
"MH_03_medium"  \
"MH_04_difficult"  \
"MH_04_difficult"  \
"MH_05_difficult"  \
"MH_05_difficult"  \
"V1_01_easy"  \
"V1_01_easy"  \
"V1_02_medium"  \
"V1_02_medium"  \
"V1_03_difficult"  \
"V1_03_difficult"  \
"V2_01_easy"  \
"V2_01_easy"  \
"V2_02_medium"  \
"V2_02_medium"  \
"V2_03_difficult"  \
"V2_03_difficult"  \
)


if [ "${#estimate_array[*]}" -ne "${#gt_array[*]}" ] ; then
	echo "estimate_array's size not equal gt_array's size"
	exit 1
fi


arr_length="${#estimate_array[*]}"

for ((i=0; i<$arr_length; i++))
do

	gt_name=${gt_path}/${gt_array[$i]}.txt

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

