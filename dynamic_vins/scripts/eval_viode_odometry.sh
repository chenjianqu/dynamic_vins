#!/bin/bash

source ~/.bashrc

export PATH="/home/chen/anaconda3/bin:$PATH" && source activate
conda activate py36

dynamic_vins_root=/home/chen/ws/dynamic_ws

gt_path=${dynamic_vins_root}/src/dynamic_vins/data/ground_truth/viode_egomotion


estimate_path=${dynamic_vins_root}/src/dynamic_vins/data/output


estimate_array=( \
"city_day_0_none_VIO_naive_LinePoint_Odometry.txt"  \
"city_day_0_none_VIO_naive_PointOnly_Odometry.txt"  \
"city_day_0_none_VIO_raw_LinePoint_Odometry.txt"  \
"city_day_0_none_VIO_raw_PointOnly_Odometry.txt"  \
"city_day_1_low_VIO_naive_LinePoint_Odometry.txt"  \
"city_day_1_low_VIO_naive_PointOnly_Odometry.txt"  \
"city_day_1_low_VIO_raw_LinePoint_Odometry.txt"  \
"city_day_1_low_VIO_raw_PointOnly_Odometry.txt"  \
"city_day_2_mid_VIO_naive_LinePoint_Odometry.txt"  \
"city_day_2_mid_VIO_naive_PointOnly_Odometry.txt"  \
"city_day_2_mid_VIO_raw_LinePoint_Odometry.txt"  \
"city_day_2_mid_VIO_raw_PointOnly_Odometry.txt"  \
"city_day_3_high_VIO_naive_LinePoint_Odometry.txt"  \
"city_day_3_high_VIO_naive_PointOnly_Odometry.txt"  \
"city_day_3_high_VIO_raw_LinePoint_Odometry.txt"  \
"city_day_3_high_VIO_raw_PointOnly_Odometry.txt"  \
"city_night_0_none_VIO_naive_LinePoint_Odometry.txt"  \
"city_night_0_none_VIO_naive_PointOnly_Odometry.txt"  \
"city_night_0_none_VIO_raw_LinePoint_Odometry.txt"  \
"city_night_0_none_VIO_raw_PointOnly_Odometry.txt"  \
"city_night_1_low_VIO_naive_LinePoint_Odometry.txt"  \
"city_night_1_low_VIO_naive_PointOnly_Odometry.txt"  \
"city_night_1_low_VIO_raw_LinePoint_Odometry.txt"  \
"city_night_1_low_VIO_raw_PointOnly_Odometry.txt"  \
"city_night_2_mid_VIO_naive_LinePoint_Odometry.txt"  \
"city_night_2_mid_VIO_naive_PointOnly_Odometry.txt"  \
"city_night_2_mid_VIO_raw_LinePoint_Odometry.txt"  \
"city_night_2_mid_VIO_raw_PointOnly_Odometry.txt"  \
"city_night_3_high_VIO_naive_LinePoint_Odometry.txt"  \
"city_night_3_high_VIO_naive_PointOnly_Odometry.txt"  \
"city_night_3_high_VIO_raw_LinePoint_Odometry.txt"  \
"city_night_3_high_VIO_raw_PointOnly_Odometry.txt"  \
"parking_lot_0_none_VIO_naive_LinePoint_Odometry.txt"  \
"parking_lot_0_none_VIO_naive_PointOnly_Odometry.txt"  \
"parking_lot_0_none_VIO_raw_LinePoint_Odometry.txt"  \
"parking_lot_0_none_VIO_raw_PointOnly_Odometry.txt"  \
"parking_lot_1_low_VIO_naive_LinePoint_Odometry.txt"  \
"parking_lot_1_low_VIO_naive_PointOnly_Odometry.txt"  \
"parking_lot_1_low_VIO_raw_LinePoint_Odometry.txt"  \
"parking_lot_1_low_VIO_raw_PointOnly_Odometry.txt"  \
"parking_lot_2_mid_VIO_naive_LinePoint_Odometry.txt"  \
"parking_lot_2_mid_VIO_naive_PointOnly_Odometry.txt"  \
"parking_lot_2_mid_VIO_raw_LinePoint_Odometry.txt"  \
"parking_lot_2_mid_VIO_raw_PointOnly_Odometry.txt"  \
"parking_lot_3_high_VIO_naive_LinePoint_Odometry.txt"  \
"parking_lot_3_high_VIO_naive_PointOnly_Odometry.txt"  \
"parking_lot_3_high_VIO_raw_LinePoint_Odometry.txt"  \
"parking_lot_3_high_VIO_raw_PointOnly_Odometry.txt"  \
)

gt_array=( \
"city_day_0_none"  \
"city_day_0_none"   \
"city_day_0_none"  \
"city_day_0_none"  \
"city_day_1_low"  \
"city_day_1_low"  \
"city_day_1_low"  \
"city_day_1_low"  \
"city_day_2_mid"  \
"city_day_2_mid"  \
"city_day_2_mid"  \
"city_day_2_mid"  \
"city_day_3_high"  \
"city_day_3_high"  \
"city_day_3_high"  \
"city_day_3_high"  \
"city_night_0_none"  \
"city_night_0_none"  \
"city_night_0_none"  \
"city_night_0_none"  \
"city_night_1_low"  \
"city_night_1_low"  \
"city_night_1_low"  \
"city_night_1_low"  \
"city_night_2_mid"  \
"city_night_2_mid"  \
"city_night_2_mid"  \
"city_night_2_mid"  \
"city_night_3_high"  \
"city_night_3_high"  \
"city_night_3_high"  \
"city_night_3_high"  \
"parking_lot_0_none"  \
"parking_lot_0_none"  \
"parking_lot_0_none"  \
"parking_lot_0_none"  \
"parking_lot_1_low"  \
"parking_lot_1_low"  \
"parking_lot_1_low"  \
"parking_lot_1_low"  \
"parking_lot_2_mid"  \
"parking_lot_2_mid"  \
"parking_lot_2_mid"  \
"parking_lot_2_mid"  \
"parking_lot_3_high"  \
"parking_lot_3_high"  \
"parking_lot_3_high"  \
"parking_lot_3_high"  \
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
	
	echo ${estimate_file}-----${gt_array[$i]}.txt

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

