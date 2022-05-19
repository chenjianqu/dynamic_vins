/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/


#include "parameters.h"
#include <filesystem>

#include "utils/dataset/viode_utils.h"
#include "utils/dataset/coco_utils.h"
#include "utils/visualization.h"

namespace dynamic_vins{\



Config::Config(const std::string &file_name)
{
    cv::FileStorage fs(file_name, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(fmt::format("ERROR: Wrong path to settings:{}\n",file_name));
    }

    string slam_type;
    fs["slam_type"]>>slam_type;
    if(slam_type=="raw")
        slam=SlamType::kRaw;
    else if(slam_type=="naive")
        slam=SlamType::kNaive;
    else
        slam=SlamType::kDynamic;
    cout<<"SlamType::"<<slam_type<<endl;

    std::string dataset_type_string;
    fs["dataset_type"]>>dataset_type_string;
    std::transform(dataset_type_string.begin(),dataset_type_string.end(),dataset_type_string.begin(),::tolower);//
    if(dataset_type_string=="kitti")
        dataset = DatasetType::kKitti;
    else if(dataset_type_string=="viode")
        dataset = DatasetType::kViode;
    else
        dataset = DatasetType::kViode;
    cout<<"dataset:"<<dataset_type_string<<endl;

    if(dataset == DatasetType::kViode && (slam == SlamType::kDynamic || Config::slam == SlamType::kNaive)){
        is_input_seg = true;
    }
    else{
        is_input_seg = false;
    }
    cout << "is_input_seg:" << is_input_seg << endl;

    fs["dataset_sequence"]>>kDatasetSequence;

    kInputHeight = fs["image_height"];
    kInputWidth = fs["image_width"];

    kCamNum = fs["num_of_cam"];
    if(kCamNum != 1 && kCamNum != 2){
        throw std::runtime_error("num_of_cam should be 1 or 2");
    }
    if(kCamNum==2){
        is_stereo = true;
    }

    is_use_imu = fs["imu"];
    cout << "USE_IMU:" << is_use_imu << endl;

    if(is_use_imu){
        fs["imu_topic"] >> kImuTopic;
    }

    if(!is_use_imu){
        is_estimate_ex = 0;
        is_estimate_td = 0;
    }
    else{
        is_estimate_ex = fs["estimate_extrinsic"];
        if (is_estimate_ex == 2)
            cout<<"is_estimate_ex = 2: have no prior about extrinsic param, calibrate extrinsic param"<<endl;
        else if (is_estimate_ex == 1)
            cout<<"is_estimate_ex = 1: Optimize extrinsic param around initial guess!"<<endl;
        else if (is_estimate_ex == 0)
            cout<<"is_estimate_ex = 0: fix extrinsic param"<<endl;

        is_estimate_td = fs["estimate_td"];
        if (is_estimate_td)
            cout<<"is_estimate_td = 1. Unsynchronized sensors, online estimate time offset, initial"<<endl;
        else
            cout<<"is_estimate_td = 0. Synchronized sensors"<<endl;
    }


    fs["visual_inst_duration"] >> kVisualInstDuration;

    fs["use_dense_flow"] >> use_dense_flow;
    cout<<"use_dense_flow: "<<use_dense_flow<<endl;
    if(use_dense_flow){
        fs["use_background_flow"]>>use_background_flow;
    }

    fs["only_frontend"]>>is_only_frontend;
    cout<<"is_only_frontend: "<<is_only_frontend<<endl;

    fs["basic_dir"] >> kBasicDir;

    kOutputFolder = kBasicDir+"/output/";
    kVinsResultPath = kOutputFolder + "/vio.csv";

    std::ofstream fout(kVinsResultPath, std::ios::out);
    fout.close();

    fs["use_dataloader"]>>use_dataloader;
    if(use_dataloader){
        fs["image_dataset_left"]>>kImageDatasetLeft;
        fs["image_dataset_right"]>>kImageDatasetRight;
        fs["image_dataset_period"] >> kImageDatasetPeriod;
    }
    else{
        fs["image0_topic"] >> kImage0Topic;
        fs["image1_topic"] >> kImage1Topic;

        if(is_input_seg){
            fs["image0_segmentation_topic"] >> kImage0SegTopic;
            fs["image1_segmentation_topic"] >> kImage1SegTopic;
        }
    }



    fs.release();

    ///初始化logger
    MyLogger::InitLogger(file_name);

    ///初始化相机模型
    InitCamera(file_name);

    coco::SetParameters(file_name);

    if(dataset == DatasetType::kViode){
        VIODE::SetParameters(file_name);
    }

    ///清除之前的轨迹
    ClearTrajectoryFile();

}




}


