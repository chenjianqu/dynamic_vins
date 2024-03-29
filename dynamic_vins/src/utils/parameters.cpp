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


namespace dynamic_vins{\


Config::Config(const std::string &file_name,const std::string &seq_name)
{
    cv::FileStorage fs(file_name, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(fmt::format("ERROR: Wrong path to settings:{}\n",file_name));
    }

    kDatasetSequence = seq_name;

    ///设置slam模式

    string slam_type;
    fs["slam_type"]>>slam_type;
    if(slam_type=="raw")
        slam=SLAM::kRaw;
    else if(slam_type=="naive")
        slam=SLAM::kNaive;
    else
        slam=SLAM::kDynamic;

    cout<<"SlamType::"<<slam_type<<endl;

    ///设置数据集

    fs["dataset_type"]>>dataset_name;
    std::transform(dataset_name.begin(),dataset_name.end(),dataset_name.begin(),::tolower);//

    if(dataset_name=="kitti")
        dataset = DatasetType::kKitti;
    else if(dataset_name=="viode")
        dataset = DatasetType::kViode;
    else if(dataset_name == "euroc")
        dataset = DatasetType::kEuRoc;
    else if(dataset_name == "custom")
        dataset = DatasetType::kCustom;
    else
        dataset = DatasetType::kViode;

    cout<<"dataset:"<<dataset_name<<endl;

    if(dataset == DatasetType::kViode && (slam == SLAM::kDynamic || Config::slam == SLAM::kNaive)){
        is_input_seg = true;
    }
    else{
        is_input_seg = false;
    }
    cout << "is_input_seg:" << is_input_seg << endl;

    is_vertical_draw = dataset == DatasetType::kKitti || dataset == DatasetType::kCustom;

    kInputHeight = fs["image_height"];
    kInputWidth = fs["image_width"];

    kCamNum = fs["num_of_cam"];
    if(kCamNum != 1 && kCamNum != 2){
        throw std::runtime_error("num_of_cam should be 1 or 2");
    }
    if(kCamNum==2){
        is_stereo = true;
    }

    use_imu = fs["imu"];
    cout << "is_use_imu:" << use_imu << endl;

    if(!use_imu){
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

    fs["use_line"] >> use_line;
    cout << "is_use_line:" << use_line << endl;

    fs["undistort_input"] >> is_undistort_input;
    cout<<"is_undistort_input: "<<is_undistort_input<<endl;

    if(!fs["use_dense_flow"].isNone()){
        fs["use_dense_flow"] >> use_dense_flow;
        cout<<"use_dense_flow: "<<use_dense_flow<<endl;
        if(use_dense_flow){
            fs["use_background_flow"]>>use_background_flow;
        }
    }

    fs["dst_mode"]>>dst_mode;
    cout<<"dst_mode: "<<dst_mode<<endl;

    if(dst_mode){
        is_only_frontend = true;
    }
    else{
        fs["only_imgprocess"]>>is_only_imgprocess;
        cout<<"is_only_imgprocess: "<<is_only_imgprocess<<endl;

        fs["only_frontend"]>>is_only_frontend;
        cout<<"is_only_frontend: "<<is_only_frontend<<endl;
    }

    fs["plane_constraint"] >> use_plane_constraint;
    cout<<"plane_constraint: "<<use_plane_constraint<<endl;

    if(slam==SLAM::kDynamic){
        if(fs["use_det3d"].isNone()){
            throw std::runtime_error("Config::Config(() fs[\"use_det3d\"].isNone()");
        }
        fs["use_det3d"] >> use_det3d;
        cout<<"use_det3d: "<<use_det3d<<endl;
    }


    fs["basic_dir"] >> kBasicDir;
    cout<<"basic_dir: "<<kBasicDir<<endl;

    fs.release();
}




}


