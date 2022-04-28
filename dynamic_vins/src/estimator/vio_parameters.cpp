/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "vio_parameters.h"

#include <opencv2/opencv.hpp>

#include "utils/dataset/kitti_utils.h"
#include "utils/def.h"


namespace dynamic_vins{\

///读取相机到IMU的外参
void ReadCameraToIMU(const std::string& config_path){
    ///设置为单位矩阵
    if (cfg::is_estimate_ex == 2){
        para::RIC.emplace_back(Eigen::Matrix3d::Identity());
        para::TIC.emplace_back(Eigen::Vector3d::Zero());
        if(cfg::kCamNum==2){
            para::RIC.emplace_back(Eigen::Matrix3d::Identity());
            para::TIC.emplace_back(Eigen::Vector3d::Zero());
        }
        return;
    }

    ///从文件中读取外参
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(std::string("ERROR: Wrong path to settings:"+config_path));
    }

    ///直接从kitti的参数文件中读取
    if(cfg::dataset == DatasetType::kKitti){
        if(fs["kitti_calib_path"].isNone()){
            cerr<<"Use the kitti dataset,but not set kitti_calib_path"<<endl;
            fs.release();
            std::terminate();
        }
        string kitti_calib_path;
        fs["kitti_calib_path"] >> kitti_calib_path;
        auto calib_map = kitti::ReadCalibFile(kitti_calib_path);

        cout<<calib_map.size()<<endl;

        //将点从IMU坐标系变换到相机坐标系0的变换矩阵
        Mat4d T_imu_c0 =calib_map["Tr_imu_velo"] * calib_map["Tr_velo_cam"];

        double baseline_2 = calib_map["P2"](0,3) / (- cam0->fx);

        Mat4d T_c0_c2 = Mat4d::Identity();
        T_c0_c2(0,3) = -baseline_2;
        Mat4d T_imu_c2 = T_imu_c0*T_c0_c2;
        Mat4d T_ic0 = T_imu_c2.inverse();

        para::RIC.emplace_back(T_ic0.block<3, 3>(0, 0));
        para::TIC.emplace_back(T_ic0.block<3, 1>(0, 3));

        if(cfg::kCamNum == 2){
            double baseline_3 = calib_map["P3"](0,3) / (- cam1->fx);
            Mat4d T_c0_c3 = Mat4d::Identity();
            T_c0_c2(0,3) = - baseline_3;
            Mat4d T_imu_c3 = T_imu_c0*T_c0_c3;
            Mat4d T_ic1 = T_imu_c3.inverse();
            para::RIC.emplace_back(T_ic1.block<3, 3>(0, 0));
            para::TIC.emplace_back(T_ic1.block<3, 1>(0, 3));
        }

    }
    ///从文件中读取相机内参
    else if(cfg::dataset == DatasetType::kViode){
        cv::Mat cv_T;
        fs["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        para::RIC.emplace_back(T.block<3, 3>(0, 0));
        para::TIC.emplace_back(T.block<3, 1>(0, 3));

        if(cfg::kCamNum == 2){
            fs["body_T_cam1"] >> cv_T;
            cv::cv2eigen(cv_T, T);
            para::RIC.emplace_back(T.block<3, 3>(0, 0));
            para::TIC.emplace_back(T.block<3, 1>(0, 3));
        }

    }

    fs.release();

    cfg::kExCalibResultPath = cfg::kOutputFolder + "/extrinsic_parameter.csv";

    cout<<"RIC0:\n"<<para::RIC[0]<<endl;
    cout<<"TIC0:\n"<<para::TIC[0]<<endl;
    if(cfg::kCamNum == 2){
        cout<<"RIC1:\n"<<para::RIC[1]<<endl;
        cout<<"TIC1:\n"<<para::TIC[1]<<endl;
    }
}

void VioParameters::SetParameters(const std::string &config_path)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(std::string("ERROR: Wrong path to settings:" + config_path));
    }

    kMaxSolverTime = fs["max_solver_time"];
    KNumIter = fs["max_num_iterations"];
    kMinParallax = fs["keyframe_parallax"];
    kMinParallax = kMinParallax / kFocalLength;

    if(cfg::is_use_imu){
        ACC_N = fs["acc_n"];
        ACC_W = fs["acc_w"];
        GYR_N = fs["gyr_n"];
        GYR_W = fs["gyr_w"];
        G.z() = fs["g_norm"];
    }

    fs["INIT_DEPTH"] >> kInitDepth;
    fs["BIAS_ACC_THRESHOLD"]>>BIAS_ACC_THRESHOLD;
    fs["BIAS_GYR_THRESHOLD"]>>BIAS_GYR_THRESHOLD;

    TD = fs["td"];

    fs.release();

    ///读取外参
    ReadCameraToIMU(config_path);

}










}
