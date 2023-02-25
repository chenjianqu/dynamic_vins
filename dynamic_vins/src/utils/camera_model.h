/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_CAMERA_MODEL_H
#define DYNAMIC_VINS_CAMERA_MODEL_H

#include <memory>
#include <string>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include <camodocal/camera_models/CameraFactory.h>

#include "basic/def.h"

namespace dynamic_vins{\

using CamModelType=camodocal::Camera::ModelType;


class CameraInfo{
public:
    camodocal::CameraPtr cam0,cam1;
    //左相机内参和畸变系数
    cv::Mat K0,D0;
    //左相机去畸变映射矩阵
    cv::Mat left_undist_map1, left_undist_map2;
    cv::Mat K1,D1;
    cv::Mat right_undist_map1, right_undist_map2;
    float fx0,fy0,cx0,cy0;//相机内参
    float fx1,fy1,cx1,cy1;
    float baseline;

    cv::Size img_size;

    CamModelType model_type;
};


/**
 * 相机到IMU的外参
 */
extern std::vector<Eigen::Matrix3d> R_IC;
extern std::vector<Eigen::Vector3d> T_IC;

extern CameraInfo cam_s;//用于segmentation线程的相机
extern CameraInfo cam_t;//用于tracking线程的相机
extern CameraInfo cam_v;//用于VIO线程的相机

void InitCamera(const std::string& config_path,const std::string& seq_name);

vector<string> GetCameraPath(const string &config_path);

}

#endif //DYNAMIC_VINS_CAMERA_MODEL_H
