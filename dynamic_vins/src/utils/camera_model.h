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

#include "utils/def.h"

namespace dynamic_vins{\

using CamModelType=camodocal::Camera::ModelType;


/*class PinHoleCamera{
public:
    using Ptr=std::shared_ptr<PinHoleCamera>;
    PinHoleCamera(){}


    bool ReadFromYamlFile(const std::string& filename);

    void LiftProjective(const Vec2d& p, Vec3d& P) const;

    void ProjectPoint(const Vec3d& p3d, Vec2d& p2d) const{
        p2d.x() = p3d.x()/p3d.z() * fx + cx;
        p2d.y() = p3d.y()/p3d.z() * fy + cy;
    }

    float DepthFromDisparity(float disp){
        return fx * baseline / disp;
    }

    float fx,fy,cx,cy;
    float baseline;
    int image_width,image_height;
    float k1,k2,p1,p2;//畸变矫正
    float inv_k11,inv_k22,inv_k13,inv_k23;//用于反投影
    std::string camera_name;
};*/


//inline std::shared_ptr<PinHoleCamera> cam0;
//inline std::shared_ptr<PinHoleCamera> cam1;


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
