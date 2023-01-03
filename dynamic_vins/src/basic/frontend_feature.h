/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_FRONTEND_FEATURE_H
#define DYNAMIC_VINS_FRONTEND_FEATURE_H

#include <optional>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "def.h"
#include "box2d.h"
#include "box3d.h"
#include "utils/parameters.h"
#include "line_detector/frame_lines.h"

#include "point_landmark.h"


namespace dynamic_vins{\


/**
 * 从前端传到后端的所有特征
 */
struct FeatureBackground{
    /*
     * 点特征
     * 格式：{id, [(camera_id,feature1),...,(camera_id,featureN)]}
     * feature1：Vector7d，分别表示
     */
    std::map<unsigned int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points;

    /*
     * 线特征,格式：{line_id,[{cam_id,line},...]}
     */
    std::map<unsigned int, std::vector<std::pair<int,Line>>> lines;
};

class FeatureInstance{
public:
    std::map<unsigned int,FeaturePoint::Ptr> features;

    cv::Scalar color;
    Box2D::Ptr box2d;
    Box3D::Ptr box3d;

    vector<Vec3d> points;
};


/**
 * 用于在前端和VIO之间传递信息
 */
struct FrontendFeature{
    FrontendFeature()=default;
    ///背景特征点
    FeatureBackground features;
    double time{0.0};

    unsigned int seq_id;//帧号

    ///根据物体的实例信息,格式：{instnce_id,{feature_id,}}
    std::map<unsigned int,FeatureInstance> instances;
};






}

#endif //DYNAMIC_VINS_FRONTEND_FEATURE_H
