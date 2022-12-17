/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <map>
#include <iostream>

#include <ros/ros.h>
#include <eigen3/Eigen/Dense>

#include "estimator/imu/imu_factor.h"
#include "estimator/utility.h"
#include "estimator/feature_manager.h"

namespace dynamic_vins{\


class ImageFrame{
public:
    ImageFrame(){};
    ImageFrame(const std::map<unsigned int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t) :t{_t},is_key_frame{false}
    {
        points = _points;
    };
    std::map<unsigned int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
    double t;
    Mat3d R;
    Vec3d T;
    IntegrationBase *pre_integration;
    bool is_key_frame;
};


void SolveGyroscopeBias(std::map<double, ImageFrame> &all_image_frame, Vec3d* Bgs);

bool VisualIMUAlignment(std::map<double, ImageFrame> &all_image_frame, Vec3d* Bgs, Vec3d &g, Eigen::VectorXd &x);

}
