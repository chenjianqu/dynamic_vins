/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_PROJECTIONFACTORSIMPLE_H
#define DYNAMIC_VINS_PROJECTIONFACTORSIMPLE_H



#include <ros/assert.h>
#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utils.h"
#include "../parameters.h"

class ProjectionTwoFrameOneCamFactorSimple : public ceres::SizedCostFunction<2, 7, 7, 7, 1>{
public:
    ProjectionTwoFrameOneCamFactorSimple(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Eigen::Vector3d pts_i, pts_j;
    Eigen::Matrix<double, 2, 3> tangent_base;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};


class ProjectionOneFrameTwoCamFactorSimple : public ceres::SizedCostFunction<2, 7, 7, 1>{
public:
    ProjectionOneFrameTwoCamFactorSimple(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Eigen::Vector3d pts_i, pts_j;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};


class ProjectionTwoFrameTwoCamFactorSimple : public ceres::SizedCostFunction<2, 7, 7, 7, 7, 1>{
public:
    ProjectionTwoFrameTwoCamFactorSimple(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Eigen::Vector3d pts_i, pts_j;
    static Eigen::Matrix2d sqrt_info;
    static double sum_t;
};


/**
 * 维度:<误差项，IMU位姿,外参,box,逆深度>
 */
class ProjectionBoxFactorSimple: public ceres::SizedCostFunction<3,7,7,3,1>{
public:
    explicit ProjectionBoxFactorSimple(const Eigen::Vector3d &pts_j_,const Eigen::Vector2d &velocity_j_,const double td_j_,const double curr_td_):
    pts_j(pts_j_),velocity_j(velocity_j_.x(),velocity_j_.y(),1),td_j(td_j_),curr_td(curr_td_){}
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;


    Eigen::Vector3d pts_j;//估计值（以前的观测值）
    Eigen::Vector3d velocity_j;
    double td_j,curr_td;

    static Eigen::Matrix3d sqrt_info;
    static double sum_t;
};



#endif //DYNAMIC_VINS_PROJECTIONFACTORSIMPLE_H
