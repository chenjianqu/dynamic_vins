/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_SPEED_FACTOR_H
#define DYNAMIC_VINS_SPEED_FACTOR_H

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <sophus/so3.hpp>
#include <utility>

#include "utils/def.h"
#include "utils/parameters.h"

namespace dynamic_vins{\


//优化变量:Twbj 7,Tbc 7,Twoj 7,Twoi 7,speed 6 ,逆深度 1,
class ProjectionInstanceSpeedFactor: public ceres::SizedCostFunction<3, 7,7,7,7,6,1>{
public:
    ProjectionInstanceSpeedFactor(Vec3d pts_j_,double time_j_,double time_i_):pts_j(std::move(pts_j_)),time_ij(time_i_-time_j_){
        sqrt_info = kFocalLength / 1.5 * Mat3d::Identity();
    }
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_j;//估计值（以前的观测值）
    double time_ij;

    inline static Mat3d sqrt_info;
    inline static double sum_t{0};
};



/**
 * 关于速度的重投影误差
 * 维度:<误差项、IMU位姿1、IMU位姿2、外参1、物体的运动速度、逆深度>
 */
class ProjectionSpeedFactor: public ceres::SizedCostFunction<2,7,7,6,1>{
public:
    /**
     * 构造函数
     * @param _pts_j j时刻的观测值
     * @param _pts_i i时刻的观测值
     * @param _velocity_j j点的速度
     * @param _velocity_i i点的速度
     * @param R_bc1_ j点的IMU外参
     * @param P_bc1_ j点的IMU外参
     * @param R_bc2_ i点的IMU外参
     * @param P_bc2_ i点的IMU外参
     * @param _td_j j点的td
     * @param _td_i i点的td
     * @param cur_td_ 当前的td
     * @param time_j_ j时刻
     * @param time_i_ i时刻
     * @param factor_ 系数
     */
    ProjectionSpeedFactor(Vec3d _pts_j, Vec3d _pts_i,
                          const Vec2d &_velocity_j, const Vec2d &_velocity_i,
                          const Mat3d &R_bc1_, const Vec3d &P_bc1_,
                          const Mat3d &R_bc2_, const Vec3d &P_bc2_,
                          const double _td_j, const double _td_i,const double cur_td_,
                          double time_j_,double time_i_,double factor_)
                          :pts_j(std::move(_pts_j)),pts_i(std::move(_pts_i)),
                          time_ij(time_i_-time_j_),
                          td_i(_td_i), td_j(_td_j),cur_td(cur_td_),factor(factor_){
        velocity_i.x() = _velocity_i.x();
        velocity_i.y() = _velocity_i.y();
        velocity_i.z() = 0;
        velocity_j.x() = _velocity_j.x();
        velocity_j.y() = _velocity_j.y();
        velocity_j.z() = 0;

        P_bc1=P_bc1_;
        R_bc1=R_bc1_;
        P_bc2=P_bc2_;
        R_bc2=R_bc2_;

        sqrt_info = kFocalLength / 1.5 * Mat2d::Identity();
    }


    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;


    Vec3d pts_j,pts_i;//估计值（以前的观测值）
    double time_ij;

    Vec3d velocity_i, velocity_j;
    double td_i, td_j,cur_td;

    inline static Mat2d sqrt_info;
    inline static double sum_t{0};

    Mat3d R_bc1,R_bc2;
    Vec3d P_bc1,P_bc2;

    double factor;
};





/**
 * 维度:<误差项、物体的运动速度、逆深度>
 */
class ProjectionSpeedSimpleFactor: public ceres::SizedCostFunction<2,6,1>{
public:
    ProjectionSpeedSimpleFactor(Vec3d _pts_j, Vec3d _pts_i,
                                const Vec2d &_velocity_j, const Vec2d &_velocity_i,
                                const double _td_j, const double _td_i, const double cur_td_, double time_j_, double time_i_,
                                Mat3d R_wbj_, Vec3d P_wbj_,
                                Mat3d R_wbi_, Vec3d P_wbi_,
                                Mat3d R_bc1_, Vec3d P_bc1_,
                                Mat3d R_bc2_, Vec3d P_bc2_,
                                const double factor_)
                                :pts_j(std::move(_pts_j)),pts_i(std::move(_pts_i)),time_ij(time_i_-time_j_),
                                td_i(_td_i), td_j(_td_j),cur_td(cur_td_),
                                R_wbj(std::move(R_wbj_)),R_wbi(std::move(R_wbi_)),R_bc1(std::move(R_bc1_)),R_bc2(std::move(R_bc2_)),
                                P_wbj(std::move(P_wbj_)),P_wbi(std::move(P_wbi_)), P_bc1(std::move(P_bc1_)),P_bc2(std::move(P_bc2_)),factor(factor_){
        velocity_i.x() = _velocity_i.x();
        velocity_i.y() = _velocity_i.y();
        velocity_i.z() = 0;
        velocity_j.x() = _velocity_j.x();
        velocity_j.y() = _velocity_j.y();
        velocity_j.z() = 0;

        sqrt_info = kFocalLength / 1.5 * Mat2d::Identity();
    }


    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_j,pts_i;//估计值（以前的观测值）
    double time_ij;

    Vec3d velocity_i, velocity_j;
    double td_i, td_j,cur_td;

    inline static Mat2d sqrt_info;
    inline static double sum_t{0};

    Mat3d R_wbj,R_wbi,R_bc1,R_bc2;
    Vec3d P_wbj,P_wbi,P_bc1,P_bc2;

    double factor;

};







class SpeedPoseSimpleFactor: public ceres::SizedCostFunction<3, 7,7,6,1>{
public:
    SpeedPoseSimpleFactor(Vec3d pts_j_, double time_j_, double time_i_, Mat3d &R_wbj_, Vec3d &P_wbj_, Mat3d &R_bc_,
                          Vec3d &P_bc_,const Vec2d &vel_j_,const double td_j_, const double cur_td_):
                          pts_j(std::move(pts_j_)),time_ij(time_i_-time_j_),td(cur_td_),td_j(td_j_){
        R_wbj=R_wbj_;
        R_bc=R_bc_;
        P_wbj=P_wbj_;
        P_bc=P_bc_;
        vel_j.x()=vel_j_.x();
        vel_j.y()=vel_j_.y();
        vel_j.z()=1;
        sqrt_info = kFocalLength / 1.5 * Mat3d::Identity();

    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_j;//估计值（以前的观测值）
    double time_ij;

    inline static Mat3d sqrt_info;
    inline static double sum_t{0};

    Mat3d R_wbj,R_bc;
    Vec3d P_wbj, P_bc;

    Vec3d vel_j;
    double td,td_j;
};



class ConstSpeedFactor: public ceres::SizedCostFunction<3,6>{
public:
    ConstSpeedFactor(Vec3d pts_j_, double time_j_, double time_i_, Mat3d &R_wbj_,
                     Vec3d &P_wbj_, Mat3d &R_bc_, Vec3d &P_bc_, double inv_depth_,
                     Vec3d &last_v_, Vec3d &last_a_)
                     :
                     pts_j(std::move(pts_j_)),time_ij(time_i_-time_j_){
        R_wbj=R_wbj_;
        R_bc=R_bc_;
        P_wbj=P_wbj_;
        P_bc=P_bc_;
        inv_depth=inv_depth_;
        last_v=last_v_;
        last_a=last_a_;

        sqrt_info = kFocalLength / 1.5 * Mat3d::Identity();
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;


    Vec3d pts_j;//估计值（以前的观测值）
    double time_ij;

    inline static Mat3d sqrt_info;
    inline static double sum_t{0};

    Mat3d R_wbj,R_bc;
    Vec3d P_wbj, P_bc;

    double inv_depth;

    Vec3d last_v,last_a;
};



class ProjectionSpeedPoseFactor: public ceres::SizedCostFunction<3, 7,7,7,7,6,1>{
public:
    ProjectionSpeedPoseFactor(Vec3d pts_j_, double time_j_, double time_i_):
    pts_j(std::move(pts_j_)),time_ij(time_i_-time_j_)
    {}
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_j;//估计值（以前的观测值）
    double time_ij;
};


class ConstSpeedSimpleFactor: public ceres::SizedCostFunction<1,6>{
public:
    ConstSpeedSimpleFactor(const Vec3d &last_v_, const Vec3d &last_a_, double factor_){
        last_v=last_v_;
        last_a=last_a_;
        factor=factor_;
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d last_v,last_a;
    double factor;
};


/**
 * 速度和物体位姿的误差
 * 误差维度6, 优化变量:物体位姿woi(未实现), 物体位姿woj(未实现),物体速度6
 */
class SpeedPoseFactor: public ceres::SizedCostFunction<6,7,7,6>{
public:
    /**
     * 位姿所处的两个时刻,其中i<j
     * @param time_j_
     * @param time_i_
     */
    SpeedPoseFactor(double time_i_, double time_j_):time_ij(time_j_-time_i_){}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    double time_ij;//时间差
};



}


#endif //DYNAMIC_VINS_SPEED_FACTOR_H
