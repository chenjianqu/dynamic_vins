/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
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






class ProjectionConstSpeedFactor: public ceres::SizedCostFunction<3,6>{
public:
    ProjectionConstSpeedFactor(Vec3d pts_j_, double time_j_, double time_i_, Mat3d &R_wbj_,
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



/**
 * 简单的速度恒定的二范数误差
 * 误差维度:1,
 * 优化变量: 速度 6
 */
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
 * 与3D点耦合的速度恒定误差
 * 误差维度:1,
 * 优化变量: 速度 6
 */
class ConstSpeedStereoPointFactor: public ceres::SizedCostFunction<1,6>{
public:
    /**
     * 构造函数
     * @param pts_i_ i时刻的3D点, i<j
     * @param pts_j_ j时刻的3D点
     * @param last_v_
     * @param last_a_
     */
    ConstSpeedStereoPointFactor(const Vec3d &pts_i_, const Vec3d &pts_j_,double time_ij_,
                                const Vec3d &last_v_, const Vec3d &last_a_)
                                : pts_i(pts_i_),pts_j(pts_j_),time_ij(time_ij_),last_v(last_v_),last_a(last_a_)
                                {}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_i,pts_j;
    double time_ij;
    Vec3d last_v,last_a;
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

    inline static int counter{0};
};


/**
 * 速度和物体位姿的误差
 * 误差维度6, 优化变量:物体速度6
 */
class SpeedPoseSimpleFactor: public ceres::SizedCostFunction<6,6>{
public:
    /**
     * 位姿所处的两个时刻,其中i<j
     * @param time_j_
     * @param time_i_
     */
    SpeedPoseSimpleFactor(double time_i_, double time_j_,Mat3d R_woi_,Vec3d P_woi_,Mat3d R_woj_,Vec3d P_woj_)
        :time_ij(time_j_-time_i_),R_woi(R_woi_),R_woj(R_woj_),P_woi(P_woi_),P_woj(P_woj_){}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    double time_ij;//时间差

    Mat3d R_woi,R_woj;
    Vec3d P_woi,P_woj;
};





/**
 * 双目3D点在不同时刻的误差,用来优化速度
 * 误差维度:1, 优化变量:速度 6,
 */
class SpeedStereoPointFactor: public ceres::SizedCostFunction<1,6>{
public:
    /**
     * 构造函数
     * @param _pts_j 第一个3D点(世界坐标系)
     * @param _pts_i 第二个3D点(世界坐标系)
     * @param time 两个点的时间差
     */
    SpeedStereoPointFactor(Vec3d _pts_j, Vec3d _pts_i,double time)
    :time_ij(time),pts_j(std::move(_pts_j) ),pts_i(std::move(_pts_i))
    {
        //Debugv("SpeedStereoPointFactor construct time_ij:{} pts_i:{} pts_j:{}",
        //       time_ij,VecToStr(pts_i),VecToStr(pts_j));
    }


    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    double time_ij{0};

    Vec3d pts_j,pts_i;//估计值（以前的观测值）

    inline static int counter=0;
};







}


#endif //DYNAMIC_VINS_SPEED_FACTOR_H
