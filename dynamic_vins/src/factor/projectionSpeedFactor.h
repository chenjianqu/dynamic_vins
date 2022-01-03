/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_PROJECTIONSPEEDFACTOR_H
#define DYNAMIC_VINS_PROJECTIONSPEEDFACTOR_H



#include <ceres/ceres.h>
#include <Eigen/Dense>

#include <sophus/so3.hpp>


#include "../parameters.h"
#include "../utils.h"
#include "../estimator/dynamic.h"



//优化变量:Twbj 7,Tbc 7,Twoj 7,Twoi 7,speed 6 ,逆深度 1,
class ProjectionInstanceSpeedFactor: public ceres::SizedCostFunction<3, 7,7,7,7,6,1>{
public:
    ProjectionInstanceSpeedFactor(const Eigen::Vector3d &pts_j_,double time_j_,double time_i_):pts_j(pts_j_),time_ij(time_i_-time_j_)
    {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Eigen::Vector3d pts_j;//估计值（以前的观测值）
    double time_ij;

    static Eigen::Matrix3d sqrt_info;
    static double sum_t;
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
    ProjectionSpeedFactor(const Eigen::Vector3d &_pts_j, const Eigen::Vector3d &_pts_i,
                          const Eigen::Vector2d &_velocity_j, const Eigen::Vector2d &_velocity_i,
                          const Eigen::Matrix3d &R_bc1_, const Eigen::Vector3d &P_bc1_,
                          const Eigen::Matrix3d &R_bc2_, const Eigen::Vector3d &P_bc2_,
                          const double _td_j, const double _td_i,const double cur_td_,
                          double time_j_,double time_i_,double factor_)
                          :pts_j(_pts_j),pts_i(_pts_i),
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
    }


    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;


    Eigen::Vector3d pts_j,pts_i;//估计值（以前的观测值）
    double time_ij;

    Eigen::Vector3d velocity_i, velocity_j;
    double td_i, td_j,cur_td;

    static Eigen::Matrix2d sqrt_info;
    static double sum_t;

    Eigen::Matrix3d R_bc1,R_bc2;
    Eigen::Vector3d P_bc1,P_bc2;

    double factor;
};





/**
 * 维度:<误差项、物体的运动速度、逆深度>
 */
class ProjectionSpeedSimpleFactor: public ceres::SizedCostFunction<2,6,1>{
public:
    ProjectionSpeedSimpleFactor(const Eigen::Vector3d &_pts_j, const Eigen::Vector3d &_pts_i,
                                const Eigen::Vector2d &_velocity_j, const Eigen::Vector2d &_velocity_i,
                                const double _td_j, const double _td_i, const double cur_td_, double time_j_, double time_i_,
                                const Eigen::Matrix3d &R_wbj_, const Eigen::Vector3d &P_wbj_,
                                const Eigen::Matrix3d &R_wbi_, const Eigen::Vector3d &P_wbi_,
                                const Eigen::Matrix3d &R_bc1_, const Eigen::Vector3d &P_bc1_,
                                const Eigen::Matrix3d &R_bc2_, const Eigen::Vector3d &P_bc2_,
                                const double factor_)
                                :pts_j(_pts_j),pts_i(_pts_i),time_ij(time_i_-time_j_),
                                td_i(_td_i), td_j(_td_j),cur_td(cur_td_),factor(factor_),
                                R_wbj(R_wbj_),R_wbi(R_wbi_),P_wbi(P_wbi_),P_wbj(P_wbj_),
                                P_bc1(P_bc1_),R_bc1(R_bc1_),P_bc2(P_bc2_),R_bc2(R_bc2_){
        velocity_i.x() = _velocity_i.x();
        velocity_i.y() = _velocity_i.y();
        velocity_i.z() = 0;
        velocity_j.x() = _velocity_j.x();
        velocity_j.y() = _velocity_j.y();
        velocity_j.z() = 0;

    }


    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Eigen::Vector3d pts_j,pts_i;//估计值（以前的观测值）
    double time_ij;

    Eigen::Vector3d velocity_i, velocity_j;
    double td_i, td_j,cur_td;

    static Eigen::Matrix2d sqrt_info;
    static double sum_t;

    Eigen::Matrix3d R_wbj,R_wbi,R_bc1,R_bc2;
    Eigen::Vector3d P_wbj,P_wbi,P_bc1,P_bc2;

    double factor;

};




class SpeedPoseFactor: public ceres::SizedCostFunction<3, 7,7,7,7,6,1>{
public:
    SpeedPoseFactor(const Eigen::Vector3d &pts_j_, double time_j_, double time_i_):
    pts_j(pts_j_),time_ij(time_i_-time_j_)
    {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Eigen::Vector3d pts_j;//估计值（以前的观测值）
    double time_ij;

    static Eigen::Matrix3d sqrt_info;
    static double sum_t;
};


class SpeedPoseSimpleFactor: public ceres::SizedCostFunction<3, 7,7,6,1>{
public:
    SpeedPoseSimpleFactor(const Eigen::Vector3d &pts_j_, double time_j_, double time_i_, Eigen::Matrix3d &R_wbj_, Eigen::Vector3d &P_wbj_, Eigen::Matrix3d &R_bc_,
                          Eigen::Vector3d &P_bc_,const Eigen::Vector2d &vel_j_,const double td_j_, const double cur_td_):
    pts_j(pts_j_),time_ij(time_i_-time_j_),td(cur_td_),td_j(td_j_)
    {
        R_wbj=R_wbj_;
        R_bc=R_bc_;
        P_wbj=P_wbj_;
        P_bc=P_bc_;
        vel_j.x()=vel_j_.x();
        vel_j.y()=vel_j_.y();
        vel_j.z()=1;

    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    Eigen::Vector3d pts_j;//估计值（以前的观测值）
    double time_ij;

    static Eigen::Matrix3d sqrt_info;
    static double sum_t;

    Eigen::Matrix3d R_wbj,R_bc;
    Eigen::Vector3d P_wbj, P_bc;

    Eigen::Vector3d vel_j;
    double td,td_j;
};



class ConstSpeedFactor: public ceres::SizedCostFunction<3,6>{
public:
    ConstSpeedFactor(const Eigen::Vector3d &pts_j_, double time_j_, double time_i_, Eigen::Matrix3d &R_wbj_,
                     Eigen::Vector3d &P_wbj_, Eigen::Matrix3d &R_bc_, Eigen::Vector3d &P_bc_, double inv_depth_,
                     Eigen::Vector3d &last_v_, Eigen::Vector3d &last_a_)
                                       :
    pts_j(pts_j_),time_ij(time_i_-time_j_)
    {
        R_wbj=R_wbj_;
        R_bc=R_bc_;
        P_wbj=P_wbj_;
        P_bc=P_bc_;
        inv_depth=inv_depth_;
        last_v=last_v_;
        last_a=last_a_;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;


    Eigen::Vector3d pts_j;//估计值（以前的观测值）
    double time_ij;

    static Eigen::Matrix3d sqrt_info;
    static double sum_t;

    Eigen::Matrix3d R_wbj,R_bc;
    Eigen::Vector3d P_wbj, P_bc;

    double inv_depth;

    Eigen::Vector3d last_v,last_a;
};


class ConstSpeedSimpleFactor: public ceres::SizedCostFunction<1,6>{
public:
    ConstSpeedSimpleFactor(const Eigen::Vector3d &last_v_, const Eigen::Vector3d &last_a_, double factor_){
        last_v=last_v_;
        last_a=last_a_;
        factor=factor_;
    }

    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    static Eigen::Matrix3d sqrt_info;
    static double sum_t;

    Eigen::Vector3d last_v,last_a;
    double factor;
};






#endif //DYNAMIC_VINS_PROJECTIONSPEEDFACTOR_H
