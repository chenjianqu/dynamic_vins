/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_PROJECT_INSTANCE_FACTOR_H
#define DYNAMIC_VINS_PROJECT_INSTANCE_FACTOR_H

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <sophus/so3.hpp>
#include <utility>

#include "utils/def.h"
#include "utils/parameters.h"

namespace dynamic_vins{\


/**
 * 两帧之间某个观测的重投影误差
 * 参数：ceres::SizedCostFunction<误差项大小,P_wbj|Q_wbj, P_wbi|Q_wbi, P_bc|Q_bc, P_woj|Q_woj, P_woi|Q_woi,inv_dep_j>
 */
class ProjectionInstanceFactor : public ceres::SizedCostFunction<2, 7,7,7,7,7,1>{
public:
    ProjectionInstanceFactor(Vec3d _pts_j, Vec3d _pts_i)
    :pts_j(std::move(_pts_j)), pts_i(std::move(_pts_i)){}

    ProjectionInstanceFactor(Vec3d _pts_j, Vec3d _pts_i,
                             const Vec2d &_velocity_j, const Vec2d &_velocity_i,
                             const double _td_j, const double _td_i,const double cur_td_) :
                             pts_j(std::move(_pts_j)),pts_i(std::move(_pts_i)),
                             td_i(_td_i), td_j(_td_j),cur_td(cur_td_){
        velocity_i.x() = _velocity_i.x();
        velocity_i.y() = _velocity_i.y();
        velocity_i.z() = 0;
        velocity_j.x() = _velocity_j.x();
        velocity_j.y() = _velocity_j.y();
        velocity_j.z() = 0;

        sqrt_info = kFocalLength / 1.5 * Mat2d::Identity();
    };

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_j;//估计值（以前的观测值）
    Vec3d pts_i;//当前观测值

    Vec3d velocity_i, velocity_j;
    double td_i{}, td_j{},cur_td{};
    inline static Mat2d sqrt_info;
    inline static double sum_t{0};
};

/**
 * 维度：<误差项大小,外参1,外参2,逆深度>
 */
class ProjInst12Factor : public ceres::SizedCostFunction<2, 7, 7, 1>{
public:
    ProjInst12Factor(Vec3d _pts_j, Vec3d _pts_i) :
    pts_i(std::move(_pts_i)), pts_j(std::move(_pts_j)){
        sqrt_info = kFocalLength / 1.5 * Mat2d::Identity();
    };

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_i, pts_j;
    inline static Eigen::Matrix2d sqrt_info;
    inline static double sum_t{0};
    static int debug_num;
};

/**
 * 维度：<误差项大小,逆深度>
 */
class ProjInst12FactorSimple : public ceres::SizedCostFunction<2,1>{
public:
    ProjInst12FactorSimple(Vec3d _pts_j, Vec3d _pts_i,
                           Mat3d R_bc0_, Vec3d P_bc0_,
                           Mat3d R_bc1_, Vec3d P_bc1_
                           ) :pts_i(std::move(_pts_i)), pts_j(std::move(_pts_j)),R_bc0(std::move(R_bc0_)),R_bc1(std::move(R_bc1_)),
                           P_bc0(std::move(P_bc0_)),P_bc1(std::move(P_bc1_)){
        sqrt_info = kFocalLength / 1.5 * Mat2d::Identity();

    };

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_i, pts_j;

    Mat3d R_bc0,R_bc1;
    Vec3d P_bc0,P_bc1;

    inline static Eigen::Matrix2d sqrt_info;
    inline static double sum_t{0};
    static int debug_num;
};


/**
 * 维度：<误差项大小,IMU位姿1,IMU位姿2,外参1,物体位姿1,物体位姿2,逆深度,td>
 */
class ProjInst21Factor : public ceres::SizedCostFunction<2, 7, 7, 7,7,7, 1>{
public:
    ProjInst21Factor(Vec3d _pts_j, Vec3d _pts_i,
                     const Vec2d &_velocity_j, const Vec2d &_velocity_i,
                     const double _td_j, const double _td_i, const double cur_td_, int landmark_id) :
                     pts_i(std::move(_pts_i)), pts_j(std::move(_pts_j)),
                     td_i(_td_i), td_j(_td_j),cur_td(cur_td_){
        velocity_i.x() = _velocity_i.x();
        velocity_i.y() = _velocity_i.y();
        velocity_i.z() = 0;
        velocity_j.x() = _velocity_j.x();
        velocity_j.y() = _velocity_j.y();
        velocity_j.z() = 0;

        debug_num=0;
        id=landmark_id;

        sqrt_info = kFocalLength / 1.5 * Mat2d::Identity();

    };

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_i, pts_j;
    Vec3d velocity_i, velocity_j;
    double td_i, td_j,cur_td;
    inline static Eigen::Matrix2d sqrt_info;
    inline static double sum_t{0};

    static int debug_num;

    int id;

};



class ProjInst21SimpleFactor : public ceres::SizedCostFunction<2,7,7, 1>{
public:
    ProjInst21SimpleFactor(Vec3d _pts_j, Vec3d _pts_i,
                           const Vec2d &_velocity_j, const Vec2d &_velocity_i,
                           const Mat3d &R_wbj_, const Vec3d &P_wbj_,
                           const Mat3d &R_wbi_, const Vec3d &P_wbi_,
                           const Mat3d &R_bc_, const Vec3d &P_bc_,
                           const double _td_j, const double _td_i, const double cur_td_, int landmark_id) :
                           pts_i(std::move(_pts_i)), pts_j(std::move(_pts_j)),
                           td_i(_td_i), td_j(_td_j),cur_td(cur_td_){
        velocity_i.x() = _velocity_i.x();
        velocity_i.y() = _velocity_i.y();
        velocity_i.z() = 0;
        velocity_j.x() = _velocity_j.x();
        velocity_j.y() = _velocity_j.y();
        velocity_j.z() = 0;

        id=landmark_id;

        R_wbj=R_wbj_;
        P_wbj=P_wbj_;
        R_wbi=R_wbi_;
        P_wbi=P_wbi_;
        R_bc=R_bc_;
        P_bc=P_bc_;

        sqrt_info = kFocalLength / 1.5 * Mat2d::Identity();
    };

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_i, pts_j;
    Vec3d velocity_i, velocity_j;
    double td_i, td_j,cur_td;
    inline static Eigen::Matrix2d sqrt_info;
    inline static double sum_t{0};

    int id;

    Mat3d R_wbj,R_wbi,R_bc;
    Vec3d P_wbj,P_wbi,P_bc;
};




/**
 * 维度:<误差项大小,,IMU位姿1,IMU位姿2,外参1,外参2,物体位姿1,物体位姿2,逆深度,td>
 */
class ProjInst22Factor : public ceres::SizedCostFunction<2, 7, 7,   7,7,  7,7, 1>{
public:
    ProjInst22Factor(Vec3d _pts_j, Vec3d _pts_i,
                     const Vec2d &_velocity_j, const Vec2d &_velocity_i,
                     const double _td_j, const double _td_i, const double cur_td_) :
                     pts_i(std::move(_pts_i)), pts_j(std::move(_pts_j)),
                     td_i(_td_i), td_j(_td_j),cur_td(cur_td_){
        velocity_i.x() = _velocity_i.x();
        velocity_i.y() = _velocity_i.y();
        velocity_i.z() = 0;
        velocity_j.x() = _velocity_j.x();
        velocity_j.y() = _velocity_j.y();
        velocity_j.z() = 0;

        sqrt_info = kFocalLength / 1.5 * Mat2d::Identity();
    };

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_i, pts_j;
    Vec3d velocity_i, velocity_j;
    double td_i, td_j,cur_td;
    inline static Eigen::Matrix2d sqrt_info;
    inline static double sum_t{0};
};


class ProjInst22SimpleFactor : public ceres::SizedCostFunction<2,7,7, 1>{
public:
    ProjInst22SimpleFactor(Vec3d _pts_j, Vec3d _pts_i,
                           const Vec2d &_velocity_j, const Vec2d &_velocity_i,
                           const Mat3d &R_wbj_, const Vec3d &P_wbj_,
                           const Mat3d &R_wbi_, const Vec3d &P_wbi_,
                           const Mat3d &R_bc1_, const Vec3d &P_bc1_,
                           const Mat3d &R_bc2_, const Vec3d &P_bc2_,
                           const double _td_j, const double _td_i, const double cur_td_, int landmark_id) :
                           pts_i(std::move(_pts_i)), pts_j(std::move(_pts_j)),
                           td_i(_td_i), td_j(_td_j),cur_td(cur_td_){
        velocity_i.x() = _velocity_i.x();
        velocity_i.y() = _velocity_i.y();
        velocity_i.z() = 0;
        velocity_j.x() = _velocity_j.x();
        velocity_j.y() = _velocity_j.y();
        velocity_j.z() = 0;

        id=landmark_id;

        R_wbj=R_wbj_;
        P_wbj=P_wbj_;
        R_wbi=R_wbi_;
        P_wbi=P_wbi_;
        R_bc1=R_bc1_;P_bc1=P_bc1_;
        R_bc2=R_bc2_;P_bc2=P_bc2_;

        sqrt_info = kFocalLength / 1.5 * Mat2d::Identity();
    };

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_i, pts_j;
    Vec3d velocity_i, velocity_j;
    double td_i, td_j,cur_td;
    inline static Eigen::Matrix2d sqrt_info;
    inline static double sum_t{0};

    int id;

    Mat3d R_wbj,R_wbi,R_bc1,R_bc2;
    Vec3d P_wbj,P_wbi,P_bc1,P_bc2;
};


class InstancePositionFactor : public ceres::SizedCostFunction<1,6>{
public:
    InstancePositionFactor(const Vec3d &_pts_j,const Mat3d &R_wbj_,const Vec3d &P_wbj_,
                           const Mat3d &R_bc_,const Vec3d &P_bc_,const double depth_){
        pts_j=_pts_j;
        R_wbj=R_wbj_;
        P_wbj=P_wbj_;
        R_bc=R_bc_;
        P_bc=P_bc_;
        depth=depth_;
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_j;

    Mat3d R_wbj,R_bc;
    Vec3d P_wbj,P_bc;

    double depth;
};





class InstanceInitAbsFactor: public ceres::SizedCostFunction<3,7,1>{
public:
    InstanceInitAbsFactor(Vec3d pts_j_,const Vec2d &velocity_j_,
                          const Mat3d &R_wbj_,const Vec3d &P_wbj_,
                          const Mat3d &R_bc_,const Vec3d &P_bc_,
                          const double td_j_,const double curr_td_):
                          pts_j(std::move(pts_j_)),velocity_j(velocity_j_.x(),velocity_j_.y(),0),td_j(td_j_),curr_td(curr_td_){
        R_wbj=R_wbj_;
        P_wbj=P_wbj_;
        R_bc=R_bc_;
        P_bc=P_bc_;
    }


    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;


    Vec3d pts_j;//估计值（以前的观测值）
    Vec3d velocity_j;
    double td_j,curr_td;

    inline static double sum_t{0};

    Mat3d R_wbj,R_bc;
    Vec3d P_wbj,P_bc;
};


/**
 * 物体的位姿(P_woj) 与 物体中路标点的世界坐标(pts_w_j) 的平方误差
 */
class InstanceInitPowFactor: public ceres::SizedCostFunction<3,7,1>{
public:
    InstanceInitPowFactor(Vec3d pts_j_,const Vec2d &velocity_j_,
                          const Mat3d &R_wbj_,const Vec3d &P_wbj_,
                          const Mat3d &R_bc_,const Vec3d &P_bc_,
                          const double td_j_,const double curr_td_,double factor_=0.01):
                          pts_j(std::move(pts_j_)),velocity_j(velocity_j_.x(),velocity_j_.y(),0),td_j(td_j_),curr_td(curr_td_),factor(factor_){
        R_wbj=R_wbj_;
        P_wbj=P_wbj_;
        R_bc=R_bc_;
        P_bc=P_bc_;
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_j;//估计值（以前的观测值）
    Vec3d velocity_j;
    double td_j,curr_td;
    double factor;

    inline static double sum_t{0};

    Mat3d R_wbj,R_bc;
    Vec3d P_wbj,P_bc;
};

/**
 * 误差维度：3
 * 优化项：线速度、角速度、参考位姿、逆深度
 */
class InstanceInitPowFactorSpeed: public ceres::SizedCostFunction<3,7,6,1>{
public:
    InstanceInitPowFactorSpeed(Vec3d pts_j_,const Vec2d &velocity_j_,
                               const Mat3d &R_wbj_,const Vec3d &P_wbj_,
                               const Mat3d &R_bc_,const Vec3d &P_bc_,
                               const double td_j_,const double curr_td_,
                               const double time_j,const double time_s,double factor_=0.01):
                               pts_j(std::move(pts_j_)),velocity_j(velocity_j_.x(),velocity_j_.y(),0),td_j(td_j_),curr_td(curr_td_),
                               time_js(time_j- time_s),factor(factor_){
        R_wbj=R_wbj_;
        P_wbj=P_wbj_;
        R_bc=R_bc_;
        P_bc=P_bc_;
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_j;//估计值（以前的观测值）
    Vec3d velocity_j;
    double td_j,curr_td;

    double time_js;

    double factor;

    inline static double sum_t{0};

    Mat3d R_wbj,R_bc;
    Vec3d P_wbj,P_bc;
};


}

#endif //DYNAMIC_VINS_PROJECT_INSTANCE_FACTOR_H
