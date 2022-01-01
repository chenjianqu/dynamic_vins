//
// Created by chen on 2021/10/11.
//

#ifndef DYNAMIC_VINS_PROJECTIONBOXFACTOR_H
#define DYNAMIC_VINS_PROJECTIONBOXFACTOR_H


#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <sophus/so3.hpp>

#include "../parameters.h"
#include "../utils.h"


/**
 * 维度:<误差项，IMU位姿,外参,物体位姿,box,逆深度>
 */
class ProjBoxFactor: public ceres::SizedCostFunction<3, 7,7,7,3,1>{
public:
    ProjBoxFactor(const Vec3d &pts_j_, const Eigen::Vector2d &velocity_j_, const double td_j_, const double curr_td_):
    pts_j(pts_j_),velocity_j(velocity_j_.x(),velocity_j_.y(),0),td_j(td_j_),curr_td(curr_td_){}
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;


    Vec3d pts_j;//估计值（以前的观测值）
    Vec3d velocity_j;
    double td_j,curr_td;

    static double sum_t;
};


class ProjBoxSimpleFactor: public ceres::SizedCostFunction<3,7,7,3,1>{
public:
    ProjBoxSimpleFactor(const Vec3d &pts_j_, const Eigen::Vector2d &velocity_j_, const double td_j_, const double curr_td_, Mat3d &R, Vec3d &P):
    pts_j(pts_j_),velocity_j(velocity_j_.x(),velocity_j_.y(),0),td_j(td_j_),curr_td(curr_td_){
        R_woj=R;
        P_woj=P;
    }
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;


    Vec3d pts_j;//估计值（以前的观测值）
    Vec3d velocity_j;
    double td_j,curr_td;

    Mat3d R_woj;
    Vec3d P_woj;

    static double sum_t;
};

/**
 * 优化变量：物体位姿、包围框、逆深度
 */
class BoxSqrtFactor: public ceres::SizedCostFunction<3,7,3,1>{
public:
    BoxSqrtFactor(const Vec3d &pts_j_,const Eigen::Vector2d &velocity_j_,
              const Mat3d &R_wbj_,const Vec3d &P_wbj_,
              const Mat3d &R_bc_,const Vec3d &P_bc_,
              const double td_j_,const double curr_td_):
    pts_j(pts_j_),velocity_j(velocity_j_.x(),velocity_j_.y(),0),td_j(td_j_),curr_td(curr_td_){
        R_wbj=R_wbj_;
        P_wbj=P_wbj_;
        R_bc=R_bc_;
        P_bc=P_bc_;
    }


    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;


    Vec3d pts_j;//估计值（以前的观测值）
    Vec3d velocity_j;
    double td_j,curr_td;

    static double sum_t;

    Mat3d R_wbj,R_bc;
    Vec3d P_wbj,P_bc;
};



class BoxAbsFactor: public ceres::SizedCostFunction<3,7,3,1>{
public:
    BoxAbsFactor(const Vec3d &pts_j_,const Eigen::Vector2d &velocity_j_,
              const Mat3d &R_wbj_,const Vec3d &P_wbj_,
              const Mat3d &R_bc_,const Vec3d &P_bc_,
              const double td_j_,const double curr_td_):
              pts_j(pts_j_),velocity_j(velocity_j_.x(),velocity_j_.y(),0),td_j(td_j_),curr_td(curr_td_){
        R_wbj=R_wbj_;
        P_wbj=P_wbj_;
        R_bc=R_bc_;
        P_bc=P_bc_;
    }

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_j;//估计值（以前的观测值）
    Vec3d velocity_j;
    double td_j,curr_td;

    static double sum_t;

    Mat3d R_wbj,R_bc;
    Vec3d P_wbj,P_bc;
};


class BoxPowFactor: public ceres::SizedCostFunction<3,3,1>{
public:
    BoxPowFactor(const Vec3d &pts_j_,const Eigen::Vector2d &velocity_j_,
                 const Mat3d &R_wbj_,const Vec3d &P_wbj_,
                 const Mat3d &R_bc_,const Vec3d &P_bc_,
                 const Mat3d &R_woj_,const Vec3d &P_woj_,
                 const double td_j_,const double curr_td_):
                 pts_j(pts_j_),velocity_j(velocity_j_.x(),velocity_j_.y(),0),td_j(td_j_),curr_td(curr_td_){
        R_wbj=R_wbj_;
        P_wbj=P_wbj_;
        R_bc=R_bc_;
        P_bc=P_bc_;
        R_woj=R_wbj_;
        P_woj=P_wbj_;
    }


    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;


    Vec3d pts_j;//估计值（以前的观测值）
    Vec3d velocity_j;
    double td_j,curr_td;

    static double sum_t;

    Mat3d R_wbj,R_bc,R_woj;
    Vec3d P_wbj,P_bc,P_woj;
};




#endif //DYNAMIC_VINS_PROJECTIONBOXFACTOR_H
