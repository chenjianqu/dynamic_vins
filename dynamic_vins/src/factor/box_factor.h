/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_BOX_FACTOR_H
#define DYNAMIC_VINS_BOX_FACTOR_H


#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <sophus/so3.hpp>
#include <utility>

#include "utils/def.h"

namespace dynamic_vins{\

/*
//维度:<误差项，IMU位姿,外参,物体位姿,box,逆深度>
class ProjBoxFactor: public ceres::SizedCostFunction<3, 7,7,7,3,1>{
public:
    ProjBoxFactor(const Vec3d &pts_j_, const Eigen::Vector2d &velocity_j_, const double td_j_, const double curr_td_):
    pts_j(pts_j_),velocity_j(velocity_j_.x(),velocity_j_.y(),0),td_j(td_j_),curr_td(curr_td_){}
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;


    Vec3d pts_j;//估计值（以前的观测值）
    Vec3d velocity_j;
    double td_j,curr_td;

    inline static double sum_t;
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

    inline static double sum_t;
};

//优化变量：物体位姿、包围框、逆深度
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

    inline static double sum_t;

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

    inline static double sum_t;

    Mat3d R_wbj,R_bc,R_woj;
    Vec3d P_wbj,P_bc,P_woj;
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

    inline static double sum_t;

    Mat3d R_wbj,R_bc;
    Vec3d P_wbj,P_bc;
};*/


/**
 * 双目三角化得到的3D点应该要落在包围框内
 * 误差维度3, 优化变量：物体位姿、包围框
 */
class BoxEncloseStereoPointFactor: public ceres::SizedCostFunction<3,7,3>{
public:
    explicit BoxEncloseStereoPointFactor(Vec3d &point_w): pts_w(point_w){}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Vec3d pts_w;//3D点

    inline static int counter{0};

};

/**
 * 帧间三角化得到的3D点应该要落在包围框内
 * 误差维度3, 优化变量：物体位姿、包围框,逆深度
 */
class BoxEncloseTrianglePointFactor: public ceres::SizedCostFunction<3,7,3,1>{
public:
    BoxEncloseTrianglePointFactor(Vec3d pts_j_,const Vec2d &velocity_j_,
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

    Mat3d R_wbj,R_bc;
    Vec3d P_wbj,P_bc;
};


/**
 *
 * 包围框的顶点的欧氏距离误差,
 * 优化变量: 物体位姿7,包围框3
 */
class BoxVertexFactor: public ceres::SizedCostFunction<3,7,3>{
public:
    BoxVertexFactor(Mat38d corners_,Vec3d dims_,Vec3d indicate_symbol_,Mat3d R_bc_,Mat3d R_wbi_)
    : corners(std::move(corners_)),dims(std::move(dims_)),indicate_symbol(std::move(indicate_symbol_)),R_bc(std::move(R_bc_)),R_wbi(std::move(R_wbi_)){}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Mat38d corners;//观测的物体的长宽高
    Vec3d dims;
    Vec3d indicate_symbol;//指示该点属于哪个顶点
    Mat3d R_bc,R_wbi;

    inline static int counter{0};
};


/**
 * 包围框大小的直接误差
 * 优化变量:包围框的大小3
 */
class BoxDimsFactor:public ceres::SizedCostFunction<1,3>{
public:
    explicit BoxDimsFactor(Vec3d dims_):dims(std::move(dims_)){}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;


    Vec3d dims;
};


/**
 * 定义的orientation误差,
 * 优化变量 相机位姿7, 物体位姿7.
 * note:虽然公式里只关于旋转的误差,但由于参数块与position放在一起,故这里定义变量的global维度为7
 */
class BoxOrientationFactor:public ceres::SizedCostFunction<3,7,7>{
public:
    BoxOrientationFactor(Mat3d R_cioi_,Mat3d R_bc_):R_cioi(std::move(R_cioi_)),R_bc(std::move(R_bc_)){}

    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    Mat3d R_cioi;
    Mat3d R_bc;

    inline static int counter{0};
};




}


#endif //DYNAMIC_VINS_BOX_FACTOR_H
