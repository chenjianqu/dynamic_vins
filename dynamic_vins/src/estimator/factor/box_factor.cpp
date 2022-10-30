/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "box_factor.h"
#include "utils/log_utils.h"
#include "utils/box3d.h"
#include "utils/io_utils.h"

namespace dynamic_vins{\



template<typename T>
Mat3d hat(T && v){
    return Sophus::SO3d::hat(std::forward<T>(v)); //完美转发
}

template<typename T>
Vec3d vee(T && m){
    return Sophus::SO3d::vee(m);
}

using namespace Sophus;



/*
 //优化变量:Twbj 7,Tbc 7,Twoj 7,box 3 ,逆深度 1,

bool ProjBoxFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;

    Vec3d P_wbj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_wbj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_bc(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_bc(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vec3d P_woj(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quatd Q_woj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    double box_x=parameters[3][0],box_y=parameters[3][1],box_z=parameters[3][2];
    double inv_dep_j = parameters[4][0];

    Vec3d pts_j_td;
    pts_j_td = pts_j - (curr_td - td_j) * velocity_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=Q_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=Q_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=Q_woj.inverse()*(pts_w_j-P_woj);//k点在j时刻的物体坐标

    double pts_abs_x=std::abs(pts_obj_j.x());
    double pts_abs_y=std::abs(pts_obj_j.y());
    double pts_abs_z=std::abs(pts_obj_j.z());

    residuals[0]=(pts_abs_x-box_x)*(pts_abs_x-box_x);
    residuals[1]=(pts_abs_y-box_y)*(pts_abs_y-box_y);
    residuals[2]=(pts_abs_z-box_z)*(pts_abs_z-box_z);


    if (jacobians)
    {
        Mat3d R_wbj = Q_wbj.toRotationMatrix();
        Mat3d R_bc = Q_bc.toRotationMatrix();
        Mat3d R_woj = Q_woj.toRotationMatrix();
        Mat3d R_ojw=R_woj.transpose();

        double reduce11=2* (pts_abs_x - box_x) * pts_obj_j.x() / pts_abs_x;
        double reduce22=2* (pts_abs_y - box_y) * pts_obj_j.y() / pts_abs_y;
        double reduce33=2* (pts_abs_z - box_z) * pts_obj_j.z() / pts_abs_z;

        ///d(e)/d(pts_oj)
        Mat3d reduce=Mat3d::Zero();
        reduce(0,0)=reduce11;
        reduce(1,1)=reduce22;
        reduce(2,2)=reduce33;


        /// 计算雅可比矩阵
        //计算残差相对于p_bj和q_bj的雅可比
        if (jacobians[0])
        {
            Mat36d jaco_j;
            //P_ci相对于p_wbj的导数
            jaco_j.leftCols<3>() = R_ojw;
            //P_ci相对于q_wbj的导数
            jaco_j.rightCols<3>() = -(R_ojw *R_wbj * hat(pts_imu_j));

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[0]);
            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[1])
        {
            Mat36d jaco_ex;
            //P_ci相对于p_wbj的导数
            jaco_ex.leftCols<3>() = R_ojw*R_wbj;
            //P_ci相对于q_wbj的导数
            jaco_ex.rightCols<3>() = -(R_ojw*R_wbj*R_bc * hat(pts_imu_j));

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Mat36d jaco_oj;
            jaco_oj.leftCols<3>() = -R_ojw;
            jaco_oj.rightCols<3>() = hat(R_ojw * (pts_w_j - P_woj));

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[2]);
            jacobian_pose_oj.leftCols<6>() = reduce * jaco_oj;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
            double box11= -2 * (pts_abs_x - box_x);
            double box22= -2 * (pts_abs_y - box_y);
            double box33= -2 * (pts_abs_z - box_z);

            Mat3d jaco_box=Eigen::Matrix<double,3,3>::Zero();
            jaco_box(0,0)=box11;
            jaco_box(1,1)=box22;
            jaco_box(2,2)=box33;

            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_box(jacobians[3]);
            jacobian_box=jaco_box;
        }
        if(jacobians[4])
        {
            Eigen::Map<Vec3d> jacobian_feature(jacobians[4]);
            jacobian_feature = -reduce *  R_ojw * R_wbj * R_bc * pts_j / (inv_dep_j*inv_dep_j);
        }
    }

    sum_t += tic_toc.Toc();

    return true;

}



bool ProjBoxSimpleFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;

    Vec3d P_wbj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_wbj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_bc(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_bc(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    double box_x=parameters[2][0],box_y=parameters[2][1],box_z=parameters[2][2];
    double inv_dep_j = parameters[3][0];

    Vec3d pts_j_td= pts_j - (curr_td - td_j) * velocity_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=Q_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=Q_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=R_woj.inverse()*(pts_w_j-P_woj);//k点在j时刻的物体坐标

    double pts_abs_x=std::abs(pts_obj_j.x());
    double pts_abs_y=std::abs(pts_obj_j.y());
    double pts_abs_z=std::abs(pts_obj_j.z());

    ///乘以一个系数，以减小误差，使得每次box改变的幅度变小
    const double ALPHA=0.1;

    residuals[0]=std::sqrt(std::abs(pts_abs_x-box_x)* ALPHA ) ;//(pts_abs_x-box_x) * ALPHA;
    residuals[1]=std::sqrt(std::abs(pts_abs_y-box_y)* ALPHA );//(pts_abs_y-box_y) * ALPHA;
    residuals[2]=std::sqrt(std::abs(pts_abs_z-box_z)* ALPHA );//(pts_abs_z-box_z) * ALPHA;


    if (jacobians)
    {
        Mat3d R_wbj = Q_wbj.toRotationMatrix();
        Mat3d R_bc = Q_bc.toRotationMatrix();
        Mat3d R_ojw=R_woj.transpose();

        double reduce11=2* (pts_abs_x - box_x) * pts_obj_j.x() / pts_abs_x;
        double reduce22=2* (pts_abs_y - box_y) * pts_obj_j.y() / pts_abs_y;
        double reduce33=2* (pts_abs_z - box_z) * pts_obj_j.z() / pts_abs_z;

        ///d(e)/d(pts_oj)
        Mat3d reduce=Mat3d::Zero();
        reduce(0,0)=reduce11;
        reduce(1,1)=reduce22;
        reduce(2,2)=reduce33;


        /// 计算雅可比矩阵
        //计算残差相对于p_bj和q_bj的雅可比
        if (jacobians[0])
        {
            Mat36d jaco_j;
            //P_ci相对于p_wbj的导数
            jaco_j.leftCols<3>() = R_ojw;
            //P_ci相对于q_wbj的导数
            jaco_j.rightCols<3>() = -(R_ojw *R_wbj * hat(pts_imu_j));

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[0]);
            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[1])
        {
            Mat36d jaco_ex;
            //P_ci相对于p_wbj的导数
            jaco_ex.leftCols<3>() = R_ojw*R_wbj;
            //P_ci相对于q_wbj的导数
            jaco_ex.rightCols<3>() = -(R_ojw*R_wbj*R_bc * hat(pts_imu_j));

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            double box11= -2 * (pts_abs_x - box_x);
            double box22= -2 * (pts_abs_y - box_y);
            double box33= -2 * (pts_abs_z - box_z);

            Mat3d jaco_box=Eigen::Matrix<double,3,3>::Zero();
            jaco_box(0,0)=box11;
            jaco_box(1,1)=box22;
            jaco_box(2,2)=box33;

            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_box(jacobians[2]);
            jacobian_box=jaco_box;
        }
        if(jacobians[3])
        {
            Eigen::Map<Vec3d> jacobian_feature(jacobians[3]);
            //jacobian_feature = -reduce *  R_ojw * R_wbj * R_bc * pts_j / (inv_dep_j*inv_dep_j);
            jacobian_feature=Vec3d::Zero();
        }
    }


    sum_t += tic_toc.Toc();

    return true;

}


bool BoxPowFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    double box_x=parameters[0][0],box_y=parameters[0][1],box_z=parameters[0][2];
    double inv_dep_j = parameters[1][0];

    Vec3d pts_j_td;
    pts_j_td = pts_j - (curr_td - td_j) * velocity_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=R_woj.inverse()*(pts_w_j-P_woj);//k点在j时刻的物体坐标


    double sub_x= std::abs(pts_obj_j.x() ) - box_x;
    double sub_y= std::abs(pts_obj_j.y() ) - box_y;
    double sub_z= std::abs(pts_obj_j.z() ) - box_z;

    residuals[0]= sub_x*sub_x ;
    residuals[1]= sub_y*sub_y ;
    residuals[2]= sub_z*sub_z ;

    if (jacobians)
    {
        ///对包围框求导
        if (jacobians[0])
        {
            Mat3d jaco_box=Mat3d::Zero();
            jaco_box(0,0)=-2 * sub_x;
            jaco_box(1,1)=-2 * sub_y;
            jaco_box(2,2)=-2 * sub_z;

            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_box(jacobians[0]);
            jacobian_box=jaco_box;
        }
        if(jacobians[1])
        {
            Eigen::Map<Vec3d> jacobian_feature(jacobians[1]);
            //jacobian_feature = -reduce *  R_ojw * R_wbj * R_bc * pts_j / (inv_dep_j*inv_dep_j);
            jacobian_feature=Vec3d::Zero();
        }
    }

    sum_t += tic_toc.Toc();

    return true;

}


bool BoxSqrtFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d P_woj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    double box_x=parameters[1][0],box_y=parameters[1][1],box_z=parameters[1][2];
    double inv_dep_j = parameters[2][0];

    Vec3d pts_j_td;
    pts_j_td = pts_j - (curr_td - td_j) * velocity_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=Q_woj.inverse()*(pts_w_j-P_woj);//k点在j时刻的物体坐标


    double sub_x= std::abs(pts_obj_j.x() ) - box_x;
    double sub_y= std::abs(pts_obj_j.y() ) - box_y;
    double sub_z= std::abs(pts_obj_j.z() ) - box_z;
    double abs_sub_x =std::abs(sub_x);
    double abs_sub_y =std::abs(sub_y);
    double abs_sub_z =std::abs(sub_z);
    double sqrt_abs_x=std::sqrt(abs_sub_x);
    double sqrt_abs_y=std::sqrt(abs_sub_y);
    double sqrt_abs_z=std::sqrt(abs_sub_z);

    double sqa_x=std::sqrt(std::abs(P_woj.x() - pts_w_j.x()));
    double sqa_y=std::sqrt(std::abs(P_woj.y() - pts_w_j.y()));
    double sqa_z=std::sqrt(std::abs(P_woj.z() - pts_w_j.z()));

    residuals[0]= sqrt_abs_x + sqa_x;
    residuals[1]= sqrt_abs_y + sqa_y;
    residuals[2]= sqrt_abs_z + sqa_z;


    if (jacobians)
    {
        Mat3d R_woj = Q_woj.toRotationMatrix();
        Mat3d R_ojw=R_woj.transpose();

        double delta_abs_x= pts_obj_j.x() >= 0? 1. : -1;
        double delta_abs_y= pts_obj_j.y() >= 0? 1. : -1;
        double delta_abs_z= pts_obj_j.z() >= 0? 1. : -1;

        double delta_pts_x= sub_x >= 0 ? delta_abs_x : -delta_abs_x ;
        double delta_pts_y= sub_y >= 0 ? delta_abs_y : -delta_abs_y ;
        double delta_pts_z= sub_z >= 0 ? delta_abs_z : -delta_abs_z ;

        Mat3d reduce=Mat3d::Zero();
        reduce(0,0)= (1./(2*sqrt_abs_x)) * delta_pts_x;
        reduce(1,1)= (1./(2*sqrt_abs_y)) * delta_pts_y;
        reduce(2,2)= (1./(2*sqrt_abs_z)) * delta_pts_z;

        /// 对物体位姿求导
        if (jacobians[0])
        {
            double delta11= P_woj.x() >= pts_w_j.x() ? 1 : -1;
            double delta22= P_woj.y() >= pts_w_j.y() ? 1 : -1;
            double delta33= P_woj.z() >= pts_w_j.z() ? 1 : -1;

            Mat3d jaco_init=Mat3d::Zero();
            jaco_init(0,0) = delta11 / (2.*sqa_x);
            jaco_init(1,1) = delta22 / (2.*sqa_y);
            jaco_init(2,2) = delta33 / (2.*sqa_z);

            Mat36d jaco_oj;
            jaco_oj.leftCols<3>() = -R_ojw + jaco_init;
            jaco_oj.rightCols<3>() = Sophus::SO3d::exp(R_ojw * (pts_w_j - P_woj)).matrix();

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[0]);
            jacobian_pose_oj.leftCols<6>() = reduce * jaco_oj;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        ///对包围框求导
        if (jacobians[1])
        {
            double box11= (sub_x >= 0? -1 : 1) * 1./(2.*sqrt_abs_x);
            double box22= (sub_y >= 0? -1 : 1) * 1./(2.*sqrt_abs_y);
            double box33= (sub_z >= 0? -1 : 1) * 1./(2.*sqrt_abs_z);

            Mat3d jaco_box=Mat3d::Zero();
            jaco_box(0,0)=box11;
            jaco_box(1,1)=box22;
            jaco_box(2,2)=box33;

            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_box(jacobians[1]);
            jacobian_box = Mat3d::Zero();
        }
        if(jacobians[2])
        {
            Eigen::Map<Vec3d> jacobian_feature(jacobians[2]);
            //jacobian_feature = -reduce *  R_ojw * R_wbj * R_bc * pts_j / (inv_dep_j*inv_dep_j);
            jacobian_feature=Vec3d::Zero();
        }
    }

    sum_t += tic_toc.Toc();

    return true;

}


bool BoxAbsFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d P_woj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    double box_x=parameters[1][0],box_y=parameters[1][1],box_z=parameters[1][2];
    double inv_dep_j = parameters[2][0];

    Vec3d pts_j_td;
    pts_j_td = pts_j - (curr_td - td_j) * velocity_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=Q_woj.inverse()*(pts_w_j-P_woj);//k点在j时刻的物体坐标


    double sub_x= std::abs(pts_obj_j.x() ) - box_x;
    double sub_y= std::abs(pts_obj_j.y() ) - box_y;
    double sub_z= std::abs(pts_obj_j.z() ) - box_z;
    double abs_sub_x =std::abs(sub_x);
    double abs_sub_y =std::abs(sub_y);
    double abs_sub_z =std::abs(sub_z);

    double abs_oj_x=std::abs(P_woj.x() - pts_w_j.x());
    double abs_oj_y=std::abs(P_woj.y() - pts_w_j.y());
    double abs_oj_z=std::abs(P_woj.z() - pts_w_j.z());

    residuals[0]=abs_sub_x + abs_oj_x;
    residuals[1]= abs_sub_y + abs_oj_y;
    residuals[2]= abs_sub_z + abs_oj_z;

    if (jacobians)
    {
        Mat3d R_woj = Q_woj.toRotationMatrix();
        Mat3d R_ojw=R_woj.transpose();

        double delta_abs_x= pts_obj_j.x() >= 0? 1. : -1;
        double delta_abs_y= pts_obj_j.y() >= 0? 1. : -1;
        double delta_abs_z= pts_obj_j.z() >= 0? 1. : -1;

        double delta_pts_x= sub_x >= 0 ? delta_abs_x : -delta_abs_x ;
        double delta_pts_y= sub_y >= 0 ? delta_abs_y : -delta_abs_y ;
        double delta_pts_z= sub_z >= 0 ? delta_abs_z : -delta_abs_z ;

        Mat3d reduce=Mat3d::Zero();
        reduce(0,0)= sub_x * delta_pts_x;
        reduce(1,1)= sub_y * delta_pts_y;
        reduce(2,2)= sub_z * delta_pts_z;

        /// 对物体位姿求导
        if (jacobians[0])
        {
            double delta11= P_woj.x() >= pts_w_j.x() ? 1 : -1;
            double delta22= P_woj.y() >= pts_w_j.y() ? 1 : -1;
            double delta33= P_woj.z() >= pts_w_j.z() ? 1 : -1;

            Mat3d jaco_init=Mat3d::Zero();
            jaco_init(0,0) = delta11 *(P_woj.x()-pts_w_j.x());
            jaco_init(1,1) = delta22 *(P_woj.y()-pts_w_j.y());
            jaco_init(2,2) = delta33 *(P_woj.z()-pts_w_j.z());

            Mat36d jaco_oj;
            jaco_oj.leftCols<3>() = -R_ojw + jaco_init;
            jaco_oj.rightCols<3>() = Sophus::SO3d::exp(R_ojw * (pts_w_j - P_woj)).matrix();

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[0]);
            jacobian_pose_oj.leftCols<6>() = reduce * jaco_oj;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        ///对包围框求导
        if (jacobians[1])
        {
            double box11= (sub_x >= 0? -1 : 1) * (sub_x);
            double box22= (sub_y >= 0? -1 : 1) * (sub_y);
            double box33= (sub_z >= 0? -1 : 1) * (sub_z);

            Mat3d jaco_box=Mat3d::Zero();
            jaco_box(0,0)=box11;
            jaco_box(1,1)=box22;
            jaco_box(2,2)=box33;

            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_box(jacobians[1]);
            //jacobian_box=jaco_box;
            jacobian_box = Mat3d::Zero();
        }
        if(jacobians[2])
        {
            Eigen::Map<Vec3d> jacobian_feature(jacobians[2]);
            //jacobian_feature = -reduce *  R_ojw * R_wbj * R_bc * pts_j / (inv_dep_j*inv_dep_j);
            jacobian_feature=Vec3d::Zero();
        }
    }

    sum_t += tic_toc.Toc();

    return true;

}*/


/**
 * 误差计算和雅可比计算
 * @param parameters 顺序:物体位姿
 * @param residuals
 * @param jacobians
 * @return
 */
bool BoxEncloseStereoPointFactor::Evaluate(const double *const *parameters, double *residuals,
                                           double **jacobians) const{
    Vec3d P_woj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    ///误差计算
    Vec3d pts_obj_j=Q_woj.inverse()*(pts_w-P_woj);//k点在j时刻的物体坐标
    Vec3d abs_v=pts_obj_j.cwiseAbs();
    Vec3d vec_err = abs_v - dims/2;
    vec_err *=10;
    residuals[0]= std::max(0.,vec_err.x());
    residuals[1]= std::max(0.,vec_err.y());
    residuals[2]= std::max(0.,vec_err.z());

    counter++;
    string log_text;


    ///雅可比计算
    if (jacobians){

        /// 对物体位姿求导
        if (jacobians[0]){
            Mat3d R_ojw = Q_woj.inverse().matrix();
            //计算 指示矩阵
            Vec3d e =  R_ojw * (pts_obj_j - P_woj) ;
            Mat3d N_p = Mat3d::Zero();
            N_p(0,0) = e.x()/std::abs(e.x());
            N_p(1,1) = e.y()/std::abs(e.y());
            N_p(2,2) = e.z()/std::abs(e.z());

            /**
             * 注意,这里没有直接写出对max()函数的求导,这是因为已经对误差使用max()函数,这在更新时会间接使用了max函数
             */
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[0]);
            jacobian_pose_oj.leftCols<3>() = N_p * R_ojw; //对position的雅可比
            jacobian_pose_oj.middleCols(3,3).setZero();
            jacobian_pose_oj.rightCols<1>().setZero();

            /*if(counter%50==0){
                log_text += fmt::format("Evaluate 对物体位姿求导:\n{}\n", EigenToStr(jacobian_pose_oj));
            }*/

        }

    }


    ///Debug
    /*if(lid%20==0){
        log_text = fmt::format("Evaluate开始 lid:{} counter:{}\n",lid,counter);
        log_text += fmt::format("P_woj:{}, Q_woj:{},\n dims:{} pts_w:{}\n",
                                VecToStr(P_woj), QuaternionToStr(Q_woj),VecToStr(dims), VecToStr(pts_w));
        log_text += fmt::format("Evaluate p_obj:{} 误差:{}\n", VecToStr(abs_v), VecToStr(vec_err));
        log_text += fmt::format("Evaluate结束");

        WriteTextFile(MyLogger::kLogOutputDir + "FactorDebugMsg.txt",log_text);
    }*/


    return true;
}





bool BoxEncloseTrianglePointFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d P_woj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    Vec3d box(parameters[1][0],parameters[1][1],parameters[1][2]);
    double inv_dep_j = parameters[2][0];

    Vec3d pts_j_td = pts_j - (curr_td - td_j) * velocity_j;
    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=Q_woj.inverse()*(pts_w_j-P_woj);//k点在j时刻的物体坐标

    ///误差计算
    Vec3d vec_err = box - pts_obj_j.cwiseAbs();
    residuals[0]= std::max(0.,vec_err.x());
    residuals[1]= std::max(0.,vec_err.y());
    residuals[2]= std::max(0.,vec_err.z());

    ///雅可比计算
    if (jacobians){

        Mat3d R_ojw = Mat3d(Q_woj.inverse());
        //计算 指示矩阵
        Vec3d p_j = - R_ojw * P_woj;
        Mat3d N_p = Mat3d::Zero();
        N_p(0,0)= p_j.x() / std::abs(p_j.x());
        N_p(1,1)= p_j.y() / std::abs(p_j.y());
        N_p(2,2)= p_j.z() / std::abs(p_j.z());
        Vec3d pts_kl_oj = R_ojw * (pts_w_j - P_woj);
        Mat3d N_R = Mat3d::Zero();
        N_R(0,0)= pts_kl_oj.x() / std::abs(pts_kl_oj.x());
        N_R(1,1)= pts_kl_oj.y() / std::abs(pts_kl_oj.y());
        N_R(2,2)= pts_kl_oj.z() / std::abs(pts_kl_oj.z());

        /// 对物体位姿求导
        if (jacobians[0]){
            /**
             * 注意,这里没有直接写出对max()函数的求导,这是因为已经对误差使用max()函数,这在更新时会间接使用了max函数
             */
            Mat36d jaco_oj;
            jaco_oj.leftCols<3>() =  N_p * R_ojw; //对position的雅可比
            //jaco_oj.rightCols<3>() = - N_R * hat( R_ojw*(pts_w - P_woj) ).matrix();
            jaco_oj.rightCols<3>() = Mat3d::Zero();//对orientation的雅可比

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[0]);
            jacobian_pose_oj.leftCols<6>() =   jaco_oj;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        ///对包围框求导
        if (jacobians[1]){
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_box(jacobians[1]);
            //jacobian_box = - Mat3d::Identity();
            jacobian_box = Mat3d::Zero();
        }
        ///对逆深度求导
        if(jacobians[2]){
/*            Vec3d I_v = R_ojw * pts_w_j;
            Mat3d N_pts = Mat3d::Zero();//指示矩阵
            N_pts(0,0)= I_v.x() / std::abs(I_v.x());
            N_pts(1,1)= I_v.y() / std::abs(I_v.y());
            N_pts(2,2)= I_v.z() / std::abs(I_v.z());


            Eigen::Map<Vec3d> jacobian_feature(jacobians[2]);
            jacobian_feature = N_pts * R_ojw * R_wbj * R_bc * pts_j_td / (inv_dep_j*inv_dep_j);*/

            Eigen::Map<Vec3d> jacobian_feature(jacobians[2]);
            jacobian_feature=Vec3d::Zero();
        }
    }


    return true;
}




/**
 * 包围框的顶点的欧氏距离误差,
 * 优化变量: 物体位姿7,物体的dims 3
 */
bool BoxVertexFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d P_woi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    //Vec3d box(parameters[1][0],parameters[1][1],parameters[1][2]);

    Mat3d R_woi(Q_woi);

    ///计算估计值
    Vec3d pts_oi = (dims/2).cwiseProduct(indicate_symbol);
    Vec3d pts_w = R_woi * pts_oi;
    ///计算观测值
    Vec3d pts_ci_obs = corners.col( Box3D::CoordinateDirection(indicate_symbol.x(),indicate_symbol.y(),indicate_symbol.z()));

    Vec3d pts_w_obs = R_wbi * R_bc * pts_ci_obs;

    Vec3d err_vec = pts_w - pts_w_obs;
    residuals[0]= err_vec.x();
    residuals[1]= err_vec.y();
    residuals[2]= err_vec.z();


    counter++;
    if(counter<=8){
        Debugv("BoxVertexFactor:\n indicate_symbol:{} pts_oi:{} pts_w:{} \n pts_ci_obs:{} pts_w_obs:{} err_vec:{}",
               VecToStr(indicate_symbol), VecToStr(pts_oi), VecToStr(pts_w), VecToStr(pts_ci_obs), VecToStr(pts_w_obs),
               VecToStr(err_vec));
    }


    ///雅可比计算
    if (jacobians){

        ///对物体位姿求导
        if (jacobians[0]){
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[0]);
            jacobian_pose_oj.leftCols<3>() = Mat3d::Identity(); //对position的雅可比
            jacobian_pose_oj.middleCols(3,3) = -hat(R_woi * pts_oi);//对orientation的雅可比
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        ///对包围框求导
        if (jacobians[1]){
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_box(jacobians[1]);
            //jacobian_box = - Mat3d::Identity();
            jacobian_box = Mat3d::Zero();
        }
    }


    return true;
}




bool BoxDimsFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d box(parameters[0][0],parameters[0][1],parameters[0][2]);

    double err = (box- dims).squaredNorm();//返回二范数
    residuals[0] = err * err /100.; //不开方

    if(jacobians){
        if (jacobians[0]){
            Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> jacobian_box(jacobians[0]);
            jacobian_box = 2 * (box - dims).transpose();
        }
    }

    return true;
}



/**
 * 定义的orientation误差,
 * 优化变量 相机位姿7, 物体位姿7.
 * note:虽然公式里只关于旋转的误差,但由于参数块与position放在一起,故这里定义变量的global维度为7
 */
bool BoxOrientationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d P_wbi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_wbi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_woi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_woi(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Mat3d R_wbi(Q_wbi);
    Mat3d R_woi(Q_woi);
    Mat3d R_oiw = R_woi.transpose();

    Mat3d R = R_oiw * R_wbi * R_bc * R_cioi;//中间矩阵
    Sophus::Vector3d err = SO3d(R).log()  ;

    residuals[0] = err.x();
    residuals[1] = err.y();
    residuals[2] = err.z();


    if(jacobians){
        ///相机位姿, TODO
        if(jacobians[0]){
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_wbi(jacobians[0]);
            jacobian_pose_wbi.leftCols<6>() =   Mat36d::Zero();///未定义
            jacobian_pose_wbi.rightCols<1>().setZero();
        }
        ///物体位姿
        if(jacobians[1]){
            Sophus::Vector3d phi = SO3d(R).log();
            ///构造右乘矩阵
            double theta = - phi.norm();//右乘矩阵取负号
            Vec3d a = phi.normalized();
            Mat3d J_r = sin(theta)/theta * Mat3d::Identity() + (1-sin(theta)/theta) * a * a.transpose() + (1-cos(theta)/theta)*hat(a);
            ///计算雅可比
            Mat3d jaco_R = - J_r.inverse() * R.transpose();

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_woi(jacobians[1]);
            jacobian_pose_woi.leftCols<3>() = Mat3d::Zero();///平移项的雅可比不计算
            jacobian_pose_woi.middleCols(3,3) = jaco_R;
            jacobian_pose_woi.rightCols<1>().setZero();


            /*counter++;
            if(counter<10){
                Debugv("BoxOrientationFactor:\n err:{} theta:{} a:{} \n jaco_R:{} ",
                       VecToStr(err), theta, VecToStr(a), EigenToStr(jaco_R));
            }*/
        }

    }



    return true;
}




bool BoxPoseFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d P_wbi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_wbi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_woi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_woi(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Mat3d R_wbi(Q_wbi);
    Mat3d R_woi(Q_woi);
    Mat3d R_oiw = R_woi.transpose();

    Mat3d R = R_oiw * R_wbi * R_bc * R_cioi;//中间矩阵

    Vec3d err_t = P_woi - (R_wbi*(R_bc * P_cioi + P_bc)+P_wbi);
    Sophus::Vector3d err_w = SO3d(R).log();

    residuals[0] = err_t.x();
    residuals[1] = err_t.y();
    residuals[2] = err_t.z();
    residuals[3] = err_w.x();
    residuals[4] = err_w.y();
    residuals[5] = err_w.z();

    if(jacobians){
        ///相机位姿, TODO
        if(jacobians[0]){
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_wbi(jacobians[0]);
            jacobian_pose_wbi.leftCols<6>() =   Eigen::Matrix<double,6,6>::Zero();///未定义
            jacobian_pose_wbi.rightCols<1>().setZero();
        }
        ///物体位姿
        if(jacobians[1]){
            Sophus::Vector3d phi = SO3d(R).log();
            ///构造右乘矩阵
            double theta = - phi.norm();//右乘矩阵取负号
            Vec3d a = phi.normalized();
            Mat3d J_r = sin(theta)/theta * Mat3d::Identity() + (1-sin(theta)/theta) * a * a.transpose() + (1-cos(theta)/theta)*hat(a);
            ///计算雅可比
            Mat3d jaco_R = - J_r.inverse() * R.transpose();

            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_woi(jacobians[1]);
            jacobian_pose_woi = Eigen::Matrix<double,6,7>::Zero();
            jacobian_pose_woi.block<3,3>(0,0) = Mat3d::Identity();///左上角
            //jacobian_pose_woi.block<3,3>(0,0) = Mat3d::Zero();
            jacobian_pose_woi.block<3,3>(3,3) = jaco_R;//右下角

            /*counter++;
            if(counter%10==0){
                Debugv("BoxPoseFactor:\n err_t:{} err_w:{}\n theta:{} a:{} \n J_r:{} \n jaco_R:{} ",
                       VecToStr(err_t), VecToStr(err_w), theta, VecToStr(a), EigenToStr(J_r),
                       EigenToStr(jaco_R));
            }*/
        }
    }



    return true;
}




bool BoxPositionNormFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d P_woi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Mat3d R_woi(Q_woi);
    Mat3d R_oiw = R_woi.transpose();

    Vec3d P_woi_observe = (R_wbi*(R_bc * P_cioi + P_bc)+P_wbi);
    Vec3d err_t = P_woi - P_woi_observe;

    residuals[0] = err_t.squaredNorm();

    if(jacobians){
        ///物体位姿
        if(jacobians[0]){

            ///TODO,some bugs here

            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_pose_woi(jacobians[0]);
            jacobian_pose_woi = Eigen::Matrix<double,1,7>::Zero();
            jacobian_pose_woi.leftCols(3) =  2* (P_woi - P_woi_observe).transpose();

            counter++;
            if(counter%10==0){
                Debugv("BoxPoseNormFactor:\n P_woi:{} P_woi_observe:{} err_t:{}  \n jaco_R:{} ",
                       VecToStr(P_woi), VecToStr(P_woi_observe),err_t.squaredNorm(),
                       EigenToStr(jacobian_pose_woi));
            }
        }

    }

    return true;
}




bool BoxPositionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d P_woi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Mat3d R_woi(Q_woi);
    Mat3d R_oiw = R_woi.transpose();

    Vec3d P_woi_observe = (R_wbi*(R_bc * P_cioi + P_bc)+P_wbi);
    Vec3d err_t = P_woi - P_woi_observe;

    residuals[0] = err_t.x();
    residuals[1] = err_t.y();
    residuals[2] = err_t.z();

    if(jacobians){
        ///物体位姿
        if(jacobians[0]){

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_woi(jacobians[0]);
            jacobian_pose_woi = Eigen::Matrix<double,3,7>::Zero();
            jacobian_pose_woi.leftCols(3) =  Eigen::Matrix3d::Identity();

        }

    }

    return true;
}





}






