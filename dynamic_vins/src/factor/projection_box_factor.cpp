/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "projection_box_factor.h"

#include "utils/utility.h"

namespace dynamic_vins{\

double ProjBoxFactor::sum_t;
double ProjBoxSimpleFactor::sum_t;
double BoxAbsFactor::sum_t;
double BoxSqrtFactor::sum_t;
double BoxPowFactor::sum_t;


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

    /*    residuals[0]=std::max(pts_abs_x-box_x,0.);
        residuals[1]=std::max(pts_abs_y-box_y,0.);
        residuals[2]=std::max(pts_abs_z-box_z,0.);*/

    if (jacobians)
    {
        Mat3d R_wbj = Q_wbj.toRotationMatrix();
        Mat3d R_bc = Q_bc.toRotationMatrix();
        Mat3d R_woj = Q_woj.toRotationMatrix();
        Mat3d R_ojw=R_woj.transpose();

        /*        double reduce11=pts_abs_x-box_x>0? pts_obj_j.x()/pts_abs_x: 0.;
                double reduce22=pts_abs_y-box_y>0? pts_obj_j.y()/pts_abs_y: 0.;
                double reduce33=pts_abs_z-box_z>0? pts_obj_j.z()/pts_abs_z: 0.;*/
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
            jaco_j.rightCols<3>() = -(R_ojw *R_wbj *Utility::skewSymmetric(pts_imu_j));

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
            jaco_ex.rightCols<3>() = -(R_ojw*R_wbj*R_bc *Utility::skewSymmetric(pts_imu_j));

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Mat36d jaco_oj;
            jaco_oj.leftCols<3>() = -R_ojw;
            jaco_oj.rightCols<3>() = Utility::skewSymmetric(R_ojw * (pts_w_j - P_woj));

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[2]);
            jacobian_pose_oj.leftCols<6>() = reduce * jaco_oj;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
            /*            double box11=pts_abs_x-box_x>0 ? -1 : 0;
                        double box22=pts_abs_y-box_y>0 ? -1 : 0;
                        double box33=pts_abs_z-box_z>0 ? -1 : 0;*/
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

    residuals[0]=std::sqrt(std::abs(pts_abs_x-box_x)* ALPHA ) ;//*(pts_abs_x-box_x) * ALPHA;
    residuals[1]=std::sqrt(std::abs(pts_abs_y-box_y)* ALPHA );//*(pts_abs_y-box_y) * ALPHA;
    residuals[2]=std::sqrt(std::abs(pts_abs_z-box_z)* ALPHA );//*(pts_abs_z-box_z) * ALPHA;



    /*    residuals[0]=std::max(pts_abs_x-box_x,0.);
        residuals[1]=std::max(pts_abs_y-box_y,0.);
        residuals[2]=std::max(pts_abs_z-box_z,0.);*/

    if (jacobians)
    {
        Mat3d R_wbj = Q_wbj.toRotationMatrix();
        Mat3d R_bc = Q_bc.toRotationMatrix();
        Mat3d R_ojw=R_woj.transpose();

        /*        double reduce11=pts_abs_x-box_x>0? pts_obj_j.x()/pts_abs_x: 0.;
                double reduce22=pts_abs_y-box_y>0? pts_obj_j.y()/pts_abs_y: 0.;
                double reduce33=pts_abs_z-box_z>0? pts_obj_j.z()/pts_abs_z: 0.;*/
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
            jaco_j.rightCols<3>() = -(R_ojw *R_wbj *Utility::skewSymmetric(pts_imu_j));

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
            jaco_ex.rightCols<3>() = -(R_ojw*R_wbj*R_bc *Utility::skewSymmetric(pts_imu_j));

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            /*            double box11=pts_abs_x-box_x>0 ? -1 : 0;
                        double box22=pts_abs_y-box_y>0 ? -1 : 0;
                        double box33=pts_abs_z-box_z>0 ? -1 : 0;*/
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

}






}












