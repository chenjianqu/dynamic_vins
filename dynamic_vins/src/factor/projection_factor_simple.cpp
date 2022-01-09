/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "projection_factor_simple.h"

namespace dynamic_vins{\


Eigen::Matrix2d ProjectionTwoFrameOneCamFactorSimple::sqrt_info;
double ProjectionTwoFrameOneCamFactorSimple::sum_t;

Eigen::Matrix2d ProjectionOneFrameTwoCamFactorSimple::sqrt_info;
double ProjectionOneFrameTwoCamFactorSimple::sum_t;

Eigen::Matrix2d ProjectionTwoFrameTwoCamFactorSimple::sqrt_info;
double ProjectionTwoFrameTwoCamFactorSimple::sum_t;

double ProjectionBoxFactorSimple::sum_t;

ProjectionTwoFrameOneCamFactorSimple::ProjectionTwoFrameOneCamFactorSimple(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) :
pts_i(_pts_i), pts_j(_pts_j){
}

bool ProjectionTwoFrameOneCamFactorSimple::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_i = parameters[3][0];


    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Map<Eigen::Vector2d> residual(residuals);


    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();

    residual = sqrt_info * residual;

    if (jacobians)
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3);

        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
        0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);

        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
            Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                    Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i * -1.0 / (inv_dep_i * inv_dep_i);
        }

    }
    sum_t += tic_toc.Toc();

    return true;
}




ProjectionTwoFrameTwoCamFactorSimple::ProjectionTwoFrameTwoCamFactorSimple(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) :
pts_i(_pts_i), pts_j(_pts_j){
}

bool ProjectionTwoFrameTwoCamFactorSimple::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    Eigen::Vector3d tic2(parameters[3][0], parameters[3][1], parameters[3][2]);
    Eigen::Quaterniond qic2(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

    double inv_dep_i = parameters[4][0];

    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2);
    Eigen::Map<Eigen::Vector2d> residual(residuals);


    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();


    residual = sqrt_info * residual;

    if (jacobians)
    {
        Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        Eigen::Matrix3d Rj = Qj.toRotationMatrix();
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix3d ric2 = qic2.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm();
        Eigen::Matrix3d norm_jaco;
        double x1, x2, x3;
        x1 = pts_camera_j(0);
        x2 = pts_camera_j(1);
        x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), - x1 * x2 / pow(norm, 3),            - x1 * x3 / pow(norm, 3),
        - x1 * x2 / pow(norm, 3),            1.0 / norm - x2 * x2 / pow(norm, 3), - x2 * x3 / pow(norm, 3),
        - x1 * x3 / pow(norm, 3),            - x2 * x3 / pow(norm, 3),            1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco;
#else
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
        0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>() = ric2.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric2.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>() = ric2.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric2.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric2.transpose() * Rj.transpose() * Ri;
            jaco_ex.rightCols<3>() = ric2.transpose() * Rj.transpose() * Ri * ric * -Utility::skewSymmetric(pts_camera_i);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose1(jacobians[3]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = - ric2.transpose();
            jaco_ex.rightCols<3>() = Utility::skewSymmetric(pts_camera_j);
            jacobian_ex_pose1.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose1.rightCols<1>().setZero();
        }
        if (jacobians[4])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[4]);
            jacobian_feature = reduce * ric2.transpose() * Rj.transpose() * Ri * ric * pts_i * -1.0 / (inv_dep_i * inv_dep_i);

        }

    }
    sum_t += tic_toc.Toc();

    return true;
}



ProjectionOneFrameTwoCamFactorSimple::ProjectionOneFrameTwoCamFactorSimple(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j) :
pts_i(_pts_i), pts_j(_pts_j)
{}

bool ProjectionOneFrameTwoCamFactorSimple::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;

    Eigen::Vector3d tic(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond qic(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d tic2(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond qic2(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    double inv_dep_i = parameters[2][0];

    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_imu_j = pts_imu_i;
    Eigen::Vector3d pts_camera_j = qic2.inverse() * (pts_imu_j - tic2);
    Eigen::Map<Eigen::Vector2d> residual(residuals);

#ifdef UNIT_SPHERE_ERROR
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j.head<2>();
#endif

    residual = sqrt_info * residual;

    if (jacobians)
    {
        Eigen::Matrix3d ric = qic.toRotationMatrix();
        Eigen::Matrix3d ric2 = qic2.toRotationMatrix();
        Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm();
        Eigen::Matrix3d norm_jaco;
        double x1, x2, x3;
        x1 = pts_camera_j(0);
        x2 = pts_camera_j(1);
        x3 = pts_camera_j(2);
        norm_jaco << 1.0 / norm - x1 * x1 / pow(norm, 3), - x1 * x2 / pow(norm, 3),            - x1 * x3 / pow(norm, 3),
        - x1 * x2 / pow(norm, 3),            1.0 / norm - x2 * x2 / pow(norm, 3), - x2 * x3 / pow(norm, 3),
        - x1 * x3 / pow(norm, 3),            - x2 * x3 / pow(norm, 3),            1.0 / norm - x3 * x3 / pow(norm, 3);
        reduce = tangent_base * norm_jaco;
#else
        reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),
        0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[0]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = ric2.transpose();
            jaco_ex.rightCols<3>() = ric2.transpose() * ric * -Utility::skewSymmetric(pts_camera_i);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose1(jacobians[1]);
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = - ric2.transpose();
            jaco_ex.rightCols<3>() = Utility::skewSymmetric(pts_camera_j);
            jacobian_ex_pose1.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose1.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[2]);
            jacobian_feature = reduce * ric2.transpose() * ric * pts_i * -1.0 / (inv_dep_i * inv_dep_i);

        }

    }
    sum_t += tic_toc.Toc();

    return true;
}


/**
 * 维度:<误差项，IMU位姿,外参,box,逆深度>
 */
bool ProjectionBoxFactorSimple::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;

    Eigen::Vector3d P_wbj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Q_wbj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d P_bc(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Q_bc(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    double box_x=parameters[2][0],box_y=parameters[2][1],box_z=parameters[2][2];
    double inv_dep_j = parameters[3][0];

    Eigen::Vector3d pts_j_td;
    pts_j_td = pts_j - (curr_td - td_j) * velocity_j;

    Eigen::Vector3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Eigen::Vector3d pts_imu_j=Q_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Eigen::Vector3d pts_w_j=Q_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标

    double pts_abs_x=std::abs(pts_w_j.x());
    double pts_abs_y=std::abs(pts_w_j.y());
    double pts_abs_z=std::abs(pts_w_j.z());

    residuals[0]=(pts_abs_x-box_x)*(pts_abs_x-box_x);
    residuals[1]=(pts_abs_y-box_y)*(pts_abs_y-box_y);
    residuals[2]=(pts_abs_z-box_z)*(pts_abs_z-box_z);

    /*    residuals[0]=std::max(pts_abs_x-box_x,0.);
        residuals[1]=std::max(pts_abs_y-box_y,0.);
        residuals[2]=std::max(pts_abs_z-box_z,0.);*/

    if (jacobians)
    {
        Eigen::Matrix3d R_wbj = Q_wbj.toRotationMatrix();
        //Eigen::Matrix3d R_bjw=R_wbj.transpose();
        Eigen::Matrix3d R_bc = Q_bc.toRotationMatrix();
        //Eigen::Matrix3d R_cb=R_bc.transpose();

        /*        double reduce11=pts_abs_x-box_x>0? pts_obj_j.x()/pts_abs_x: 0.;
                double reduce22=pts_abs_y-box_y>0? pts_obj_j.y()/pts_abs_y: 0.;
                double reduce33=pts_abs_z-box_z>0? pts_obj_j.z()/pts_abs_z: 0.;*/
        double reduce11=2* (pts_abs_x - box_x) * pts_w_j.x() / pts_abs_x;
        double reduce22=2* (pts_abs_y - box_y) * pts_w_j.y() / pts_abs_y;
        double reduce33=2* (pts_abs_z - box_z) * pts_w_j.z() / pts_abs_z;

        ///d(e)/d(pts_oj)
        Eigen::Matrix3d reduce=Eigen::Matrix3d::Zero();
        reduce(0,0)=reduce11;
        reduce(1,1)=reduce22;
        reduce(2,2)=reduce33;


        /// 计算雅可比矩阵
        //计算残差相对于p_bj和q_bj的雅可比
        if (jacobians[0])
        {
            Eigen::Matrix<double, 3, 6> jaco_j;
            //P_ci相对于p_wbj的导数
            jaco_j.leftCols<3>() = Eigen::Matrix3d::Identity();
            //P_ci相对于q_wbj的导数
            jaco_j.rightCols<3>() = -(R_wbj * Utility::skewSymmetric(pts_imu_j));

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[0]);
            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[1])
        {
            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = R_wbj;
            jaco_ex.rightCols<3>() = -(R_wbj*R_bc *Utility::skewSymmetric(pts_imu_j));

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

            Eigen::Matrix3d jaco_box=Eigen::Matrix<double,3,3>::Zero();
            jaco_box(0,0)=box11;
            jaco_box(1,1)=box22;
            jaco_box(2,2)=box33;

            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian_box(jacobians[2]);
            jacobian_box=jaco_box;
        }
        if(jacobians[3])
        {
            Eigen::Map<Eigen::Vector3d> jacobian_feature(jacobians[3]);
            jacobian_feature = -reduce  * R_wbj * R_bc * pts_j / (inv_dep_j*inv_dep_j);
        }
    }

    sum_t += tic_toc.Toc();

    return true;

}

}