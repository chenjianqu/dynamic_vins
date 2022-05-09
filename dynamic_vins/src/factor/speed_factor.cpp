/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "speed_factor.h"

#include "estimator/vio_util.h"


namespace dynamic_vins{\


//优化变量:Twbj 7,Tbc 7,Twoj 7,Twoi 7,speed 6 ,逆深度 1,

bool ProjectionInstanceSpeedFactor::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    int index=0;
    Vec3d P_wbj(parameters[index][0], parameters[index][1], parameters[index][2]);
    Quatd Q_wbj(parameters[index][6], parameters[index][3], parameters[index][4], parameters[index][5]);

    index++;
    Vec3d P_bc(parameters[index][0], parameters[index][1], parameters[index][2]);
    Quatd Q_bc(parameters[index][6], parameters[index][3], parameters[index][4], parameters[index][5]);

    index++;
    Vec3d P_woj(parameters[index][0], parameters[index][1], parameters[index][2]);
    Quatd Q_woj(parameters[index][6], parameters[index][3], parameters[index][4], parameters[index][5]);

    index++;
    Vec3d P_woi(parameters[index][0], parameters[index][1], parameters[index][2]);
    Quatd Q_woi(parameters[index][6], parameters[index][3], parameters[index][4], parameters[index][5]);

    index++;
    Vec3d speed_v(parameters[index][0], parameters[index][1], parameters[index][2]);
    Vec3d speed_a(parameters[index][3], parameters[index][4], parameters[index][5]);

    index++;
    double inv_dep_j = parameters[index][0];

    Vec3d P_oioj=time_ij * speed_v;
    Vec3d se3_oioj=time_ij * speed_a;
    Mat3d Roioj=Sophus::SO3d::exp(se3_oioj).matrix();

    Vec3d pts_cam_j=pts_j / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=Q_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=Q_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=Q_woj.inverse()*(pts_w_j-P_woj);//k点在j时刻的物体坐标
    Vec3d pts_obj_ij=Roioj * pts_obj_j + P_oioj;

    Vec3d pts_obj_i=Q_woi.inverse() * (pts_w_j - P_woi);

    Eigen::Map<Vec3d> residual(residuals);
    residual=pts_obj_i-pts_obj_ij;


    //计算雅可比矩阵

    if (jacobians)
    {
        Mat3d R_wbj = Q_wbj.toRotationMatrix();
        Mat3d R_bc = Q_bc.toRotationMatrix();
        Mat3d R_woj = Q_woj.toRotationMatrix();
        Mat3d R_ojw=R_woj.transpose();
        Mat3d R_woi = Q_woi.toRotationMatrix();
        Mat3d R_oiw=R_woi.transpose();


        /// 计算雅可比矩阵
        //计算残差相对于p_bj和q_bj的雅可比
        if (jacobians[0])
        {
            Mat36d jaco_j;
            //P_ci相对于p_wbj的导数
            jaco_j.leftCols<3>() = -Roioj*R_ojw;
            //P_ci相对于q_wbj的导数
            jaco_j.rightCols<3>() = Roioj * R_ojw * R_wbj * Sophus::SO3d::hat(pts_imu_j);

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[0]);
            jacobian_pose_j.leftCols<6>() = jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        //计算残差相对于p_bc和q_bc的雅可比
        if (jacobians[1])
        {
            Mat36d jaco_ex;
            //P_ci相对于p_bc的导数
            Mat3d temp=Roioj * R_ojw * R_wbj;
            jaco_ex.leftCols<3>() = -temp;
            //P_ci相对于q_bc的导数
            jaco_ex.rightCols<3>() = - temp * R_bc * Sophus::SO3d::hat(pts_cam_j);

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);
            jacobian_ex_pose.leftCols<6>() = jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        //计算残差相对于p_woj和q_woj的雅可比
        if (jacobians[2])
        {
            Mat36d jaco_oj;
            //P_ci相对于p_wbj的导数
            jaco_oj.leftCols<3>() = Roioj * R_ojw;
            //P_ci相对于q_wbj的导数
            jaco_oj.rightCols<3>() = Roioj * Sophus::SO3d::hat(R_ojw * (pts_w_j - P_woj)).matrix();

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[2]);
            jacobian_pose_oj.leftCols<6>() = jaco_oj;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        //计算残差相对于p_woi和q_woi的雅可比
        if (jacobians[3])
        {
            Mat36d jaco_oi;
            //P_ci相对于p_woi的导数
            jaco_oi.leftCols<3>() = -R_oiw;
            //P_ci相对于q_woi的导数
            jaco_oi.rightCols<3>() = Sophus::SO3d::hat(R_oiw*(pts_w_j-P_woi));

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_oi(jacobians[3]);
            jacobian_pose_oi.leftCols<6>() = jaco_oi;
            jacobian_pose_oi.rightCols<1>().setZero();
        }
        if (jacobians[4])
        {
            Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobian_velocity(jacobians[4]);
            jacobian_velocity.leftCols<3>() = -time_ij * Mat3d::Identity();
            jacobian_velocity.rightCols<3>()=time_ij * Sophus::SO3d::hat(pts_obj_j);
        }

        //计算残差相对于逆深度的雅可比
        if (jacobians[5])
        {
            Eigen::Map<Vec3d> jacobian_feature(jacobians[5]);
            jacobian_feature =  Roioj * R_ojw * R_wbj * R_bc * pts_j / (inv_dep_j*inv_dep_j);
        }
    }
    sum_t += tic_toc.Toc();

    return true;


}



bool ProjectionSpeedFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d P_wbj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_wbj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_wbi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_wbi(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vec3d speed_v(parameters[2][0], parameters[2][1], parameters[2][2]);
    Vec3d speed_a(parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_j = parameters[3][0];

    Vec3d P_oioj=time_ij * speed_v;
    Mat3d R_oioj=Sophus::SO3d::exp(time_ij * speed_a).matrix();

    Vec3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (cur_td - td_i) * velocity_i;
    pts_j_td = pts_j - (cur_td - td_j) * velocity_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc1 * pts_cam_j + P_bc1;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=Q_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_wij=R_oioj*pts_w_j + P_oioj;
    Vec3d pts_imu_i=Q_wbi.inverse()*(pts_wij-P_wbi);//k点在i时刻的IMU坐标
    Vec3d pts_cam_i=R_bc2.inverse()*(pts_imu_i - P_bc2);//k点在i时刻的相机坐标

    Eigen::Map<Vec2d> residual(residuals);

    double dep_i = pts_cam_i.z();
    double inv_dep_i=1. / dep_i;
    residual = (pts_cam_i / dep_i).head<2>() - pts_i_td.head<2>();

    residual =   residual * 0.1;

    if (jacobians)
    {
        Mat3d R_wbj = Q_wbj.toRotationMatrix();
        Mat3d R_wbi = Q_wbi.toRotationMatrix();
        Mat3d R_biw=R_wbi.transpose();

        Mat3d R_cb2=R_bc2.transpose();

        Eigen::Matrix<double, 2, 3> reduce(2, 3);
        //计算残差相对于相机坐标的雅可比
        reduce <<
        inv_dep_i,     0,          -pts_cam_i(0) / (dep_i * dep_i),
        0,              inv_dep_i, -pts_cam_i(1) / (dep_i * dep_i);
        /// 计算雅可比矩阵
        //计算残差相对于p_bj和q_bj的雅可比
        if (jacobians[0])
        {
            Mat36d jaco_j;
            Mat3d temp=R_cb2 * R_biw * R_oioj;
            jaco_j.leftCols<3>() = temp;
            jaco_j.rightCols<3>() = -(temp * R_wbj * hat(pts_imu_j));;

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[0]);
            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        //计算残差相对于p_bi和q_bi的雅可比
        if (jacobians[1])
        {
            Mat36d jaco_i;
            jaco_i.leftCols<3>() = -R_cb2 * R_biw;
            jaco_i.rightCols<3>() = R_cb2 * hat(R_biw * (pts_wij - P_wbi));

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[1]);
            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }
        //计算残差相对于速度的雅可比
        if (jacobians[2])
        {
            Mat36d jaco_velocity;
            jaco_velocity.leftCols<3>() = R_cb2 * R_biw * time_ij;
            //jaco_velocity.rightCols<3>() = R_cb2 * R_biw  * R_oioj * hat(pts_w_j) * Hat(time_ij*Vec3d::Identity());
            jaco_velocity.rightCols<3>() = R_cb2 * R_biw * (-hat(pts_w_j)) * time_ij;

            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_v(jacobians[2]);
            jacobian_v = reduce * jaco_velocity;


        }

        //计算残差相对于逆深度的雅可比
        if (jacobians[3])
        {
            Eigen::Map<Vec2d> jacobian_feature(jacobians[3]);
            jacobian_feature = - reduce * R_cb2 * R_biw * R_oioj * R_wbj * R_bc1 * pts_j_td / (inv_dep_j*inv_dep_j);
        }
    }
    sum_t += tic_toc.Toc();

    return true;
}






bool ProjectionSpeedSimpleFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d speed_v(parameters[0][0], parameters[0][1], parameters[0][2]);
    Vec3d speed_a(parameters[0][3], parameters[0][4], parameters[0][5]);

    double inv_dep_j = parameters[1][0];

    Vec3d P_oioj=time_ij * speed_v;
    Mat3d R_oioj=Sophus::SO3d::exp(time_ij * speed_a).matrix();

    Vec3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (cur_td - td_i) * velocity_i;
    pts_j_td = pts_j - (cur_td - td_j) * velocity_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc1 * pts_cam_j + P_bc1;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_wij=R_oioj*pts_w_j + P_oioj;
    Vec3d pts_imu_i=R_wbi.inverse()*(pts_wij-P_wbi);//k点在i时刻的IMU坐标
    Vec3d pts_cam_i=R_bc2.inverse()*(pts_imu_i - P_bc2);//k点在i时刻的相机坐标

    Eigen::Map<Vec2d> residual(residuals);

    double dep_i = pts_cam_i.z();
    double inv_dep_i=1. / dep_i;
    residual = (pts_cam_i / dep_i).head<2>() - pts_i_td.head<2>();

    //residual = sqrt_info * residual;
    residual =  residual * factor;

    if (jacobians)
    {
        Mat3d R_cb2=R_bc2.transpose();
        Mat3d R_biw=R_wbi.transpose();

        Eigen::Matrix<double, 2, 3> reduce(2, 3);
        //计算残差相对于相机坐标的雅可比
        reduce <<
        inv_dep_i,     0,          -pts_cam_i(0) / (dep_i * dep_i),
        0,              inv_dep_i, -pts_cam_i(1) / (dep_i * dep_i);
        //reduce = sqrt_info * reduce ;
        reduce = factor * reduce ;
        //计算残差相对于速度的雅可比
        if (jacobians[0])
        {
            Mat36d jaco_velocity;
            jaco_velocity.leftCols<3>() = R_cb2 * R_biw * time_ij;
            //jaco_velocity.rightCols<3>() = R_cb2 * R_biw  * R_oioj * hat(pts_w_j) * Hat(time_ij*Vec3d::Identity());
            jaco_velocity.rightCols<3>() = R_cb2 * R_biw * (-hat(pts_w_j)) * time_ij;

            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> jacobian_v(jacobians[0]);
            jacobian_v = reduce * jaco_velocity;

            /*debug_flag++;
            if(debug_flag%50==0){
                printf("******\n");
                printf("speed:v(%.2lf,%.2lf,%.2lf) a(%.2lf,%.2lf,%.2lf)\n",speed_v.x(),speed_v.y(),speed_v.z(),speed_a.x(),speed_a.y(),speed_a.z());
                printf("inv_dep_j:%.2lf,time_ij:%.2lf\n",inv_dep_j,time_ij);
                printf("P_oioj:(%.2lf,%.2lf,%.2lf)\n",P_oioj.x(),P_oioj.y(),P_oioj.z());
                printf("pts_cam_j:(%.2lf,%.2lf,%.2lf)\n",pts_cam_j.x(),pts_cam_j.y(),pts_cam_j.z());
                printf("pts_w_j:(%.2lf,%.2lf,%.2lf)\n",pts_w_j.x(),pts_w_j.y(),pts_w_j.z());
                printf("pts_wij:(%.2lf,%.2lf,%.2lf)\n",pts_wij.x(),pts_wij.y(),pts_wij.z());
                printf("pts_cam_i:(%.2lf,%.2lf,%.2lf)\n",pts_cam_i.x(),pts_cam_i.y(),pts_cam_i.z());
                printf("residual:(%.4lf,%.4lf)\n",residual.x(),residual.y());
                printf("jaco_velocity\n");
                cout<<jaco_velocity.matrix()<<endl;
                printf("reduce\n");
                cout<<reduce.matrix()<<endl;
                printf("jacobian_v\n");
                cout<<jacobian_v<<endl;
            }*/

        }

        //计算残差相对于逆深度的雅可比
        if (jacobians[1])
        {
            Eigen::Map<Vec2d> jacobian_feature(jacobians[1]);
            jacobian_feature = - reduce * R_cb2 * R_biw * R_oioj * R_wbj * R_bc1 * pts_j_td / (inv_dep_j*inv_dep_j);
        }
    }

    return true;
}







bool ProjectionSpeedPoseFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d P_wbj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_wbj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_bc(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_bc(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vec3d P_woj(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quatd Q_woj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    Vec3d P_woi(parameters[3][0], parameters[3][1], parameters[3][2]);
    Quatd Q_woi(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

    Vec3d speed_v(parameters[4][0], parameters[4][1], parameters[4][2]);
    Vec3d speed_a(parameters[4][3], parameters[4][4], parameters[4][5]);

    double inv_dep_j = parameters[5][0];

    Vec3d P_oioj=time_ij * speed_v;
    Mat3d R_oioj=Sophus::SO3d::exp(time_ij * speed_a).matrix();

    Vec3d pts_cam_j=pts_j / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=Q_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=Q_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=Q_woj.inverse() * (pts_w_j - P_woj);

    Vec3d pts1=Q_woi * pts_obj_j + P_woi;

    Vec3d pts2=R_oioj * pts_w_j + P_oioj;

    Eigen::Map<Vec3d> residual(residuals);
    residual=pts1-pts2;


    if (jacobians)
    {
        Mat3d R_wbj = Q_wbj.toRotationMatrix();
        Mat3d R_bc = Q_bc.toRotationMatrix();
        Mat3d R_woj = Q_woj.toRotationMatrix();
        Mat3d R_ojw=R_woj.transpose();
        Mat3d R_woi = Q_woi.toRotationMatrix();

        Mat3d temp=R_woi*R_ojw - R_oioj;


        /// 计算雅可比矩阵
        //计算残差相对于p_bj和q_bj的雅可比
        if (jacobians[0])
        {
            Mat36d jaco_j;
            jaco_j.leftCols<3>() = temp;
            jaco_j.rightCols<3>() = - temp * R_wbj * hat(pts_imu_j);;

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[0]);
            jacobian_pose_j.leftCols<6>() = jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        // 计算残差相对于p_bc和q_bc的雅可比
        if (jacobians[1])
        {
            Mat36d jaco_i;
            jaco_i.leftCols<3>() = temp * R_wbj;
            jaco_i.rightCols<3>() = -temp * R_wbj * R_bc * hat(pts_cam_j);

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[1]);
            jacobian_pose_i.leftCols<6>() = jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Mat36d jaco_ex;
            jaco_ex.leftCols<3>() = - R_woi*R_ojw;
            jaco_ex.rightCols<3>() = -R_woi + hat(R_ojw * (pts_w_j - P_woj)) ;

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            jacobian_ex_pose.leftCols<6>() = jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
            Mat36d jaco_ex;
            jaco_ex.leftCols<3>() = Mat3d::Identity();
            jaco_ex.rightCols<3>() = -R_woi * hat(R_ojw * (pts_w_j - P_woj)) ;

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[3]);
            jacobian_ex_pose.leftCols<6>() = jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        //计算残差相对于速度的雅可比
        if (jacobians[4])
        {
            Mat36d jaco_velocity;
            jaco_velocity.leftCols<3>() = - time_ij * Mat3d::Identity();
            jaco_velocity.rightCols<3>() = time_ij * hat(pts_w_j);

            Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobian_v(jacobians[4]);
            jacobian_v =  jaco_velocity;
        }

        //计算残差相对于逆深度的雅可比
        if (jacobians[5])
        {
            Eigen::Map<Vec3d> jacobian_feature(jacobians[5]);
            jacobian_feature = - temp*R_wbj*R_bc*pts_cam_j/(inv_dep_j*inv_dep_j);
        }
    }

    return true;


}




bool SpeedPoseSimpleFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d P_woj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_woi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_woi(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vec3d speed_v(parameters[2][0], parameters[2][1], parameters[2][2]);
    Vec3d speed_a(parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_j = parameters[3][0];

    Vec3d P_oioj=time_ij * speed_v;
    Mat3d R_oioj=Sophus::SO3d::exp(time_ij * speed_a).matrix();

    Vec3d pts_j_td = pts_j - (td - td_j) * vel_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=Q_woj.inverse() * (pts_w_j - P_woj);

    Vec3d pts1=Q_woi * pts_obj_j + P_woi;

    Vec3d pts2=R_oioj * pts_w_j + P_oioj;

    Eigen::Map<Vec3d> residual(residuals);
    residual=pts1-pts2;


    if (jacobians)
    {
        Mat3d R_woj = Q_woj.toRotationMatrix();
        Mat3d R_ojw=R_woj.transpose();
        Mat3d R_woi = Q_woi.toRotationMatrix();

        /// 计算雅可比矩阵
        //计算残差相对于p_bj和q_bj的雅可比
        if (jacobians[0])
        {
            Mat36d jaco_ex;
            jaco_ex.leftCols<3>() = - R_woi*R_ojw;
            jaco_ex.rightCols<3>() = -R_woi + hat(R_ojw * (pts_w_j - P_woj)) ;

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[0]);
            jacobian_ex_pose.leftCols<6>() = jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[1])
        {
            Mat36d jaco_ex;
            jaco_ex.leftCols<3>() = Mat3d::Identity();
            jaco_ex.rightCols<3>() = -R_woi * hat(R_ojw * (pts_w_j - P_woj)) ;

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);
            jacobian_ex_pose.leftCols<6>() = jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        //计算残差相对于速度的雅可比
        if (jacobians[2])
        {
            Mat36d jaco_velocity;
            jaco_velocity.leftCols<3>() = - time_ij * Mat3d::Identity();
            jaco_velocity.rightCols<3>() = time_ij * hat(pts_w_j);

            Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobian_v(jacobians[2]);
            jacobian_v =  jaco_velocity;
        }
        //计算残差相对于逆深度的雅可比
        if (jacobians[3])
        {
            Eigen::Map<Vec3d> jacobian_feature(jacobians[3]);
            //jacobian_feature = - temp*R_wbj*R_bc*pts_cam_j/(inv_dep_j*inv_dep_j);
            jacobian_feature = Vec3d::Zero();
        }
    }

    return true;


}




bool ConstSpeedFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d speed_v(parameters[0][0], parameters[0][1], parameters[0][2]);
    Vec3d speed_a(parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_oioj=time_ij * speed_v;
    Mat3d R_oioj=Sophus::SO3d::exp(time_ij * speed_a).matrix();

    Vec3d last_P_oioj=time_ij * last_v;
    Mat3d last_R_oioj=Sophus::SO3d::exp(time_ij * last_a).matrix();

    Vec3d pts_cam_j = pts_j / inv_depth;
    Vec3d pts_imu_j=R_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标

    Vec3d pts1=R_oioj * pts_w_j + P_oioj;
    Vec3d pts2=last_R_oioj * pts_w_j + last_P_oioj;

    Eigen::Map<Vec3d> residual(residuals);
    residual=pts1-pts2;

    residual*=100;

    if (jacobians)
    {
        if (jacobians[0])
        {
            Mat36d jaco_velocity;
            jaco_velocity.leftCols<3>() = - time_ij * Mat3d::Identity();
            jaco_velocity.rightCols<3>() = time_ij * hat(pts_w_j);
            Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobian_v(jacobians[0]);
            jacobian_v =  jaco_velocity;
        }
    }

    return true;
}




bool ConstSpeedSimpleFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d speed_v(parameters[0][0], parameters[0][1], parameters[0][2]);
    Vec3d speed_a(parameters[0][3], parameters[0][4], parameters[0][5]);


    residuals[0]= ((speed_a - last_a).norm() + (speed_v - last_v).norm()) * factor;

    if (jacobians)
    {
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> jacobian_v(jacobians[0]);

            jacobian_v.leftCols<3>() = 2 * speed_v.transpose() * factor;
            jacobian_v.rightCols<3>() = 2 * speed_a.transpose() * factor;
        }
    }

    return true;
}



/**
 * 速度和物体位姿的误差
 * 误差维度6, 优化变量:物体位姿woi(未实现), 物体位姿woj(未实现),物体速度6
 */
bool SpeedPoseFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d P_woi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_woj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_woj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vec3d speed_v(parameters[2][0], parameters[2][1], parameters[2][2]);
    Vec3d speed_w(parameters[2][3], parameters[2][4], parameters[2][5]);

    Mat3d R_woj(Q_woj);
    Mat3d R_ojw = R_woj.transpose();
    Mat3d R_woi(Q_woi);

    ///误差计算
    Vec3d err_v = time_ij * speed_v - (P_woj - P_woi);

    Sophus::SO3d R_delta = Sophus::SO3d::exp(speed_w * time_ij);

    Sophus::SO3d R_err(R_delta.matrix() * R_ojw * R_woi);
    Vec3d err_w = R_err.log();

    residuals[0]=err_v.x();
    residuals[1]=err_v.y();
    residuals[2]=err_v.z();
    residuals[3]=err_w.x();
    residuals[4]=err_w.y();
    residuals[5]=err_w.z();

    if(jacobians){
        if (jacobians[0]){
            ///TODO
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
            jacobian_pose.leftCols<6>() = Eigen::Matrix<double,6,6>::Zero();
            jacobian_pose.rightCols<1>().setZero();
        }
        if (jacobians[1]){
            ///TODO
            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose(jacobians[1]);
            jacobian_pose.leftCols<6>() = Eigen::Matrix<double,6,6>::Zero();
            jacobian_pose.rightCols<1>().setZero();
        }
        ///对速度的雅可比
        if (jacobians[2]){
            Mat3d R=R_ojw * R_woi;

            Sophus::Vector3d phi = Sophus::SO3d(R).log();
            double theta = - phi.norm();//右乘矩阵取负号
            Vec3d a = phi.normalized();
            //构造右乘矩阵
            Mat3d J_r = sin(theta)/theta * Mat3d::Identity() + (1-sin(theta)/theta) * a * a.transpose() + (1-cos(theta)/theta)*hat(a);

            Eigen::Matrix<double,6,3> jacobian_v = Eigen::Matrix<double,6,3>::Zero();
            jacobian_v.topRows(3) = time_ij * Mat3d::Identity();
            Eigen::Matrix<double,6,3> jacobian_w = Eigen::Matrix<double,6,3>::Zero();
            jacobian_w.bottomRows(3) = - time_ij * J_r.inverse() * R.transpose();

            Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_speed(jacobians[2]);
            jacobian_speed.leftCols<3>() = jacobian_v;
            jacobian_speed.middleCols(3,3) = jacobian_w;
            jacobian_speed.rightCols<1>().setZero();
        }

    }


    return true;
}








}