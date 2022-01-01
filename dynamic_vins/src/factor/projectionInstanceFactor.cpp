//
// Created by chen on 2021/10/8.
//

#include "projectionInstanceFactor.h"

#include "../estimator/dynamic.h"



int ProjInst12Factor::debug_num=0;
int ProjInst21Factor::debug_num=0;


Mat2d ProjectionInstanceFactor::sqrt_info;
double ProjectionInstanceFactor::sum_t;

Mat2d ProjInst12Factor::sqrt_info;
double ProjInst12Factor::sum_t;

Mat2d ProjInst12FactorSimple::sqrt_info;
double ProjInst12FactorSimple::sum_t;

Mat2d ProjInst21Factor::sqrt_info;
double ProjInst21Factor::sum_t;

Mat2d ProjInst21SimpleFactor::sqrt_info;
double ProjInst21SimpleFactor::sum_t;

Mat2d ProjInst22Factor::sqrt_info;
double ProjInst22Factor::sum_t;

Mat2d ProjInst22SimpleFactor::sqrt_info;
double ProjInst22SimpleFactor::sum_t;

/**
 * 计算残差、Jacobian
 * @param parameters 优化变量，是参数数组.包括:
 * para_State[frame_i],para_State[frame_j],para_Ex_Pose[0],inst.para_State[frame_i],inst.para_State[frame_j], inst.para_Depth[depth_index]);
 * @param residuals 计算完成的残差
 * @param jacobians 计算完成的雅可比矩阵
 * @return
 */
bool ProjectionInstanceFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d P_wbj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_wbj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_wbi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_wbi(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vec3d P_bc(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quatd Q_bc(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    Vec3d P_woj(parameters[3][0], parameters[3][1], parameters[3][2]);
    Quatd Q_woj(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

    Vec3d P_woi(parameters[4][0], parameters[4][1], parameters[4][2]);
    Quatd Q_woi(parameters[4][6], parameters[4][3], parameters[4][4], parameters[4][5]);

    double inv_dep_j = parameters[5][0];


    Vec3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (cur_td - td_i) * velocity_i;
    pts_j_td = pts_j - (cur_td - td_j) * velocity_j;


    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=Q_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=Q_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=Q_woj.inverse()*(pts_w_j-P_woj);//k点在j时刻的物体坐标
    Vec3d pts_w_i=Q_woi*pts_obj_j+P_woi;//k点在i时刻的世界坐标
    Vec3d pts_imu_i=Q_wbi.inverse()*(pts_w_i-P_wbi);//k点在i时刻的IMU坐标
    Vec3d pts_cam_i=Q_bc.inverse()*(pts_imu_i - P_bc);//k点在i时刻的相机坐标

    double dep_i = pts_cam_i.z();
    double inv_dep_i=1.0 / dep_i;

    Eigen::Map<Vec2d> residual(residuals);

    ///残差=估计值-观测值，误差维度2
    residual = (pts_cam_i / dep_i).head<2>() - pts_i_td.head<2>();

    residual = sqrt_info * residual;


    ///计算雅可比矩阵

    if (jacobians)
    {
        Mat3d R_wbj = Q_wbj.toRotationMatrix();
        //Mat3d R_bjw=R_wbj.transpose();
        Mat3d R_wbi = Q_wbi.toRotationMatrix();
        Mat3d R_biw=R_wbi.transpose();
        Mat3d R_bc = Q_bc.toRotationMatrix();
        Mat3d R_cb=R_bc.transpose();
        Mat3d R_woj = Q_woj.toRotationMatrix();
        Mat3d R_ojw=R_woj.transpose();
        Mat3d R_woi = Q_woi.toRotationMatrix();
        //Mat3d R_oiw=R_woi.transpose();

        Mat23d reduce(2, 3);
        //计算残差相对于相机坐标的雅可比
        reduce <<   inv_dep_i,     0,          -pts_cam_i(0) / (dep_i * dep_i),
        0,              inv_dep_i, -pts_cam_i(1) / (dep_i * dep_i);
        reduce = sqrt_info * reduce;

        /// 计算雅可比矩阵
        //计算残差相对于p_bj和q_bj的雅可比
        if (jacobians[0])
        {
            Mat36d jaco_j;
            Mat3d temp=R_cb * R_biw * R_woi *R_ojw;
            //P_ci相对于p_wbj的导数
            jaco_j.leftCols<3>() = temp;
            //P_ci相对于q_wbj的导数
            jaco_j.rightCols<3>() = -(temp *R_wbj *Utility::skewSymmetric(pts_imu_j));

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[0]);
            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        //计算残差相对于p_bi和q_bi的雅可比
        if (jacobians[1])
        {
            Mat36d jaco_i;
            //P_ci相对于p_wbi的导数
            jaco_i.leftCols<3>() = -R_cb * R_biw;
            //P_ci相对于q_wbi的导数
            jaco_i.rightCols<3>() = R_cb * Utility::skewSymmetric(R_biw*(pts_w_i-P_wbi));

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[1]);
            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }
        //计算残差相对于p_bc和q_bc的雅可比
        if (jacobians[2])
        {
            Mat36d jaco_ex;
            //P_ci相对于p_bc的导数
            Mat3d temp=R_cb * R_biw * R_woi *R_ojw *R_wbj;
            jaco_ex.leftCols<3>() = temp-R_cb;
            //P_ci相对于q_bc的导数
            jaco_ex.rightCols<3>() = - temp * R_bc * Utility::skewSymmetric(pts_cam_j) + Utility::skewSymmetric(R_cb * (pts_imu_i - P_bc)) ;

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        //计算残差相对于p_woj和q_woj的雅可比
        if (jacobians[3])
        {
            Mat36d jaco_oj;
            Mat3d temp=R_cb * R_biw * R_woi;
            //P_ci相对于p_wbj的导数
            jaco_oj.leftCols<3>() = -temp * R_ojw;
            //P_ci相对于q_wbj的导数
            jaco_oj.rightCols<3>() = temp*Utility::skewSymmetric(R_ojw * (pts_w_j - P_woj));

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[3]);
            jacobian_pose_oj.leftCols<6>() = reduce * jaco_oj;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        //计算残差相对于p_woi和q_woi的雅可比
        if (jacobians[4])
        {
            Mat36d jaco_oi;
            //P_ci相对于p_woi的导数
            jaco_oi.leftCols<3>() = R_cb * R_biw;
            //P_ci相对于q_woi的导数
            jaco_oi.rightCols<3>() = -R_cb * R_biw * R_woi * Utility::skewSymmetric(pts_obj_j);

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_oi(jacobians[4]);
            jacobian_pose_oi.leftCols<6>() = reduce * jaco_oi;
            jacobian_pose_oi.rightCols<1>().setZero();
        }
        //计算残差相对于逆深度的雅可比
        if (jacobians[5])
        {
            Eigen::Map<Vec2d> jacobian_feature(jacobians[5]);
            jacobian_feature = reduce * R_cb * R_biw * R_woi * R_ojw * R_wbj * R_bc * pts_j / (inv_dep_j*inv_dep_j);

        }
    }
    sum_t += tic_toc.toc();

    return true;



/*
    Vec3d Pj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Qj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d Pi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Qi(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vec3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quatd qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_j = parameters[3][0];


    Vec3d  pts_cam_j=pts_j / inv_dep_j;
    Vec3d pts_imu_j = qic * pts_cam_j + tic;
    Vec3d pts_w = Qj * pts_imu_j + Pj;
    Vec3d pts_imu_i = Qi.inverse() * (pts_w - Pi);
    Vec3d pts_camera_i = qic.inverse() * (pts_imu_i - tic);
    Eigen::Map<Vec2d> residual(residuals);

    double dep_i = pts_camera_i.z();
    residual = (pts_camera_i / dep_i).head<2>() - pts_i.head<2>();

    residual = sqrt_info * residual;

    if (jacobians)
    {
        Mat3d Ri = Qi.toRotationMatrix();
        Mat3d Rj = Qj.toRotationMatrix();
        Mat3d ric = qic.toRotationMatrix();
        Mat23d reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
        double norm = pts_camera_j.norm();
        Mat3d norm_jaco;
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

            Mat36d jaco_i;
            jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
            jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * -Utility::skewSymmetric(pts_imu_i);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Mat36d jaco_j;
            jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
            jaco_j.rightCols<3>() = ric.transpose() * Utility::skewSymmetric(pts_imu_j);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            Mat36d jaco_ex;
            jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Mat3d::Identity());
            Mat3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
            jaco_ex.rightCols<3>() = -tmp_r * Utility::skewSymmetric(pts_camera_i) + Utility::skewSymmetric(tmp_r * pts_camera_i) +
                    Utility::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
            Eigen::Map<Vec2d> jacobian_feature(jacobians[3]);
            jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
        }
        if (jacobians[4])
        {
            Eigen::Map<Vec2d> jacobian_td(jacobians[4]);
            jacobian_td = reduce * ric.transpose() * Rj.transpose() * Ri * ric * velocity_i / inv_dep_i * -1.0  +
                    sqrt_info * velocity_j.head(2);
        }
    }
    sum_t += tic_toc.toc();

    return true;
*/

}


bool ProjInst12Factor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d P_bc1(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_bc1(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_bc2(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_bc2(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    double inv_dep_j = parameters[2][0];

    Mat3d R_bc1=Q_bc1.toRotationMatrix();
    //Mat3d R_cb1=R_bc1.transpose();
    Mat3d R_bc2=Q_bc2.toRotationMatrix();
    Mat3d R_cb2=R_bc2.transpose();

    Vec3d pts_cam_j=pts_j / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_cam_i=R_cb2 * R_bc1 * pts_cam_j + R_cb2 * P_bc1 - R_cb2*P_bc2;

    Eigen::Map<Vec2d> residual(residuals);

    double dep_i = pts_cam_i.z();
    double inv_dep_i=1. / dep_i;
    residual = (pts_cam_i / dep_i).head<2>() - pts_i.head<2>();

    residual = sqrt_info * residual;

    if (jacobians)
    {
        Mat23d reduce(2, 3);
        reduce <<   inv_dep_i,      0,          -pts_cam_i(0) / (dep_i * dep_i),
                    0,              inv_dep_i,  -pts_cam_i(1) / (dep_i * dep_i);
        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[0]);
            Mat36d jaco_ex;
            jaco_ex.leftCols<3>() = R_cb2;
            jaco_ex.rightCols<3>() = - R_cb2 * R_bc1 * Utility::skewSymmetric(pts_cam_j);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose1(jacobians[1]);
            Mat36d jaco_ex;
            jaco_ex.leftCols<3>() = - R_cb2;
            jaco_ex.rightCols<3>() = Utility::skewSymmetric(R_cb2*(R_bc1*pts_cam_j+P_bc1-P_bc2));
            jacobian_ex_pose1.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose1.rightCols<1>().setZero();
        }
        if (jacobians[2])
        {
            Eigen::Map<Vec2d> jacobian_feature(jacobians[2]);
            jacobian_feature = - reduce * R_cb2 * R_bc1 * pts_j  / (inv_dep_j * inv_dep_j);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}


bool ProjInst12FactorSimple::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    double inv_dep_j = parameters[0][0];

    Mat3d R_cb1 = R_bc1.transpose();

    Vec3d pts_cam_j=pts_j / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_cam_i=R_cb1 * R_bc0 * pts_cam_j + R_cb1 * P_bc0 - R_cb1*P_bc1;

    Eigen::Map<Vec2d> residual(residuals);

    double dep_i = pts_cam_i.z();
    double inv_dep_i=1. / dep_i;
    residual = (pts_cam_i / dep_i).head<2>() - pts_i.head<2>();

    residual = sqrt_info * residual;

    if (jacobians)
    {
        Mat23d reduce(2, 3);
        reduce <<   inv_dep_i,      0,          -pts_cam_i(0) / (dep_i * dep_i),
        0,              inv_dep_i,  -pts_cam_i(1) / (dep_i * dep_i);
        reduce = sqrt_info * reduce;

        if (jacobians[0])
        {
            Eigen::Map<Vec2d> jacobian_feature(jacobians[0]);
            jacobian_feature = - reduce * R_cb1 * R_bc1 * pts_j  / (inv_dep_j * inv_dep_j);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}


/**
 * 维度：<误差项大小,IMU位姿1,IMU位姿2,外参1,物体位姿1,物体位姿2,逆深度>
 */
bool ProjInst21Factor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d P_wbj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_wbj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_wbi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_wbi(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vec3d P_bc(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quatd Q_bc(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    Vec3d P_woj(parameters[3][0], parameters[3][1], parameters[3][2]);
    Quatd Q_woj(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

    Vec3d P_woi(parameters[4][0], parameters[4][1], parameters[4][2]);
    Quatd Q_woi(parameters[4][6], parameters[4][3], parameters[4][4], parameters[4][5]);

    double inv_dep_j = parameters[5][0];


    Vec3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (cur_td - td_i) * velocity_i;
    pts_j_td = pts_j - (cur_td - td_j) * velocity_j;

    Mat3d R_wbj = Q_wbj.toRotationMatrix();
    //Mat3d R_bjw=R_wbj.transpose();
    Mat3d R_wbi = Q_wbi.toRotationMatrix();
    Mat3d R_biw=R_wbi.transpose();
    Mat3d R_bc = Q_bc.toRotationMatrix();
    Mat3d R_cb=R_bc.transpose();
    Mat3d R_woj = Q_woj.toRotationMatrix();
    Mat3d R_ojw=R_woj.transpose();
    Mat3d R_woi = Q_woi.toRotationMatrix();
    //Mat3d R_oiw=R_woi.transpose();

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj * pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=R_ojw * (pts_w_j - P_woj);//k点在j时刻的物体坐标
    Vec3d pts_w_i=R_woi * pts_obj_j + P_woi;//k点在i时刻的世界坐标
    Vec3d pts_imu_i=R_biw * (pts_w_i - P_wbi);//k点在i时刻的IMU坐标
    Vec3d pts_cam_i=R_cb * (pts_imu_i - P_bc);//k点在i时刻的相机坐标

    Eigen::Map<Vec2d> residual(residuals);

    double dep_i = pts_cam_i.z();
    double inv_dep_i=1. / dep_i;
    residual = (pts_cam_i / dep_i).head<2>() - pts_i_td.head<2>();

    residual = sqrt_info * residual;

/*    if(debug_num%10==0){
        printf("lid:%d %d: depth:%.2lf,pts_j_td:(%.2lf,%.2lf,%.2lf) pts_i_td:(%.2lf,%.2lf,%.2lf) residual:(%.2lf,%.2lf) pts_imu_j:(%.2lf,%.2lf,%.2lf) pts_w_j:(%.2lf,%.2lf,%.2lf) pts_obj_j:(%.2lf,%.2lf,%.2lf) " \
        " pts_w_i:(%.2lf,%.2lf,%.2lf),pts_imu_i:(%.2lf,%.2lf,%.2lf)  pts_cam_i:(%.2lf,%.2lf,%.2lf) ",
           id, debug_num,1./inv_dep_j,
                pts_j_td.x(),pts_j_td.y(),pts_j_td.z(),pts_i_td.x(),pts_i_td.y(),pts_i_td.z(),residual.x(),residual.y(),pts_imu_j.x(),pts_imu_j.y(),pts_imu_j.z(),
               pts_w_j.x(),pts_w_j.y(),pts_w_j.z(),pts_obj_j.x(),pts_obj_j.y(),pts_obj_j.z(),pts_w_i.x(),pts_w_i.y(),pts_w_i.z(),
               pts_imu_i.x(),pts_imu_i.y(),pts_imu_i.z(),pts_cam_i.x(),pts_cam_i.y(),pts_cam_i.z());
        printf("  P_woi:(%.2lf,%.2lf,%.2lf),Q_woi:(%.2lf,%.2lf,%.2lf,%.2lf)\n",P_woi.x(),P_woi.y(),P_woi.z(),Q_woi.x(),Q_woi.y(),Q_woi.z(),Q_woi.w());
    }
    debug_num++;*/

    if (jacobians)
    {
        Mat23d reduce(2, 3);
        //计算残差相对于相机坐标的雅可比
        reduce <<   inv_dep_i,      0,          -pts_cam_i(0) / (dep_i * dep_i),
                    0,              inv_dep_i,  -pts_cam_i(1) / (dep_i * dep_i);
        reduce = sqrt_info * reduce;

        /// 计算雅可比矩阵
        //计算残差相对于p_bj和q_bj的雅可比
        if (jacobians[0])
        {
            Mat36d jaco_j;
            Mat3d temp=R_cb * R_biw * R_woi *R_ojw;
            //P_ci相对于p_wbj的导数
            jaco_j.leftCols<3>() = temp;
            //P_ci相对于q_wbj的导数
            jaco_j.rightCols<3>() = -(temp *R_wbj * Utility::skewSymmetric(pts_imu_j));

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[0]);
            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        //计算残差相对于p_bi和q_bi的雅可比
        if (jacobians[1])
        {
            Mat36d jaco_i;
            jaco_i.leftCols<3>() = -R_cb * R_biw;
            jaco_i.rightCols<3>() = R_cb * Utility::skewSymmetric(R_biw*(pts_w_i-P_wbi));

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[1]);
            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }
        //计算残差相对于p_bc和q_bc的雅可比
        if (jacobians[2])
        {
            Mat36d jaco_ex;
            Mat3d temp=R_cb * R_biw * R_woi *R_ojw *R_wbj;
            jaco_ex.leftCols<3>() = temp-R_cb;
            jaco_ex.rightCols<3>() = - temp * R_bc * Utility::skewSymmetric(pts_cam_j) + Utility::skewSymmetric(R_cb * (pts_imu_i - P_bc)) ;

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        //计算残差相对于p_woj和q_woj的雅可比
        if (jacobians[3])
        {
            Mat36d jaco_oj;
            Mat3d temp=R_cb * R_biw * R_woi;
            jaco_oj.leftCols<3>() = -temp * R_ojw;
            jaco_oj.rightCols<3>() = temp*Utility::skewSymmetric(R_ojw * (pts_w_j - P_woj));

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[3]);
            jacobian_pose_oj.leftCols<6>() = reduce * jaco_oj;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        //计算残差相对于p_woi和q_woi的雅可比
        if (jacobians[4])
        {
            Mat36d jaco_oi;
            jaco_oi.leftCols<3>() = R_cb * R_biw;
            jaco_oi.rightCols<3>() = -R_cb * R_biw * R_woi * Utility::skewSymmetric(pts_obj_j);

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_oi(jacobians[4]);
            jacobian_pose_oi.leftCols<6>() = reduce * jaco_oi;
            jacobian_pose_oi.rightCols<1>().setZero();
        }
        //计算残差相对于逆深度的雅可比
        if (jacobians[5])
        {
            Eigen::Map<Vec2d> jacobian_feature(jacobians[5]);
            jacobian_feature = reduce * R_cb * R_biw * R_woi * R_ojw * R_wbj * R_bc * pts_j_td / (inv_dep_j*inv_dep_j);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}



bool ProjInst21SimpleFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d P_woj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_woi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_woi(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    double inv_dep_j = parameters[2][0];

    Vec3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (cur_td - td_i) * velocity_i;
    pts_j_td = pts_j - (cur_td - td_j) * velocity_j;

    //Mat3d R_bjw=R_wbj.transpose();
    Mat3d R_biw=R_wbi.transpose();
    Mat3d R_cb=R_bc.transpose();
    Mat3d R_woj = Q_woj.toRotationMatrix();
    Mat3d R_ojw=R_woj.transpose();
    Mat3d R_woi = Q_woi.toRotationMatrix();
    //Mat3d R_oiw=R_woi.transpose();

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj * pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=R_ojw * (pts_w_j - P_woj);//k点在j时刻的物体坐标
    Vec3d pts_w_i=R_woi * pts_obj_j + P_woi;//k点在i时刻的世界坐标
    Vec3d pts_imu_i=R_biw * (pts_w_i - P_wbi);//k点在i时刻的IMU坐标
    Vec3d pts_cam_i=R_cb * (pts_imu_i - P_bc);//k点在i时刻的相机坐标

    Eigen::Map<Vec2d> residual(residuals);

    double dep_i = pts_cam_i.z();
    double inv_dep_i=1. / dep_i;
    residual = (pts_cam_i / dep_i).head<2>() - pts_i_td.head<2>();

    residual = sqrt_info * residual;


    if (jacobians)
    {
        Mat23d reduce(2, 3);
        //计算残差相对于相机坐标的雅可比
        reduce <<   inv_dep_i,      0,          -pts_cam_i(0) / (dep_i * dep_i),
        0,              inv_dep_i,  -pts_cam_i(1) / (dep_i * dep_i);
        reduce = sqrt_info * reduce;

        //计算残差相对于p_woj和q_woj的雅可比
        if (jacobians[0])
        {
            Mat36d jaco_oj;
            Mat3d temp=R_cb * R_biw * R_woi;
            jaco_oj.leftCols<3>() = -temp * R_ojw;
            jaco_oj.rightCols<3>() = temp*Utility::skewSymmetric(R_ojw * (pts_w_j - P_woj));

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[0]);
            jacobian_pose_oj.leftCols<6>() = reduce * jaco_oj;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        //计算残差相对于p_woi和q_woi的雅可比
        if (jacobians[1])
        {
            Mat36d jaco_oi;
            jaco_oi.leftCols<3>() = R_cb * R_biw;
            jaco_oi.rightCols<3>() = -R_cb * R_biw * R_woi * Utility::skewSymmetric(pts_obj_j);

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_oi(jacobians[1]);
            jacobian_pose_oi.leftCols<6>() = reduce * jaco_oi;
            jacobian_pose_oi.rightCols<1>().setZero();
        }
        //计算残差相对于逆深度的雅可比
        if (jacobians[2])
        {
            Eigen::Map<Vec2d> jacobian_feature(jacobians[2]);
            jacobian_feature = reduce * R_cb * R_biw * R_woi * R_ojw * R_wbj * R_bc * pts_j_td / (inv_dep_j*inv_dep_j);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}



bool ProjInst22Factor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d P_wbj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_wbj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_wbi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_wbi(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vec3d P_bc1(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quatd Q_bc1(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    Vec3d P_bc2(parameters[3][0], parameters[3][1], parameters[3][2]);
    Quatd Q_bc2(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

    Vec3d P_woj(parameters[4][0], parameters[4][1], parameters[4][2]);
    Quatd Q_woj(parameters[4][6], parameters[4][3], parameters[4][4], parameters[4][5]);

    Vec3d P_woi(parameters[5][0], parameters[5][1], parameters[5][2]);
    Quatd Q_woi(parameters[5][6], parameters[5][3], parameters[5][4], parameters[5][5]);

    double inv_dep_j = parameters[6][0];

    Vec3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (cur_td - td_i) * velocity_i;
    pts_j_td = pts_j - (cur_td - td_j) * velocity_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=Q_bc1 * pts_cam_j + P_bc1;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=Q_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=Q_woj.inverse()*(pts_w_j-P_woj);//k点在j时刻的物体坐标
    Vec3d pts_w_i=Q_woi*pts_obj_j+P_woi;//k点在i时刻的世界坐标
    Vec3d pts_imu_i=Q_wbi.inverse()*(pts_w_i-P_wbi);//k点在i时刻的IMU坐标
    Vec3d pts_cam_i=Q_bc2.inverse()*(pts_imu_i - P_bc2);//k点在i时刻的相机坐标

    Eigen::Map<Vec2d> residual(residuals);

    double dep_i = pts_cam_i.z();
    double inv_dep_i=1. / dep_i;
    residual = (pts_cam_i / dep_i).head<2>() - pts_i_td.head<2>();

    residual = sqrt_info * residual;

    if (jacobians)
    {
        Mat3d R_wbj = Q_wbj.toRotationMatrix();
        //Mat3d R_bjw=R_wbj.transpose();
        Mat3d R_wbi = Q_wbi.toRotationMatrix();
        Mat3d R_biw=R_wbi.transpose();
        Mat3d R_bc1 = Q_bc1.toRotationMatrix();
        //Mat3d R_cb1=R_bc1.transpose();
        Mat3d R_bc2 = Q_bc2.toRotationMatrix();
        Mat3d R_cb2=R_bc2.transpose();
        Mat3d R_woj = Q_woj.toRotationMatrix();
        Mat3d R_ojw=R_woj.transpose();
        Mat3d R_woi = Q_woi.toRotationMatrix();
        //Mat3d R_oiw=R_woi.transpose();

        Mat23d reduce(2, 3);
        //计算残差相对于相机坐标的雅可比
        reduce <<   inv_dep_i,   0,          -pts_cam_i(0) / (dep_i * dep_i),
                    0,           inv_dep_i,  -pts_cam_i(1) / (dep_i * dep_i);
        reduce = sqrt_info * reduce;

        /// 计算雅可比矩阵
        //计算残差相对于p_bj和q_bj的雅可比
        if (jacobians[0])
        {
            Mat36d jaco_j;
            Mat3d temp=R_cb2 * R_biw * R_woi *R_ojw;
            jaco_j.leftCols<3>() = temp;
            jaco_j.rightCols<3>() = -(temp *R_wbj * Utility::skewSymmetric(pts_imu_j));

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[0]);
            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }
        //计算残差相对于p_bi和q_bi的雅可比
        if (jacobians[1])
        {
            Mat36d jaco_i;
            jaco_i.leftCols<3>() = -R_cb2 * R_biw;
            jaco_i.rightCols<3>() = R_cb2 * Utility::skewSymmetric(R_biw*(pts_w_i-P_wbi));

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[1]);
            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }
        //计算残差相对于p_bc1和q_bc1的雅可比
        if (jacobians[2])
        {
            Mat36d jaco_ex;
            Mat3d temp=R_cb2 * R_biw * R_woi *R_ojw *R_wbj;
            jaco_ex.leftCols<3>() = temp;
            jaco_ex.rightCols<3>() = - temp * R_bc1 * Utility::skewSymmetric(pts_cam_j);

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        //计算残差相对于p_bc1和q_bc1的雅可比
        if (jacobians[3])
        {
            Mat36d jaco_ex;
            jaco_ex.leftCols<3>() = -R_cb2;
            jaco_ex.rightCols<3>() = Utility::skewSymmetric(R_cb2 * (pts_imu_i - P_bc2)) ;

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[3]);
            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }
        //计算残差相对于p_woj和q_woj的雅可比
        if (jacobians[4])
        {
            Mat36d jaco_oj;
            Mat3d temp=R_cb2 * R_biw * R_woi;
            jaco_oj.leftCols<3>() = -temp * R_ojw;
            jaco_oj.rightCols<3>() = temp*Utility::skewSymmetric(R_ojw * (pts_w_j - P_woj));

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[4]);
            jacobian_pose_oj.leftCols<6>() = reduce * jaco_oj;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        //计算残差相对于p_woi和q_woi的雅可比
        if (jacobians[5])
        {
            Mat36d jaco_oi;
            jaco_oi.leftCols<3>() = R_cb2 * R_biw;
            jaco_oi.rightCols<3>() = -R_cb2 * R_biw * R_woi * Utility::skewSymmetric(pts_obj_j);

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_oi(jacobians[5]);
            jacobian_pose_oi.leftCols<6>() = reduce * jaco_oi;
            jacobian_pose_oi.rightCols<1>().setZero();
        }
        //计算残差相对于逆深度的雅可比
        if (jacobians[6])
        {
            Eigen::Map<Vec2d> jacobian_feature(jacobians[6]);
            jacobian_feature = reduce * R_cb2 * R_biw * R_woi * R_ojw * R_wbj * R_bc1 * pts_j_td / (inv_dep_j*inv_dep_j);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}


bool ProjInst22SimpleFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    TicToc tic_toc;
    Vec3d P_woj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d P_woi(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quatd Q_woi(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    double inv_dep_j = parameters[2][0];

    Vec3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (cur_td - td_i) * velocity_i;
    pts_j_td = pts_j - (cur_td - td_j) * velocity_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc1 * pts_cam_j + P_bc1;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标
    Vec3d pts_obj_j=Q_woj.inverse()*(pts_w_j-P_woj);//k点在j时刻的物体坐标
    Vec3d pts_w_i=Q_woi*pts_obj_j+P_woi;//k点在i时刻的世界坐标
    Vec3d pts_imu_i=R_wbi.inverse()*(pts_w_i-P_wbi);//k点在i时刻的IMU坐标
    Vec3d pts_cam_i=R_bc2.inverse()*(pts_imu_i - P_bc2);//k点在i时刻的相机坐标

    Eigen::Map<Vec2d> residual(residuals);

    double dep_i = pts_cam_i.z();
    double inv_dep_i=1. / dep_i;
    residual = (pts_cam_i / dep_i).head<2>() - pts_i_td.head<2>();

    residual = sqrt_info * residual;

    if (jacobians)
    {
        //Mat3d R_bjw=R_wbj.transpose();
        Mat3d R_biw=R_wbi.transpose();
        //Mat3d R_cb1=R_bc1.transpose();
        Mat3d R_cb2=R_bc2.transpose();
        Mat3d R_woj = Q_woj.toRotationMatrix();
        Mat3d R_ojw=R_woj.transpose();
        Mat3d R_woi = Q_woi.toRotationMatrix();
        //Mat3d R_oiw=R_woi.transpose();

        Mat23d reduce(2, 3);
        //计算残差相对于相机坐标的雅可比
        reduce <<   inv_dep_i,   0,          -pts_cam_i(0) / (dep_i * dep_i),
        0,           inv_dep_i,  -pts_cam_i(1) / (dep_i * dep_i);
        reduce = sqrt_info * reduce;

        /// 计算雅可比矩阵
        //计算残差相对于p_woj和q_woj的雅可比
        if (jacobians[0])
        {
            Mat36d jaco_oj;
            Mat3d temp=R_cb2 * R_biw * R_woi;
            jaco_oj.leftCols<3>() = -temp * R_ojw;
            jaco_oj.rightCols<3>() = temp*Utility::skewSymmetric(R_ojw * (pts_w_j - P_woj));

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[0]);
            jacobian_pose_oj.leftCols<6>() = reduce * jaco_oj;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        //计算残差相对于p_woi和q_woi的雅可比
        if (jacobians[1])
        {
            Mat36d jaco_oi;
            jaco_oi.leftCols<3>() = R_cb2 * R_biw;
            jaco_oi.rightCols<3>() = -R_cb2 * R_biw * R_woi * Utility::skewSymmetric(pts_obj_j);

            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_oi(jacobians[1]);
            jacobian_pose_oi.leftCols<6>() = reduce * jaco_oi;
            jacobian_pose_oi.rightCols<1>().setZero();
        }
        //计算残差相对于逆深度的雅可比
        if (jacobians[2])
        {
            Eigen::Map<Vec2d> jacobian_feature(jacobians[2]);
            jacobian_feature = reduce * R_cb2 * R_biw * R_woi * R_ojw * R_wbj * R_bc1 * pts_j_td / (inv_dep_j*inv_dep_j);
        }
    }
    sum_t += tic_toc.toc();

    return true;
}



bool InstancePositionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d pos(parameters[0][0], parameters[0][1], parameters[0][2]);


    Vec3d pts_cam_j=pts_j * depth; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj * pts_imu_j + P_wbj;//k点在j时刻的世界坐标

    residuals[0] = (pos-pts_w_j).norm();

    //Eigen::Map<Vec3d> residual(residuals);

/*    double abs_x=std::abs(pos.x()-pts_w_j.x());
    double abs_y=std::abs(pos.y()-pts_w_j.y());
    double abs_z=std::abs(pos.z()-pts_w_j.z());

        residual.x() =abs_x;
        residual.y() =abs_y;
        residual.z() =abs_z;*/

    //residual*=10;

    if (jacobians)
    {
        if (jacobians[0])
        {
            //Mat3d jaco=Mat3d::Zero();
            /*if(pos.x()>=pts_w_j.x())
                jaco(0,0) =1;
            else
                jaco(0,0) =-1;

            if(pos.y()>=pts_w_j.y())
                jaco(1,1) =1;
            else
                jaco(1,1) =-1;

            if(pos.z()>=pts_w_j.z())
                jaco(2,2) =1;
            else
                jaco(2,2) =-1;*/

            Eigen::Matrix<double,1,3> jaco;
            jaco = (pos - pts_w_j).transpose() / residuals[0];


            Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor>> jacobian_pose_oj(jacobians[0]);
            jacobian_pose_oj.leftCols<3>() =jaco;
            //jacobian_pose_oj.rightCols<3>() = Mat3d::Zero();
            jacobian_pose_oj.rightCols<3>() = Eigen::Matrix<double,1,3>::Zero();
        }
    }

    return true;
}











bool InstanceInitAbsFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d P_woj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    double inv_dep_j = parameters[1][0];

    Vec3d pts_j_td;
    pts_j_td = pts_j - (curr_td - td_j) * velocity_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标

    const double factor=40.;

    double abs_oj_x=std::abs(P_woj.x() - pts_w_j.x());
    double abs_oj_y=std::abs(P_woj.y() - pts_w_j.y());
    double abs_oj_z=std::abs(P_woj.z() - pts_w_j.z());

    residuals[0]= abs_oj_x *factor;
    residuals[1]= abs_oj_y *factor;
    residuals[2]= abs_oj_z *factor;

    if (jacobians)
    {
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
            jaco_oj.leftCols<3>() =  jaco_init;
            jaco_oj.rightCols<3>() = Mat3d::Zero();

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[0]);
            jacobian_pose_oj.leftCols<6>() = jaco_oj *factor;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        if(jacobians[1])
        {
            Eigen::Map<Vec3d> jacobian_feature(jacobians[1]);
            //jacobian_feature = -reduce *  R_ojw * R_wbj * R_bc * pts_j / (inv_dep_j*inv_dep_j);
            jacobian_feature=Vec3d::Zero();
        }
    }

    return true;

}





bool InstanceInitPowFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d P_woj(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_woj(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    double inv_dep_j = parameters[1][0];

    Vec3d pts_j_td;
    pts_j_td = pts_j - (curr_td - td_j) * velocity_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标


    residuals[0]= (P_woj.x() - pts_w_j.x()) *(P_woj.x() - pts_w_j.x()) *factor;
    residuals[1]= (P_woj.y() - pts_w_j.y()) *(P_woj.y() - pts_w_j.y()) *factor;
    residuals[2]= (P_woj.z() - pts_w_j.z()) *(P_woj.z() - pts_w_j.z()) *factor;

    if (jacobians)
    {
        /// 对物体位姿求导
        if (jacobians[0])
        {
            Mat3d jaco_init=Mat3d::Zero();
            jaco_init(0,0) = 2. * (P_woj.x() - pts_w_j.x());
            jaco_init(1,1) = 2. * (P_woj.y() - pts_w_j.y());
            jaco_init(2,2) = 2. * (P_woj.z() - pts_w_j.z());

            Mat36d jaco_oj;
            jaco_oj.leftCols<3>() =  jaco_init;
            jaco_oj.rightCols<3>() = Mat3d::Zero();

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[0]);
            jacobian_pose_oj.leftCols<6>() = jaco_oj *factor;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        if(jacobians[1])
        {
            Eigen::Map<Vec3d> jacobian_feature(jacobians[1]);
            //jacobian_feature = -reduce *  R_ojw * R_wbj * R_bc * pts_j / (inv_dep_j*inv_dep_j);
            jacobian_feature=Vec3d::Zero();
        }
    }

    return true;

}









bool InstanceInitPowFactorSpeed::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d P_wos(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quatd Q_wos(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d speed_v(parameters[1][0], parameters[1][1], parameters[1][2]);
    Vec3d speed_a(parameters[1][3], parameters[1][4], parameters[1][5]);

    double inv_dep_j = parameters[2][0];

    Vec3d pts_j_td;
    pts_j_td = pts_j - (curr_td - td_j) * velocity_j;

    Vec3d pts_cam_j=pts_j_td / inv_dep_j; //k点在j时刻的相机坐标
    Vec3d pts_imu_j=R_bc * pts_cam_j + P_bc;//k点在j时刻的IMU坐标
    Vec3d pts_w_j=R_wbj*pts_imu_j + P_wbj;//k点在j时刻的世界坐标

    Vec3d P_oioj=time_js * speed_v;
    Mat3d R_oioj=Sophus::SO3d::exp(time_js * speed_a).matrix();
    Vec3d P_woj = R_oioj * P_wos + P_oioj;

    residuals[0]= (P_woj.x() - pts_w_j.x()) *(P_woj.x() - pts_w_j.x()) *factor;
    residuals[1]= (P_woj.y() - pts_w_j.y()) *(P_woj.y() - pts_w_j.y()) *factor;
    residuals[2]= (P_woj.z() - pts_w_j.z()) *(P_woj.z() - pts_w_j.z()) *factor;

    if (jacobians)
    {
        Mat3d jaco_init=Mat3d::Zero();
        jaco_init(0,0) = 2. * (P_woj.x() - pts_w_j.x());
        jaco_init(1,1) = 2. * (P_woj.y() - pts_w_j.y());
        jaco_init(2,2) = 2. * (P_woj.z() - pts_w_j.z());

        /// 对物体位姿求导
        if (jacobians[0])
        {
            Mat36d jaco_oj;
            jaco_oj.leftCols<3>() =  R_oioj;
            jaco_oj.rightCols<3>() = Mat3d::Zero();

            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose_oj(jacobians[0]);
            jacobian_pose_oj.leftCols<6>() = jaco_init*jaco_oj *factor;
            jacobian_pose_oj.rightCols<1>().setZero();
        }
        ///对速度进行求导
        if(jacobians[1]){
            Mat36d jaco_velocity;
            jaco_velocity.leftCols<3>() = Mat3d::Identity() * time_js;
            jaco_velocity.rightCols<3>() = -(hat(P_wos))*time_js ;

            Eigen::Map<Eigen::Matrix<double, 3, 6, Eigen::RowMajor>> jacobian_v(jacobians[1]);
            jacobian_v = jaco_init * jaco_velocity;
        }
        if(jacobians[2])
        {
            Eigen::Map<Vec3d> jacobian_feature(jacobians[2]);
            //jacobian_feature = -reduce *  R_ojw * R_wbj * R_bc * pts_j / (inv_dep_j*inv_dep_j);
            jacobian_feature=Vec3d::Zero();
        }
    }

    return true;

}


