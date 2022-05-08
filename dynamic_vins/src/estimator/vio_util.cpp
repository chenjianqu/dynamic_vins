/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "vio_util.h"

namespace dynamic_vins{\



/**
 * 特征点的三角化
 * @param Pose0
 * @param Pose1
 * @param point0
 * @param point1
 * @param point_3d
 */
void TriangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                      Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);//
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);

    Eigen::Vector4d triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

/**
 * 动态特征点的三角化
 * @param Pose0
 * @param Pose1
 * @param point0
 * @param point1
 * @param v
 * @param a
 * @param delta_t
 * @param point_3d
 */
void TriangulateDynamicPoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                             Eigen::Vector2d &point0, Eigen::Vector2d &point1,
                             Eigen::Vector3d &v, Eigen::Vector3d &a, double delta_t,
                             Eigen::Vector3d &point_3d){
    //构造T_delta
    Eigen::Matrix4d Ma;
    Ma.block<3,3>(0,0) =Sophus::SO3d::exp(a*delta_t).matrix();
    Ma.block<3,1>(0,3)=v*delta_t;
    Ma(3,3)=1;

    Eigen::Matrix<double, 2, 4> Mb;
    Mb.row(0)=point1[0] * Pose1.row(2) - Pose1.row(0);
    Mb.row(1)=point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Matrix<double,2,4> Mc = Mb * Ma;

    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);//
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = Mc.row(0);
    design_matrix.row(3) = Mc.row(1);

    Eigen::Vector4d triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


void TriangulateDynamicPoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1,
                             const Eigen::Vector2d &point0, const  Eigen::Vector2d &point1,
                             const Eigen::Matrix3d &R_woj, const Eigen::Vector3d &P_woj,
                             const Eigen::Matrix3d &R_woi, const Eigen::Vector3d &P_woi,
                             Eigen::Vector3d &point_3d){
    //构造T_delta
    Eigen::Matrix4d Ma;
    Ma.block<3,3>(0,0) =R_woi*R_woj.transpose();
    Ma.block<3,1>(0,3)=R_woi*(-R_woj.transpose() * P_woj) + P_woi;
    Ma(3,3)=1;

    Eigen::Matrix<double, 2, 4> Mb;
    Mb.row(0)=point1[0] * Pose1.row(2) - Pose1.row(0);
    Mb.row(1)=point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Matrix<double,2,4> Mc = Mb * Ma;

    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);//
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = Mc.row(0);
    design_matrix.row(3) = Mc.row(1);

    Eigen::Vector4d triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}





}