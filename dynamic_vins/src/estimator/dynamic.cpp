/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "dynamic.h"

namespace dynamic_vins{\


/**
 * 计算两个归一化坐标i和j的重投影误差
 * @param Ri
 * @param Pi
 * @param rici i的外参
 * @param tici i的外参
 * @param Rj
 * @param Pj
 * @param ricj j的外参
 * @param ticj j的外参
 * @param depth 坐标i的深度
 * @param uvi 归一化坐标 i
 * @param uvj 归一化坐标 j
 * @return
 */
double ReprojectError(Eigen::Matrix3d &Ri, Eigen::Vector3d &Pi, Eigen::Matrix3d &rici, Eigen::Vector3d &tici,
                      Eigen::Matrix3d &Rj, Eigen::Vector3d &Pj, Eigen::Matrix3d &ricj, Eigen::Vector3d &ticj,
                      double depth, Eigen::Vector3d &uvi, Eigen::Vector3d &uvj){
    Eigen::Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Eigen::Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Eigen::Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}


/**
 *动态特征点的重投影误差
 * @param Ri
 * @param Pi
 * @param rici
 * @param tici
 * @param Roi
 * @param Poi
 * @param Rj
 * @param Pj
 * @param ricj
 * @param ticj
 * @param Roj
 * @param Poj
 * @param depth
 * @param uvi
 * @param uvj
 * @return
 */
double ReprojectDynamicError(
        Eigen::Matrix3d &Ri, Eigen::Vector3d &Pi, Eigen::Matrix3d &rici, Eigen::Vector3d &tici,
        Eigen::Matrix3d &Roi, Eigen::Vector3d &Poi,Eigen::Matrix3d &Rj, Eigen::Vector3d &Pj,
        Eigen::Matrix3d &ricj, Eigen::Vector3d &ticj,Eigen::Matrix3d &Roj, Eigen::Vector3d &Poj,
        double depth, Eigen::Vector3d &uvi, Eigen::Vector3d &uvj){
    Eigen::Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Eigen::Vector3d pts_oi=Roi.transpose() * ( pts_w-Poi);
    Eigen::Vector3d pts_wj=Roj * pts_oi + Poj;
    Eigen::Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_wj - Pj) - ticj);
    Eigen::Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}



double ReprojectDynamicRightError(
        Eigen::Matrix3d &Ri, Eigen::Vector3d &Pi, Eigen::Matrix3d &rici, Eigen::Vector3d &tici,
        Eigen::Matrix3d &Roi, Eigen::Vector3d &Poi,Eigen::Matrix3d &Rj, Eigen::Vector3d &Pj,
        Eigen::Matrix3d &ricj, Eigen::Vector3d &ticj,Eigen::Matrix3d &Roj, Eigen::Vector3d &Poj,
        double depth, Eigen::Vector3d &uvi, Eigen::Vector3d &uvj) {
    Eigen::Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Eigen::Vector3d pts_oi=Roi.transpose() * ( pts_w-Poi);
    Eigen::Vector3d pts_wj=Roj * pts_oi + Poj;
    Eigen::Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_wj - Pj) - ticj);
    Eigen::Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}





Eigen::Vector3d CalDelta(Eigen::Vector3d &pts_j, Eigen::Vector3d &pts_i, double depth, Eigen::Matrix3d &R_bc, Eigen::Vector3d &P_bc,
                         Eigen::Matrix3d &R_wbj, Eigen::Vector3d &P_wbj, Eigen::Matrix3d &R_wbi, Eigen::Vector3d &P_wbi,
                         double td, double td_j, double td_i, Eigen::Vector2d &velocity_j, Eigen::Vector2d &velocity_i){
    Eigen::Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i) * Eigen::Vector3d(velocity_i.x(),velocity_i.y(),1);
    pts_j_td = pts_j - (td - td_j) * Eigen::Vector3d(velocity_j.x(),velocity_j.y(),1);


    Eigen::Vector3d pts_cam_j=pts_j_td * depth;
    Eigen::Vector3d pts_imu_j=R_bc * pts_cam_j + P_bc;
    Eigen::Vector3d pts_w_j=R_wbj*pts_imu_j + P_wbj;

    Eigen::Vector3d pts_cam_i=pts_i_td * depth;
    Eigen::Vector3d pts_imu_i=R_bc * pts_cam_i + P_bc;
    Eigen::Vector3d pts_w_i=R_wbi*pts_imu_i + P_wbi;

    auto delta=pts_w_i-pts_w_j;


    return delta;
}





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





void ImageTranslate(const cv::Mat &src, cv::Mat &dst, int rows_shift, int cols_shift)
{
    dst=cv::Mat(src.rows,src.cols,src.type(),cv::Scalar(0));
    cv::Mat src_rect,dst_rect;
    if(rows_shift>=0 && cols_shift>=0){
        src_rect=src(cv::Range(0,src.rows-rows_shift),cv::Range(0,src.cols-cols_shift));
        dst_rect=dst(cv::Range(rows_shift,dst.rows),cv::Range(cols_shift,dst.cols));
    }
    else if(rows_shift>=0 && cols_shift<0){
        src_rect=src(cv::Range(0,src.rows-rows_shift),cv::Range(-cols_shift,src.cols));
        dst_rect=dst(cv::Range(rows_shift,dst.rows),cv::Range(0,dst.cols+cols_shift));
    }
    else if(rows_shift<0 && cols_shift>=0){
        src_rect=src(cv::Range(-rows_shift,src.rows),cv::Range(0,src.cols-cols_shift));
        dst_rect=dst(cv::Range(0,dst.rows+rows_shift),cv::Range(cols_shift,dst.cols));
    }
    else if(rows_shift<0 && cols_shift<0){
        src_rect=src(cv::Range(-rows_shift,src.rows),cv::Range(-cols_shift,src.cols));
        dst_rect=dst(cv::Range(0,dst.rows+rows_shift),cv::Range(0,dst.cols+cols_shift));
    }
    src_rect.copyTo(dst_rect);
}



}