/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "vio_util.h"

#include "utils/log_utils.h"


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


/**
 * 使用RANSAC拟合包围框,当50%的点位于包围框内时,则满足要求
 * @param points 相机坐标系下的3D点
 * @param dims 包围框的长度
 * @return
 */
std::optional<Vec3d> FitBox3DFromObjectFrame(vector<Vec3d> &points,const Vec3d& dims)
{
    Vec3d center_pt = Vec3d::Zero();
    if(points.empty()){
        return {};
    }
    ///计算初始值
    for(auto &p:points){
        center_pt+=p;
    }
    center_pt /= (double)points.size();
    bool is_find=false;

    ///最多迭代10次
    for(int iter=0;iter<10;++iter){
        //计算每个点到中心的距离
        vector<tuple<double,Vec3d>> points_with_dist;
        for(auto &p : points){
            points_with_dist.emplace_back((p-center_pt).norm(),p);
        }
        //根据距离排序,升序
        std::sort(points_with_dist.begin(),points_with_dist.end(),[](tuple<double,Vec3d> &a,tuple<double,Vec3d> &b){
            return std::get<0>(a) < std::get<0>(b);
        });
        //选择前50%的点重新计算中心
        center_pt.setZero();
        double len = points_with_dist.size();
        for(int i=0;i<len/2;++i){
            center_pt += std::get<1>(points_with_dist[i]);
        }
        center_pt /= (len/2);
        //如前50的点位于包围框内,则退出
        if(std::get<0>(points_with_dist[int(len/2)]) <= dims.norm()){
            is_find=true;
            break;
        }
    }

    if(is_find){
        return center_pt;
    }
    else{
        return {};
    }
}





/**
 * 使用RANSAC拟合包围框,根据距离的远近删除点
 * @param points 相机坐标系下的3D点
 * @param dims 包围框的长度
 * @return
 */
std::optional<Vec3d> FitBox3DFromCameraFrame(vector<Vec3d> &points,const Vec3d& dims)
{
    Vec3d center_pt = Vec3d::Zero();
    if(points.empty()){
        return {};
    }
    std::list<Vec3d> points_rest;
    ///计算初始值
    for(auto &p:points){
        center_pt+=p;
        points_rest.push_back(p);
    }
    center_pt /= (double)points.size();
    bool is_find=false;

    double dims_norm = (dims/2.).norm();

    string log_text = "FitBox3DFromCameraFrame: \n";

    ///最多迭代10次
    for(int iter=0;iter<10;++iter){
        //计算每个点到中心的距离
        vector<tuple<double,Vec3d>> points_with_dist;
        for(auto &p : points_rest){
            points_with_dist.emplace_back((p-center_pt).norm(),p);
        }
        //根据距离排序,升序
        std::sort(points_with_dist.begin(),points_with_dist.end(),[](tuple<double,Vec3d> &a,tuple<double,Vec3d> &b){
            return std::get<0>(a) < std::get<0>(b);
        });

        //选择前80%的点重新计算中心
        center_pt.setZero();
        double len = points_with_dist.size();
        int len_used=len*0.8;
        for(int i=0;i<len_used;++i){
            center_pt += std::get<1>(points_with_dist[i]);
        }
        if(len_used<2){
            break;
        }
        center_pt /= len_used;

        log_text += fmt::format("iter:{} center_pt:{}\n",iter, VecToStr(center_pt));

        //如前80%的点位于包围框内,则退出
        if(std::get<0>(points_with_dist[len_used]) <= dims_norm){
            is_find=true;
            break;
        }

        ///只将距离相机中心最近的前50%点用于下一轮的计算
        vector<tuple<double,Vec3d>> points_cam_dist;
        for(auto &p : points_rest){
            points_cam_dist.emplace_back(p.norm(),p);
        }
        //根据距离排序,升序
        std::sort(points_cam_dist.begin(),points_cam_dist.end(),[](tuple<double,Vec3d> &a,tuple<double,Vec3d> &b){
            return std::get<0>(a) < std::get<0>(b);
        });
        points_rest.clear();
        for(int i=0;i<len*0.5;++i){
            points_rest.push_back(std::get<1>(points_cam_dist[i]));
        }

    }

    Debugv(log_text);

    if(is_find){
        return center_pt;
    }
    else{
        return {};
    }
}


/**
 * 选择距离相机最近的前30%的3D点用于计算物体中心
 * @param points
 * @param dims
 * @return
 */
std::optional<Vec3d> FitBox3DSimple(vector<Vec3d> &points,const Vec3d& dims){
    if(points.size() < 4){
        return {};
    }

    vector<tuple<double,Vec3d>> points_cam_dist;
    for(auto &p : points){
        points_cam_dist.emplace_back(p.norm(),p);
    }
    //根据距离排序,升序
    std::sort(points_cam_dist.begin(),points_cam_dist.end(),[](tuple<double,Vec3d> &a,tuple<double,Vec3d> &b){
        return std::get<0>(a) < std::get<0>(b);
    });

    double len = points_cam_dist.size();
    Vec3d center_pt = Vec3d::Zero();
    int size = len*0.3;
    for(int i=0;i<size;++i){
        center_pt += std::get<1>(points_cam_dist[i]);
    }
    center_pt /= double(size);
    return center_pt;
}



}