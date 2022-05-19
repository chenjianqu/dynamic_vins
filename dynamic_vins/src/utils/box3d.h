/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_BOX3D_H
#define DYNAMIC_VINS_BOX3D_H

#include <vector>
#include <string>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

#include "utils/def.h"
#include "utils/camera_model.h"


namespace dynamic_vins{\


class Box2D{
public:
    using Ptr = std::shared_ptr<Box2D>;
    Box2D() =default;
    Box2D(cv::Point2f &min_p,cv::Point2f &max_p):min_pt(min_p),max_pt(max_p){
        center_pt = (min_pt+max_pt)/2;
    }

    cv::Point2f min_pt,max_pt;//边界框的两个点
    cv::Point2f center_pt;
};


class Box3D{
public:
    using Ptr = std::shared_ptr<Box3D>;

    Box3D()=default;

    Box3D(int class_id_,int attribution_id_,double score_)
    :class_id(class_id_),attribution_id(attribution_id_),score(score_){}

    bool InsideBox(Eigen::Vector3d &point);

    Mat28d CornersProjectTo2D(PinHoleCamera &cam);

    static std::vector<std::pair<int,int>> GetLineVetexPair();

    void VisCorners2d(cv::Mat &img,const cv::Scalar& color,PinHoleCamera &cam);

    Mat38d GetCornersInWorld(const Mat3d &R_wbi,const Vec3d &P_wbi,const Mat3d &R_bc,const Vec3d &P_bc);

    static VecVector3d GetCoordinateVectorFromCorners(Mat38d &corners);

    static Mat3d GetCoordinateRotationFromCorners(Mat38d &corners);

    static int CoordinateDirection(int x_d,int y_d,int z_d);


    ///每行的前3个数字是类别,属性,分数
    int class_id;
    int attribution_id ;
    double score;

    Vec3d bottom_center{0,0,0};//单目3D目标检测算法预测的包围框底部中心(在相机坐标系下)
    Vec3d dims{0,0,0};//预测的大小
    double yaw{0};//预测的yaw角(沿着垂直向下的z轴)

    Eigen::Matrix<double,3,8> corners;//包围框的8个顶点在相机坐标系下的坐标
    Vec3d center{0,0,0};//包围框中心坐标

    Eigen::Matrix<double,2,8> corners_2d;////包围框的8个顶点在图像坐标系下的像素坐标
    Box2D box2d;
};



}

#endif //DYNAMIC_VINS_BOX3D_H
