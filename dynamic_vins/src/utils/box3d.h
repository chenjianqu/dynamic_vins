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

#include <utility>
#include <vector>
#include <string>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

#include "camodocal/camera_models/CameraFactory.h"

#include "utils/def.h"
#include "utils/camera_model.h"


namespace dynamic_vins{\


class Rect2D{
public:
    using Ptr = std::shared_ptr<Rect2D>;
    Rect2D() =default;
    Rect2D(cv::Point2f &min_p, cv::Point2f &max_p): min_pt(min_p), max_pt(max_p){
        center_pt = (min_pt+max_pt)/2;
    }

    cv::Point2f min_pt,max_pt;//边界框的两个点
    cv::Point2f center_pt;
};


class Box3D{
public:
    using Ptr = std::shared_ptr<Box3D>;

    Box3D()=default;

    Box3D(int class_id_,string class_name_,int attribution_id_,double score_)
    :class_id(class_id_),class_name(std::move(class_name_)),attribution_id(attribution_id_),score(score_){}

    Box3D(int class_id_,string class_name_,double score_)
    :class_id(class_id_),class_name(std::move(class_name_)),score(score_){}

    [[nodiscard]] Mat34d GetCoordinateVectorInCamera(double axis_len=1.) const;

    bool InsideBox(Eigen::Vector3d &point);

    Mat28d CornersProjectTo2D(camodocal::CameraPtr &cam);

    static std::vector<std::pair<int,int>> GetLineVetexPair();

    void VisCorners2d(cv::Mat &img,const cv::Scalar& color,camodocal::CameraPtr &cam);

    static Mat38d GetCorners(Vec3d &dims,Mat3d &R_xo,Vec3d &P_xo);

    static Box3D::Ptr Box3dFromFCOS3D(vector<string> &tokens,camodocal::CameraPtr &cam);

    static Box3D::Ptr Box3dFromKittiTracking(vector<string> &tokens,camodocal::CameraPtr &cam);

    static VecVector3d GetCoordinateVectorFromCorners(Mat38d &corners);

    static Mat3d GetCoordinateRotationFromCorners(Mat38d &corners);

    static int CoordinateDirection(int x_d,int y_d,int z_d);

    static Mat38d GetCornersFromPose(Mat3d &R_woi,Vec3d &P_woi,Vec3d &dims);

    /**
     * 根据yaw角构造物体位姿的旋转矩阵
     * @return
     */
    [[nodiscard]] Mat3d R_cioi() const{
        Mat3d R;
        R<<cos(yaw),0, -sin(yaw),   0,1,0,   sin(yaw),0,cos(yaw);
        return R.transpose();
    }

    ///每行的前3个数字是类别,属性,分数
    unsigned int id{};
    int class_id{};
    string class_name{};
    int attribution_id{};
    double score{};
    int frame;

    Vec3d bottom_center{0,0,0};//单目3D目标检测算法预测的包围框底部中心(在相机坐标系下)
    Vec3d dims{0,0,0};//预测的大小
    double yaw{0};//预测的yaw角(沿着垂直向下的y轴)

    Eigen::Matrix<double,3,8> corners;//包围框的8个顶点在相机坐标系下的坐标
    Vec3d center_pt{0, 0, 0};//包围框中心坐标

    Eigen::Matrix<double,2,8> corners_2d;//包围框的8个顶点在图像坐标系下的像素坐标
    Rect2D box2d;
};



}

#endif //DYNAMIC_VINS_BOX3D_H
