/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_LANDMARK_H
#define DYNAMIC_VINS_LANDMARK_H


#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <sophus/so3.hpp>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include "utils/def.h"

namespace dynamic_vins{\


struct Vel3d{
    Vel3d(){
        v=Vec3d::Zero();
        a=Vec3d::Zero();
    }
    void SetZero(){
        v=Vec3d::Zero();
        a=Vec3d::Zero();
    }
    Vec3d v;
    Vec3d a;
};


struct Vel : Vel3d{
    Vel()=default;
    Vec2d img_v;
};


struct FeaturePoint{
    FeaturePoint()=default;

    FeaturePoint(cv::Point2f &point_,int frame_):
    point(point_.x,point_.y,1), is_stereo(false), frame(frame_){}

    FeaturePoint(cv::Point2f &point_,cv::Point2f &point_right_,int frame_):
    point(point_.x,point_.y,1), is_stereo(true), point_right(point_right_.x, point_right_.y, 1), frame(frame_){}

    FeaturePoint(std::vector<Vec3d> &feat_vector,int frame_):frame(frame_){
        if(feat_vector.size()==2){
            point=feat_vector[0];
            point_right=feat_vector[1];
            is_stereo=true;
        }
        else{
            point=feat_vector[0];
            is_stereo=false;
        }
    }

    FeaturePoint(std::vector<Eigen::Matrix<double,5,1>> &feat_vector,int frame_,double td):
    frame(frame_), td(td){
        if(feat_vector.size()==2){
            point=feat_vector[0].topRows(3);
            //vel=feat_vector[0].bottomRows(2);
            vel=Vec2d::Zero();
            point_right=feat_vector[1].topRows(3);
            //vel_right=feat_vector[1].bottomRows(2);
            vel_right=Vec2d::Zero();
            is_stereo=true;
        }
        else{
            point=feat_vector[0].topRows(3);
            //vel=feat_vector[0].bottomRows(2);
            vel=Vec2d::Zero();
            is_stereo=false;
        }
    }

    Vec3d point;//归一化坐标
    bool is_stereo{false};
    Vec3d point_right;
    int frame{0};

    Vec2d vel;//左相机归一化点的速度
    Vec2d vel_right;//右相机归一化点的速度

    double td{};///特征点被构建时的时间偏移量
};



struct LandmarkPoint{
    explicit LandmarkPoint(unsigned int id_):id(id_){}
    unsigned int id;
    std::vector<FeaturePoint> feats;//每个id的特征在各个帧上的特征点,因为要经常删除，故使用链表
    double depth{-1.0};
};



struct State{
    void swap(State &rstate){
        State temp=rstate;
        rstate.R=std::move(R);
        rstate.P=std::move(P);
        rstate.time=time;
        R=std::move(temp.R);
        P=std::move(temp.P);
        time=temp.time;
    }
    Mat3d R;
    Vec3d P;
    double time;
};

}

#endif //DYNAMIC_VINS_LANDMARK_H
