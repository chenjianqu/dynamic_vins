/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_DYNAMIC_H
#define DYNAMIC_VINS_DYNAMIC_H

#include <queue>
#include <vector>
#include <unordered_map>
#include <map>
#include <chrono>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <sophus/so3.hpp>

#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include "../parameters.h"



//格式：{id, [(camera_id,feature1),...,(camera_id,featureN)]}
using FeatureMap=std::map<int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;

class InstanceFeatureSimple : public std::map<unsigned int,std::vector<Eigen::Matrix<double,5,1>>>{
public:
    cv::Scalar color;
};

//格式：{instnce_id,{feature_id,}}
using InstancesFeatureMap=std::map<unsigned int,InstanceFeatureSimple>;




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



struct FeatureFrame{
    FeatureFrame()=default;
    FeatureFrame(FeatureMap &&features_,double time_):features(features_),time(time_){} //移动构造函数
    FeatureMap features;
    double time{0.0};
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



inline Mat3d Hat(Vec3d v){
    return Sophus::SO3d::hat(v).matrix();
}


template<typename T>
void ReduceVector(std::vector<T> &v, std::vector<uchar> status){
    int j = 0;
    for (int i = 0; i < (int)v.size(); i++){
        if (status[i]) v[j++] = v[i];
    }
    v.resize(j);
}



inline bool IsInBox(Mat3d &Ri, Vec3d &Pi, Mat3d &rici, Vec3d &tici,
                    Mat3d &Roi, Vec3d &Poi, double depth, Vec3d &uv, Vec3d &box){
    Vec3d pts_w = Ri * (rici * (depth * uv) + tici) + Pi;
    Vec3d pts_oi=Roi.transpose() * ( pts_w-Poi);
    //double pts_norm=pts_oi.norm();
    //double box_norm=box.norm();
    constexpr double factor=8.;
    return !( (std::abs(pts_oi.x())>factor*box.x()) || (std::abs(pts_oi.y())>factor*box.y() ) ||
    (std::abs(pts_oi.z())>factor*box.z()) );
}


double ReprojectError(Mat3d &Ri, Vec3d &Pi, Mat3d &rici, Vec3d &tici,
                      Mat3d &Rj, Vec3d &Pj, Mat3d &ricj, Vec3d &ticj,
                      double depth, Vec3d &uvi, Vec3d &uvj);

double ReprojectDynamicError(Mat3d &Ri, Vec3d &Pi, Mat3d &rici, Vec3d &tici, Mat3d &Roi, Vec3d &Poi,
                             Mat3d &Rj, Vec3d &Pj, Mat3d &ricj, Vec3d &ticj, Mat3d &Roj, Vec3d &Poj,
                             double depth, Vec3d &uvi, Vec3d &uvj);

double ReprojectDynamicRightError(Mat3d &Ri, Vec3d &Pi, Mat3d &rici, Vec3d &tici, Mat3d &Roi, Vec3d &Poi,
                                  Mat3d &Rj, Vec3d &Pj, Mat3d &ricj, Vec3d &ticj, Mat3d &Roj, Vec3d &Poj,
                                  double depth, Vec3d &uvi, Vec3d &uvj);

Vec3d CalDelta(Vec3d &pts_j, Vec3d &pts_i, double depth, Mat3d &R_bc, Vec3d &P_bc,
               Mat3d &R_wbj, Vec3d &P_wbj, Mat3d &R_wbi, Vec3d &P_wbi,
               double td, double td_j, double td_i, Vec2d &velocity_j, Vec2d &velocity_i);



void TriangulatePoint(Mat34d &Pose0, Mat34d &Pose1, Vec2d &point0, Vec2d &point1, Vec3d &point_3d);


void TriangulateDynamicPoint(Mat34d &Pose0, Mat34d &Pose1,
                             Vec2d &point0, Vec2d &point1, Vec3d &v, Vec3d &a,
                             double delta_t, Vec3d &point_3d);



void TriangulateDynamicPoint(const Mat34d &Pose0, const Mat34d &Pose1,
                             const Vec2d &point0, const  Vec2d &point1,
                             const Mat3d &R_woj, const Vec3d &P_woj,
                             const Mat3d &R_woi, const Vec3d &P_woi,
                             Vec3d &point_3d);

void ImageTranslate(const cv::Mat &src, cv::Mat &dst, int rows_shift, int cols_shift);




#endif //DYNAMIC_VINS_DYNAMIC_H
