/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_POINT_FEATURE_H
#define DYNAMIC_VINS_POINT_FEATURE_H

#include "def.h"

namespace dynamic_vins{\


/**
 * 特征点(单次观测)
 */
struct FeaturePoint{
    using Ptr=std::shared_ptr<FeaturePoint>;

    FeaturePoint()=default;
    /**
     * 单目特征点初始化
     * @param point_
     * @param frame_
     */
    FeaturePoint(cv::Point2f &point_,int frame_)
    :point(point_.x,point_.y,1), is_stereo(false), frame(frame_){}

    /**
     * 双目特征点初始化
     * @param point_
     * @param point_right_
     * @param frame_
     */
    FeaturePoint(cv::Point2f &point_,cv::Point2f &point_right_,int frame_)
    :point(point_.x,point_.y,1),point_right(point_right_.x, point_right_.y, 1), is_stereo(true), frame(frame_){
    }

    /**
     * 根据Eigen::Vector3d来进行初始化
     * @param feat_vector
     * @param frame_
     */
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

    /**
     * 根据Eigen::Vector5d来进行初始化
     * @param feat_vector
     * @param frame_
     * @param td
     */
    FeaturePoint(std::vector<Eigen::Matrix<double,5,1>> &feat_vector,int frame_,double td):
    frame(frame_), td(td){
        if(feat_vector.size()==2){
            point=feat_vector[0].topRows(3);
            //vel=feat_vector[0].bottomRows(2);
            point_right=feat_vector[1].topRows(3);
            //vel_right=feat_vector[1].bottomRows(2);
            is_stereo=true;
        }
        else{
            point=feat_vector[0].topRows(3);
            //vel=feat_vector[0].bottomRows(2);
            is_stereo=false;
        }
    }

    Vec3d point{0,0,0};//归一化坐标
    Vec3d point_right{0,0,0};

    bool is_stereo{false};
    bool is_extra{false};

    int frame{0};

    Vec2d vel{0,0};//左相机归一化点的速度
    Vec2d vel_right{0,0};//右相机归一化点的速度

    double td{0};//特征点被构建时的时间偏移量

    Vec3d p_w{0,0,0};//该观测在世界坐标系下的坐标
    bool is_triangulated{false};//当前观测是否已经进行了三角化

    float disp{0.f};//视差
};


}

#endif //DYNAMIC_VINS_POINT_FEATURE_H
