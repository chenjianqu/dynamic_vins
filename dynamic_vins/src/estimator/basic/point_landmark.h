/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_POINT_LANDMARK_H
#define DYNAMIC_VINS_POINT_LANDMARK_H


#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <sophus/so3.hpp>

#include "utils/def.h"
#include "estimator/basic/point_feature.h"

namespace dynamic_vins{\


struct LandmarkPoint{
    explicit LandmarkPoint(unsigned int id_):id(id_){}

    FeaturePoint::Ptr front(){
        return feats.front();
    }

    [[nodiscard]] int frame() const{
        return feats.front()->frame;
    }

    [[nodiscard]] int size() const{
        return feats.size();
    }

    [[nodiscard]] bool is_extra(){
        return feats.front()->is_extra;
    }

    void EraseBegin(){
        erase(feats.begin());
    }

    /**
     * 删除路标点的某个观测,同时设置空路标点
     * @param it
     */
    void erase(std::list<FeaturePoint::Ptr>::iterator it){
        feats.erase(it);

        if(feats.empty()){
            bad=true;
        }
    }

    void erase(std::list<FeaturePoint::Ptr>::iterator left,std::list<FeaturePoint::Ptr>::iterator right){
        feats.erase(left,right);

        if(feats.empty()){
            bad=true;
        }
    }

    FeaturePoint::Ptr& operator[](int index){
        auto it=feats.begin();
        std::advance(it,index);
        return *it;
    }

    bool bad{false};//坏点

    unsigned int id;
    std::list<FeaturePoint::Ptr> feats;//每个id的特征在各个帧上的特征点,因为要经常删除，故使用链表
    double depth{-1.0};
};




class StaticLandmark{
public:
    const int feature_id;
    int start_frame;
    vector<StaticFeature> feats;
    size_t used_num;
    double depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    StaticLandmark(int _feature_id, int _start_frame)
    : feature_id(_feature_id), start_frame(_start_frame),
    used_num(0), depth(-1.0), solve_flag(0)
    {}

    int endFrame();
};



}

#endif //DYNAMIC_VINS_POINT_LANDMARK_H
