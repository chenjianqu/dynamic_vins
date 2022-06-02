/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_SEMANTIC_FEATURE_H
#define DYNAMIC_VINS_SEMANTIC_FEATURE_H

#include <optional>
#include <mutex>
#include <thread>
#include <condition_variable>

#include "utils/def.h"
#include "utils/box2d.h"
#include "utils/box3d.h"
#include "utils/parameters.h"

namespace dynamic_vins{\


/*
 * 格式：{id, [(camera_id,feature1),...,(camera_id,featureN)]}
 * feature1：Vector7d，分别表示
 */
using FeatureBackground=std::map<unsigned int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>;

class FeatureInstance : public std::map<unsigned int,std::vector<Eigen::Matrix<double,5,1>>>{
public:
    cv::Scalar color;
    Box2D::Ptr box2d;
    Box3D::Ptr box3d;
};


/**
 * 用于在前端和VIO之间传递信息
 */
struct SemanticFeature{
    SemanticFeature()=default;
    ///背景特征点
    FeatureBackground features;
    double time{0.0};

    unsigned int seq_id;//帧号

    ///根据物体的实例信息,格式：{instnce_id,{feature_id,}}
    std::map<unsigned int,FeatureInstance> instances;

};


class InstEstimatedInfo{
public:
    double time;
    Mat3d R;
    Vec3d P{0,0,0};
    Vec3d v{0,0,0};
    Vec3d a{0,0,0};

    Vec3d dims{0,0,0};
    Vec3d avg_point{0,0,0};

    bool is_init{false};
    bool is_init_velocity{false};
};



class FeatureQueue{
public:
    using Ptr = std::shared_ptr<FeatureQueue>;

    void push_back(SemanticFeature& frame){
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(frame_list.size() < kImageQueueSize){
            frame_list.push_back(frame);
        }
        queue_cond.notify_one();
    }

    std::optional<SemanticFeature> request_frame() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(!queue_cond.wait_for(lock, 30ms, [&]{return !frame_list.empty();}))
            return std::nullopt;
        //queue_cond_.wait(lock,[&]{return !seg_frame_list_.empty();});
        SemanticFeature frame=std::move(frame_list.front());
        frame_list.pop_front();
        return frame;
    }

    int size(){
        std::unique_lock<std::mutex> lock(queue_mutex);
        return (int)frame_list.size();
    }

    bool empty(){
        std::unique_lock<std::mutex> lock(queue_mutex);
        return frame_list.empty();
    }

    void clear(){
        std::unique_lock<std::mutex> lock(queue_mutex);
        frame_list.clear();
    }

    std::optional<double> front_time(){
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(frame_list.empty()){
            return std::nullopt;
        }
        else{
            return frame_list.front().time;
        }
    }


private:
    std::mutex queue_mutex;
    std::condition_variable queue_cond;
    std::list<SemanticFeature> frame_list;
};

extern FeatureQueue feature_queue;


}

#endif //DYNAMIC_VINS_SEMANTIC_FEATURE_H
