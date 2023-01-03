/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_FEATURE_QUEUE_H
#define DYNAMIC_VINS_FEATURE_QUEUE_H

#include "def.h"

namespace dynamic_vins{\


class FeatureQueue{
public:
    using Ptr = std::shared_ptr<FeatureQueue>;

    void push_back(FrontendFeature& frame){
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(frame_list.size() < kImageQueueSize){
            frame_list.push_back(frame);
        }
        queue_cond.notify_one();
    }

    std::optional<FrontendFeature> request() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(!queue_cond.wait_for(lock, 30ms, [&]{return !frame_list.empty();}))
            return std::nullopt;
        //queue_cond_.wait(lock,[&]{return !seg_frame_list_.empty();});
        FrontendFeature frame=std::move(frame_list.front());
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
    std::list<FrontendFeature> frame_list;
};

extern FeatureQueue feature_queue;



}


#endif //DYNAMIC_VINS_INST_ESTIMATED_INFO_H
