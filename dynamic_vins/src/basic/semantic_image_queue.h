/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_SEMANTIC_IMAGE_QUEUE_H
#define DYNAMIC_VINS_SEMANTIC_IMAGE_QUEUE_H

namespace dynamic_vins{\

#include "semantic_image.h"

/**
 * 多线程图像队列
 */
class SemanticImageQueue{
public:
    void push_back(SemanticImage& img){
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(img_list.size() < kImageQueueSize){
            img_list.push_back(img);
        }
        queue_cond.notify_one();
    }

    int size(){
        std::unique_lock<std::mutex> lock(queue_mutex);
        return (int)img_list.size();
    }

    std::optional<SemanticImage> request_image() {
        std::unique_lock<std::mutex> lock(queue_mutex);
        if(!queue_cond.wait_for(lock, 30ms, [&]{return !img_list.empty();}))
            return std::nullopt;
        //queue_cond_.wait(lock,[&]{return !seg_img_list_.empty();});
        SemanticImage frame=std::move(img_list.front());
        img_list.pop_front();
        return frame;
    }

    std::mutex queue_mutex;
    std::condition_variable queue_cond;
    std::list<SemanticImage> img_list;
};


}

#endif //DYNAMIC_VINS_SEMANTIC_IMAGE_QUEUE_H
