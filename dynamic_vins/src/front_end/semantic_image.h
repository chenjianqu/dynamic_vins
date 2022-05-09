/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_SEMANTIC_IMAGE_H
#define DYNAMIC_VINS_SEMANTIC_IMAGE_H

#include <string>
#include <vector>
#include <chrono>
#include <optional>

#include <spdlog/logger.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "det2d/det2d_def.h"
#include "utils/box3d.h"
#include "utils/parameters.h"

namespace dynamic_vins{\


struct SemanticImage{
    void SetMask();
    void SetBackgroundMask();

    void SetGrayImage();
    void SetGrayImageGpu();
    void SetColorImage();
    void SetColorImageGpu();

    cv::Mat color0,seg0,color1,seg1;
    cv::cuda::GpuMat color0_gpu,color1_gpu;
    double time0,seg0_time,time1,seg1_time;
    cv::Mat gray0,gray1;
    cv::cuda::GpuMat gray0_gpu,gray1_gpu;

    torch::Tensor mask_tensor;
    std::vector<InstInfo> insts_info;

    cv::Mat merge_mask,inv_merge_mask;
    cv::cuda::GpuMat merge_mask_gpu,inv_merge_mask_gpu;

    cv::Mat flow;//光流估计结果

    std::vector<Box3D::Ptr> boxes;//3D检测结果

    torch::Tensor img_tensor;

    unsigned int seq;
    bool exist_inst{false};//当前帧是否检测到物体


};


/**
 * 多线程图像队列
 */
class ImageQueue{
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


#endif //DYNAMIC_VINS_SEMANTIC_IMAGE_H
