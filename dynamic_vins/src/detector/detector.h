/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_DETECTOR_H
#define DYNAMIC_VINS_DETECTOR_H

#include <optional>
#include <memory>
#include <chrono>

#include <NvInfer.h>

#include "utils/def.h"
#include "utils/tensorrt/tensorrt_utils.h"
#include "pipeline.h"
#include "solo_head.h"
#include "buffer.h"
#include "front_end/segment_image.h"
#include "detector_parameter.h"

using namespace std::chrono_literals;


namespace dynamic_vins{\


class Detector {
public:
    using Ptr = std::shared_ptr<Detector>;
    Detector(const std::string& config_path);

    std::tuple<std::vector<cv::Mat>,std::vector<InstInfo> > Forward(cv::Mat &img);
    void ForwardTensor(cv::Mat &img, torch::Tensor &mask_tensor, std::vector<InstInfo> &insts);
    void ForwardTensor(cv::cuda::GpuMat &img, torch::Tensor &mask_tensor, std::vector<InstInfo> &insts);
    void ForwardTensor(torch::Tensor &img, torch::Tensor &mask_tensor, std::vector<InstInfo> &insts);

    void VisualizeResult(cv::Mat &input, cv::Mat &mask, std::vector<InstInfo> &insts);

    void PushBack(SegImage& img){
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if(seg_img_list_.size() < kInferImageListSize){
            seg_img_list_.push_back(img);
        }
        queue_cond_.notify_one();
    }

    int GetQueueSize(){
        std::unique_lock<std::mutex> lock(queue_mutex_);
        return (int)seg_img_list_.size();
    }

    std::optional<SegImage> WaitForResult() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if(!queue_cond_.wait_for(lock, 30ms, [&]{return !seg_img_list_.empty();}))
            return std::nullopt;
        //queue_cond_.wait(lock,[&]{return !seg_img_list_.empty();});
        SegImage frame=std::move(seg_img_list_.front());
        seg_img_list_.pop_front();
        return frame;
    }
private:
    MyBuffer::Ptr buffer;
    Pipeline::Ptr pipeline_;

    Solov2::Ptr solo_;

    std::unique_ptr<nvinfer1::IRuntime,InferDeleter> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context_;

    double infer_time_{0};

    std::mutex queue_mutex_;
    std::condition_variable queue_cond_;
    std::list<SegImage> seg_img_list_;
};

}

#endif //DYNAMIC_VINS_DETECTOR_H
