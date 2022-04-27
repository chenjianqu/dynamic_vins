/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_SEGMENT_IMAGE_H
#define DYNAMIC_VINS_SEGMENT_IMAGE_H

#include <string>
#include <vector>
#include <chrono>

#include <spdlog/logger.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "detector/detector_def.h"

namespace dynamic_vins{\


struct SegImage{
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

    unsigned int seq;
    bool exist_inst;

    void SetMask();
    void SetMaskGpu();
    void SetMaskGpuSimple();

    void SetGrayImage();
    void SetGrayImageGpu();
    void SetColorImage();
    void SetColorImageGpu();
};


float CalBoxIoU(const cv::Point2f &box1_minPt, const cv::Point2f &box1_maxPt,
                const cv::Point2f &box2_minPt, const cv::Point2f &box2_maxPt);

float CalBoxIoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt);

template <typename T>
static std::string DimsToStr(torch::ArrayRef<T> list){
    int i = 0;
    std::string text= "[";
    for(auto e : list) {
        if (i++ > 0) text+= ", ";
        text += std::to_string(e);
    }
    text += "]";
    return text;
}


static std::string DimsToStr(cv::Size list){
    return "[" + std::to_string(list.height) + ", " + std::to_string(list.width) + "]";
}


inline cv::Point2f operator*(const cv::Point2f &lp,const cv::Point2f &rp){
    return {lp.x * rp.x,lp.y * rp.y};
}

}


#endif //DYNAMIC_VINS_SEGMENT_IMAGE_H
