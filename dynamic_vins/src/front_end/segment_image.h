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
#include "box3d.h"

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

    std::vector<Box3D> boxes;

    unsigned int seq;
    bool exist_inst{false};

    void SetMask();
    void SetMaskGpu();
    void SetMaskGpuSimple();

    void SetGrayImage();
    void SetGrayImageGpu();
    void SetColorImage();
    void SetColorImageGpu();
};




}


#endif //DYNAMIC_VINS_SEGMENT_IMAGE_H
