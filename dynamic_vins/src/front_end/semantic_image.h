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

#include "utils/box3d.h"
#include "utils/box2d.h"
#include "utils/parameters.h"

namespace dynamic_vins{\


struct SemanticImage{
    SemanticImage()= default;

    void SetMaskAndRoi();
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
    std::vector<Box2D::Ptr> boxes2d;

    cv::Mat merge_mask,inv_merge_mask;
    cv::cuda::GpuMat merge_mask_gpu,inv_merge_mask_gpu;

    cv::Mat flow;//光流估计结果

    cv::Mat disp;//视差图

    std::vector<Box3D::Ptr> boxes3d;//3D检测结果

    torch::Tensor img_tensor;

    unsigned int seq;
    bool exist_inst{false};//当前帧是否检测到物体
};



}


#endif //DYNAMIC_VINS_SEMANTIC_IMAGE_H
