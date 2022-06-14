/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_BOX2D_H
#define DYNAMIC_VINS_BOX2D_H

#include <string>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace dynamic_vins{\

struct Box2D{
    using Ptr=std::shared_ptr<Box2D>;

    static float IoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt);

    cv::Point2f center_pt(){
        return {(min_pt.x+max_pt.x)/2.f,(min_pt.y+max_pt.y)/2.f};
    }

    std::string class_name;
    int class_id;
    int id;
    int track_id;
    cv::Point2f min_pt,max_pt;
    cv::Rect2f rect;
    float score;

    cv::Point2f mask_center;

    cv::Mat mask_cv;
    cv::cuda::GpuMat mask_gpu;
};

}


#endif //DYNAMIC_VINS_BOX2D_H
