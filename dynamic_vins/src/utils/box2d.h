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

namespace dynamic_vins{\

/**
 * 物体的ROI
 */
struct InstRoi
{
    using Ptr=std::shared_ptr<InstRoi>;

    cv::Mat mask_cv;//物体的mask
    cv::cuda::GpuMat mask_gpu;
    cv::Mat roi_gray;//物体的灰度图像
    cv::cuda::GpuMat roi_gpu;

    cv::Mat prev_roi_gray;//上一时刻的物体的灰度图像
    cv::cuda::GpuMat prev_roi_gpu;
};


struct Box2D{
    using Ptr=std::shared_ptr<Box2D>;

    static float IoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt);

    [[nodiscard]] cv::Point2f center_pt() const{
        return {(min_pt.x+max_pt.x)/2.f,(min_pt.y+max_pt.y)/2.f};
    }

    std::string class_name;
    int class_id;
    int id;
    int track_id;
    cv::Point2f min_pt,max_pt;
    cv::Rect2f rect;
    float score;

    InstRoi::Ptr roi;
};

}


#endif //DYNAMIC_VINS_BOX2D_H
