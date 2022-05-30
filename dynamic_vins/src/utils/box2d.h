//
// Created by chen on 2022/5/30.
//

#ifndef DYNAMIC_VINS_BOX2D_H
#define DYNAMIC_VINS_BOX2D_H

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace dynamic_vins{\

struct Box2D{
    using Ptr=std::shared_ptr<Box2D>;

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
    torch::Tensor mask_tensor;
};

}


#endif //DYNAMIC_VINS_BOX2D_H
