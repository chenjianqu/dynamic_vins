//
// Created by chen on 2022/4/26.
//

#ifndef DYNAMIC_VINS_DETECTOR_DEF_H
#define DYNAMIC_VINS_DETECTOR_DEF_H

#include <string>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace dynamic_vins{\

struct ImageInfo{
    int origin_h,origin_w;
    ///图像的裁切信息
    int rect_x, rect_y, rect_w, rect_h;
};


struct InstInfo{
    std::string name;
    int label_id;
    int id;
    int track_id;
    cv::Point2f min_pt,max_pt;
    cv::Rect2f rect;
    float prob;

    cv::Point2f mask_center;

    cv::Mat mask_cv;
    cv::cuda::GpuMat mask_gpu;
    torch::Tensor mask_tensor;
};


}


#endif //DYNAMIC_VINS_DETECTOR_DEF_H
