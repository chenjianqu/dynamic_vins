/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_SOLO_HEAD_H
#define DYNAMIC_VINS_SOLO_HEAD_H

#include <memory>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torchvision/vision.h>

#include "det2d_def.h"
#include "utils/def.h"

namespace dynamic_vins{\


class Solov2 {
public:
    using Ptr=std::shared_ptr<Solov2>;
    Solov2();
    static torch::Tensor MatrixNMS(torch::Tensor &seg_masks,torch::Tensor &cate_labels,torch::Tensor &cate_scores,torch::Tensor &sum_mask);
    cv::Mat GetSingleSeg(std::vector<torch::Tensor> &outputs, torch::Device device, std::vector<InstInfo> &insts);
    std::tuple<std::vector<cv::Mat>,std::vector<InstInfo>> GetSingleSeg(std::vector<torch::Tensor> &outputs, ImageInfo& img_info);
    void GetSegTensor(std::vector<torch::Tensor> &outputs, ImageInfo& img_info, torch::Tensor &mask_tensor, std::vector<InstInfo> &insts);

private:
    torch::Tensor size_trans_;

    bool is_resized_{true};
    bool output_split_mask_{true};
};


}

#endif //DYNAMIC_VINS_SOLO_HEAD_H
