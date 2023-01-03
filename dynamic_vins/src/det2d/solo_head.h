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

#include "basic/def.h"
#include "pipeline.h"

namespace dynamic_vins{\


class Solov2 {
public:
    using Ptr=std::shared_ptr<Solov2>;
    Solov2();
    static torch::Tensor MatrixNMS(torch::Tensor &seg_masks,torch::Tensor &cate_labels,
                                   torch::Tensor &cate_scores,torch::Tensor &sum_mask);
    cv::Mat GetSingleSeg(std::vector<torch::Tensor> &outputs, torch::Device device,
                         std::vector<Box2D::Ptr> &insts);
    std::tuple<std::vector<cv::Mat>,std::vector<Box2D::Ptr>> GetSingleSeg(std::vector<torch::Tensor> &outputs,
                                                                     ImageInfo& img_info);
    void GetSegTensor(std::vector<torch::Tensor> &outputs, ImageInfo& img_info, torch::Tensor &seg_label_out,
                      torch::Tensor &cate_labels_out,torch::Tensor &cate_scores_out);

private:
    torch::Tensor size_trans_;

    bool is_resized_{true};
    bool output_split_mask_{true};
};


}

#endif //DYNAMIC_VINS_SOLO_HEAD_H
