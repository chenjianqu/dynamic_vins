/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_FLOW_VISUAL_H
#define DYNAMIC_VINS_FLOW_VISUAL_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

namespace dynamic_vins{\


torch::Tensor FlowToImage(torch::Tensor &flow_uv);
cv::Mat VisualFlow(torch::Tensor &img, torch::Tensor &flow_uv);
cv::Mat VisualFlow(torch::Tensor &flow_uv);

}

#endif //RAFT_CPP_VISUALIZATION_H
