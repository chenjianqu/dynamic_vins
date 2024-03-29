/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_EXTRACTOR_H
#define DYNAMIC_VINS_EXTRACTOR_H

#include <vector>
#include <string>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "reid_net.h"
#include "trace_reid_net.h"

namespace dynamic_vins{\


class Extractor {
public:
    Extractor();

    torch::Tensor extract(std::vector<cv::Mat> input); // return GPUTensor

private:
    //ReIdNet net;
    TraceReidNet net;
};


}

#endif //DYNAMIC_VINS_EXTRACTOR_H
