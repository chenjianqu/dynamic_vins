/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "extractor.h"
#include "mot_parameter.h"

namespace dynamic_vins{\


Extractor::Extractor() {
    net->load_form(mot_para::kExtractorModelPath);
    net->to(torch::kCUDA);
    net->eval();
}


/**
 * 执行外观特征提取
 * @param input
 * @return
 */
torch::Tensor Extractor::extract(std::vector<cv::Mat> input) {
    if (input.empty()) {
        return torch::empty({0, 512});
    }

    torch::NoGradGuard no_grad;

    static torch::Tensor mean_t=torch::from_blob(mot_para::kReIdImgMean, {3, 1, 1}, torch::kFloat32).to(torch::kCUDA).view({1, -1, 1, 1});
    static torch::Tensor std_t=torch::from_blob(mot_para::kReIdImgStd, {3, 1, 1}, torch::kFloat32).to(torch::kCUDA).view({1, -1, 1, 1});

    constexpr int kWidth=64;
    constexpr int kHeight=128;

    std::vector<torch::Tensor> resized;
    for (auto &x:input) {
        cv::resize(x, x, {kWidth, kHeight});
        cv::cvtColor(x, x, cv::COLOR_RGB2BGR);
        x.convertTo(x, CV_32F, 1.0 / 255);
        resized.push_back(torch::from_blob(x.data, {kHeight, kWidth, 3}));
    }
    auto tensor = torch::stack(resized).cuda().permute({0, 3, 1, 2}).sub_(mean_t).div_(std_t);

    ///执行推断
    return net(tensor);
}


}