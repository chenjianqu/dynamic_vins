/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include <vector>
#include <string>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>


struct NetImpl : torch::nn::Module {
public:
    NetImpl();

    torch::Tensor forward(torch::Tensor x);

    void load_form(const std::string &bin_path);

private:
    torch::nn::Sequential conv1{nullptr}, conv2{nullptr};
};

TORCH_MODULE(Net);

class Extractor {
public:
    Extractor();

    torch::Tensor extract(std::vector<cv::Mat> input); // return GPUTensor

private:
    Net net;
};


#endif //EXTRACTOR_H
