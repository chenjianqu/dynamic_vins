//
// Created by chen on 2021/11/30.
//

#ifndef DYNAMIC_VINS_EXTRACTOR_H
#define DYNAMIC_VINS_EXTRACTOR_H

#include <memory>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>


class ExtractorNetImpl : public torch::nn::Module{
public:
    ExtractorNetImpl();

    torch::Tensor forward(torch::Tensor x);
    void load_form(const std::string &bin_path);

private:


    torch::nn::Sequential conv1,conv2;
};


TORCH_MODULE(ExtractorNet);


class Extractor {
public:
    using Ptr=std::shared_ptr<Extractor>;
    Extractor(const std::string &param_path);
    torch::Tensor extract(std::vector<cv::Mat> &input);

private:
    ExtractorNet model;
};


#endif //DYNAMIC_VINS_EXTRACTOR_H
