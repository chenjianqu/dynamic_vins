/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_DETECTOR2D_H
#define DYNAMIC_VINS_DETECTOR2D_H

#include <memory>
#include <chrono>

#include "utils/def.h"
#include "utils/tensorrt/tensorrt_utils.h"
#include "pipeline.h"
#include "solo_head.h"
#include "buffer.h"
#include "front_end/semantic_image.h"
#include "det2d_parameter.h"

using namespace std::chrono_literals;


namespace dynamic_vins{\


class Detector2D {
public:
    using Ptr = std::shared_ptr<Detector2D>;
    explicit Detector2D(const std::string& config_path);

    std::tuple<std::vector<cv::Mat>,std::vector<Box2D::Ptr> > Forward(cv::Mat &img);
    void ForwardTensor(cv::Mat &img, torch::Tensor &mask_tensor, std::vector<Box2D::Ptr> &insts);
    void ForwardTensor(cv::cuda::GpuMat &img, torch::Tensor &mask_tensor, std::vector<Box2D::Ptr> &insts);
    void ForwardTensor(torch::Tensor &img, torch::Tensor &mask_tensor, std::vector<Box2D::Ptr> &insts);

    void VisualizeResult(cv::Mat &input, cv::Mat &mask, std::vector<Box2D::Ptr> &insts);


private:
    MyBuffer::Ptr buffer;
    Pipeline::Ptr pipeline_;

    Solov2::Ptr solo_;

    std::unique_ptr<nvinfer1::IRuntime,InferDeleter> runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context_;

    double infer_time_{0};
};

}

#endif //DYNAMIC_VINS_DETECTOR2D_H
