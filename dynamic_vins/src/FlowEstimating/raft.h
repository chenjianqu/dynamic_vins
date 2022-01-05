/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Dynamic_VINS.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_RAFT_H
#define DYNAMIC_VINS_RAFT_H

#include <memory>
#include <tuple>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "parameters.h"
#include "TensorRT/tensorrt_utils.h"


namespace dynamic_vins{\


class RaftData {
public:
    using Ptr = std::shared_ptr<RaftData>;
    using Tensor = torch::Tensor;

    Tensor Process(cv::Mat &img);
    Tensor Process(Tensor &img);
    Tensor Unpad(Tensor &tensor);
private:
    int h_pad,w_pad;
};


class RAFT {
public:
    using Ptr = std::unique_ptr<RAFT>;
    using Tensor = torch::Tensor;

    RAFT();
    vector<Tensor> Forward(Tensor& tensor0, Tensor& tensor1);
private:
    tuple<Tensor,Tensor> ForwardFNet(Tensor &tensor0, Tensor &tensor1);
    tuple<Tensor,Tensor> ForwardCnet(Tensor &tensor1);
    tuple<Tensor,Tensor,Tensor> ForwardUpdate(Tensor &net,Tensor &inp,Tensor &corr,Tensor &flow);
    static tuple<Tensor,Tensor> InitializeFlow(Tensor &tensor1);
    void ComputeCorrPyramid(Tensor &tensor0, Tensor &tensor1);
    Tensor IndexCorrVolume(Tensor &tensor);

    std::unique_ptr<nvinfer1::IRuntime,InferDeleter> fnet_runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> fnet_engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> fnet_context_;

    std::unique_ptr<nvinfer1::IRuntime,InferDeleter> cnet_runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> cnet_engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> cnet_context_;

    std::unique_ptr<nvinfer1::IRuntime,InferDeleter> update_runtime_;
    std::shared_ptr<nvinfer1::ICudaEngine> update_engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> update_context_;

    cudaStream_t stream_{};
    Tensor last_flow;
    vector<Tensor> corr_pyramid; //相关性金字塔
};

}

#endif //RAFT_CPP_RAFT_H
