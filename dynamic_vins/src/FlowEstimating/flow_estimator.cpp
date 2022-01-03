/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Dynamic_VINS.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "flow_estimator.h"

using Tensor = torch::Tensor;

FlowEstimator::FlowEstimator(){
    raft_ = std::make_unique<RAFT>();
    data_ = std::make_shared<RaftData>();
}

/**
 * 进行前向的光流估计
 * @param img 未经处理的图像张量，shape=[3,h,w],值范围[0-255]，数据类型Float32
 * @return 估计得到三光流张量，[2,h,w]
 */
Tensor FlowEstimator::Forward(Tensor &img) {
    auto curr_img = data_->Process(img);
    if(!last_img_.defined()){
        last_img_ = curr_img;
        auto opt = torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat32);
        return torch::zeros({2,img.sizes()[1],img.sizes()[2]},opt);
    }

    vector<Tensor> pred = raft_->Forward(last_img_,curr_img);

    auto flow = pred.back();//[1,2,h,w]
    flow = flow.squeeze();
    flow = data_->Unpad(flow);
    last_img_ = curr_img;
    return flow;
}

