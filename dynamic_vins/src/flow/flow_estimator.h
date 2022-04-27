/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of Dynamic_VINS.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_FLOW_ESTIMATOR_H
#define DYNAMIC_VINS_FLOW_ESTIMATOR_H

#include <memory>
#include <atomic>
#include <torch/torch.h>

#include "utils/def.h"
#include "raft.h"

namespace dynamic_vins{\


class FlowEstimator {
public:
    using Ptr = std::unique_ptr<FlowEstimator>;
    using Tensor = torch::Tensor;

    explicit FlowEstimator(const std::string& config_path);

    Tensor Forward(Tensor &img);

    ///异步检测光流
    void SynchronizeForward(Tensor &img);
    Tensor WaitingForwardResult();

    ///异步读取光流图像
    void SynchronizeReadFlow(unsigned int seq_id);
    cv::Mat WaitingReadFlowImage();


    //摆烂了，直接读取离线估计的光流
    static cv::Mat ReadFlowImage(unsigned int seq_id);

    bool is_running(){return is_running_;}

protected:

private:
    RAFT::Ptr raft_;
    RaftData::Ptr data_;

    Tensor last_img_;

    std::thread flow_thread_;

    Tensor output;
    cv::Mat img_flow;
    std::atomic<bool> is_running_{false};
};

}

#endif //DYNAMIC_VINS_FLOW_ESTIMATOR_H
