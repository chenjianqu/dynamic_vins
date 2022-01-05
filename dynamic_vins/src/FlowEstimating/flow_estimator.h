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

#include "raft.h"

namespace dynamic_vins{\


class FlowEstimator {
public:
    using Ptr = std::unique_ptr<FlowEstimator>;
    using Tensor = torch::Tensor;

    FlowEstimator();

    Tensor Forward(Tensor &img);

    ///异步检测光流
    void StartForward(Tensor &img){
        if(is_running_){
            return;
        }
        flow_thread_ = std::thread([this](torch::Tensor &img){
            this->is_running_=true;
            this->output = this->Forward(img);
            this->is_running_=false;
            }
            ,std::ref(img));
    }

    Tensor WaitingResult(){
        flow_thread_.join();
        return output;
    }

    bool is_running(){return is_running_;}

protected:

private:
    RAFT::Ptr raft_;
    RaftData::Ptr data_;

    Tensor last_img_;

    std::thread flow_thread_;

    Tensor output;
    std::atomic<bool> is_running_{false};
};

}

#endif //DYNAMIC_VINS_FLOW_ESTIMATOR_H
