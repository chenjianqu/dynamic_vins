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
#include <torch/torch.h>

#include "raft.h"

class FlowEstimator {
public:
    using Ptr = std::unique_ptr<FlowEstimator>;
    using Tensor = torch::Tensor;

    FlowEstimator();

    Tensor Forward(Tensor &img);

protected:

private:
    RAFT::Ptr raft_;
    RaftData::Ptr data_;

    Tensor last_img_;
};


#endif //DYNAMIC_VINS_FLOW_ESTIMATOR_H
