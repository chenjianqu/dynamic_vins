/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "trace_reid_net.h"
#include <torch/script.h>
#include "utils/log_utils.h"


torch::Tensor TraceReidNet::forward(torch::Tensor x){
    return model.forward({ x }).toTensor();
}

torch::Tensor TraceReidNet::operator()(torch::Tensor x){
    return forward(x);
}

void TraceReidNet::load_form(const std::string &model_path){

    dynamic_vins::Debugv("ReId model path:{}",model_path);

    model = torch::jit::load(model_path);
    model.eval();
    model.to(at::kCUDA);
}



