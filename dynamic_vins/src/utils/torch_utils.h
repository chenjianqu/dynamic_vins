/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_TORCH_UTILS_H
#define DYNAMIC_VINS_TORCH_UTILS_H

#include <string>
#include <torch/torch.h>

namespace dynamic_vins{\

template <typename T>
static std::string DimsToStr(torch::ArrayRef<T> list){
    int i = 0;
    std::string text= "[";
    for(auto e : list) {
        if (i++ > 0) text+= ", ";
        text += std::to_string(e);
    }
    text += "]";
    return text;
}



}

#endif //DYNAMIC_VINS_TORCH_UTILS_H
