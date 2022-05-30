//
// Created by chen on 2022/5/30.
//

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
