/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_TENSORRT_UTILS_H
#define DYNAMIC_VINS_TENSORRT_UTILS_H

#include <string>

namespace dynamic_vins{\

struct InferDeleter{
    template <typename T> void operator()(T* obj) const{
        if (obj)
            obj->destroy();
    }
};

int BuildTensorRT(const std::string &onnx_path,const std::string &tensorrt_path);


}


#endif //DYNAMIC_VINS_TENSORRT_UTILS_H
