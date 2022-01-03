/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_TENSORRT_UTILS_H
#define DYNAMIC_VINS_TENSORRT_UTILS_H


struct InferDeleter{
    template <typename T>
    void operator()(T* obj) const{
        if (obj)
            obj->destroy();
    }
};


#endif //DYNAMIC_VINS_TENSORRT_UTILS_H
