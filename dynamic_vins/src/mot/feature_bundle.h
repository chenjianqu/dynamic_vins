/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_FEATURE_BUNDLE_H
#define DYNAMIC_VINS_FEATURE_BUNDLE_H

#include <vector>
#include <torch/torch.h>
#include "mot_parameter.h"

namespace dynamic_vins{\

/**
 * 外观特征存储器
 */
class FeatureBundle {
public:
    FeatureBundle() : full(false), next(0), store(torch::empty({mot_para::kBudget, mot_para::kFeatDim}).cuda()) {}

    void clear() {
        next = 0;
        full = false;
    }

    [[nodiscard]] bool empty() const {
        return next == 0 && !full;
    }

    /**
     * 增加一个新的特征
     * @param feat
     */
    void add(torch::Tensor feat) {
        if (next == mot_para::kBudget) {
            full = true;
            next = 0;
        }
        store[next++] = feat;
    }

    [[nodiscard]] torch::Tensor get() const {
        return full ? store : store.slice(0, 0, next);
    }

private:
    bool full;
    int64_t next;

    torch::Tensor store;
};


}

#endif //DYNAMIC_VINS_FEATURE_BUNDLE_H
