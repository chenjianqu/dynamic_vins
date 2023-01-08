/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_FEATURE_METRIC_H
#define DYNAMIC_VINS_FEATURE_METRIC_H

#include <vector>
#include <torch/torch.h>
#include "mot_parameter.h"

namespace dynamic_vins{\


template<typename T>
class FeatureMetric {
public:
    explicit FeatureMetric(std::vector<T> &data) : data(data) {}

    /**
     * 计算轨迹的外观和检测框外观之间两两的余弦距离
     * @param features
     * @param targets
     * @return 代价矩阵A，其中A[i,j]表示第i个轨迹和第j个检测框之间的代价
     */
    torch::Tensor distance(torch::Tensor features, const std::vector<int> &targets) {
        auto dist = torch::empty({int64_t(targets.size()), features.size(0)});
        if (features.size(0)) {
            for (size_t i = 0; i < targets.size(); ++i) {
                dist[i] = nn_cosine_distance(data[targets[i]].feats.get(), features);
            }
        }
        return dist;
    }

    /**
     * 更新轨迹的视觉特征
     * @param feats
     * @param targets
     */
    void update(torch::Tensor feats, const std::vector<int> &targets) {
        for (size_t i = 0; i < targets.size(); ++i) {
            data[targets[i]].feats.add(feats[i]);
        }
    }

    /**
     * 特征的余弦距离
     * @param x
     * @param y
     * @return
     */
    torch::Tensor nn_cosine_distance(torch::Tensor x, torch::Tensor y) {
        return std::get<0>(torch::min(1 - torch::matmul(x, y.t()), 0)).cpu();
    }

private:
    std::vector<T> &data;
};




}

#endif //DYNAMIC_VINS_FEATURE_METRIC_H
