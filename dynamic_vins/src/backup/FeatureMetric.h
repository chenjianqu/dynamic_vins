//
// Created by chen on 2021/11/30.
//

#ifndef DYNAMIC_VINS_FEATUREMETRIC_H
#define DYNAMIC_VINS_FEATUREMETRIC_H

#include <memory>
#include <cfloat>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>


/**
 * 实例的特征保存模块
 */
class FeatureBundle {
public:
    FeatureBundle() :
    store(torch::empty({BUDGET, FEAT_DIM},torch::kCUDA)),
    full(false),
    next(0)
    {}

    /**
     * 清除特征列表
     */
    void clear() {
        next = 0;
        full = false;
    }

    /**
     * 实例的特征列表是否为空
     * @return
     */
    [[nodiscard]] bool empty() const {
        return next == 0 && !full;
    }

    /**
     * 将当前帧特征添加到实例的特征列表中
     * @param feat
     */
    void add(torch::Tensor feat) {
        if (next == BUDGET) {
            full = true;
            next = 0;
        }
        store[next++] = std::move(feat);
    }

    /**
     * 获取某个实例的全部特征
     * @return
     */
    [[nodiscard]] torch::Tensor get() const {
        return full ? store : store.slice(0, 0, next);
    }

    static constexpr int64_t BUDGET = 100;
    static constexpr int64_t FEAT_DIM = 512;//特征的维度
private:
    torch::Tensor store{};//存储每个实例在不同时刻的特征
    bool full;//特征列表是否已经满了
    int64_t next;//当前特征的游标
};







template<typename TrackData>
class FeatureMetric {
public:
    using Ptr=std::unique_ptr<FeatureMetric>;

    explicit FeatureMetric(std::vector<TrackData> &data) : data(data) {}

    /**
     * 计算当前检测的box和跟踪的box之间的特征距离
     * @param features 当前帧box之间的特征
     * @param targets 正在跟踪的box的索引
     * @return 返回一个二维张量，每个元素表示box之间的特征的余弦距离
     */
    torch::Tensor distance(torch::Tensor &features, const std::vector<int> &targets) {
        auto dist = torch::empty({int64_t(targets.size()), features.size(0)});
        if (features.size(0)) {
            for (int64_t i = 0; i < targets.size(); ++i) {
                dist[i] = nn_cosine_distance(data[targets[i]].feats.get(), features);
            }
        }
        return dist;
    }

    /**
     * 更新正在跟踪的实例的特征
     * @param feats 新检测的特征
     * @param targets 要添加到的实例索引
     */
    void update( torch::Tensor feats,  std::vector<int> targets) {
        for (int64_t i = 0; i < targets.size(); ++i) {
            data[targets[i]].feats.add(feats[i]);
        }
    }

private:
    std::vector<TrackData> &data;

    /**
     * 计算两个特征的cosine距离
     * @param x
     * @param y
     * @return
     */
    torch::Tensor nn_cosine_distance(const torch::Tensor& x, const torch::Tensor &y) {
        return std::get<0>(torch::min(1 - torch::matmul(x, y.t()), 0)).cpu();
    }
};


#endif //DYNAMIC_VINS_FEATUREMETRIC_H
