/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_TRACKER_H
#define DYNAMIC_VINS_TRACKER_H

#include <vector>
#include <torch/torch.h>
#include <tuple>

#include "kalman_tracker.h"
#include "basic/def.h"
#include "basic/box2d.h"
#include "mot_def.h"

namespace dynamic_vins{\

using DistanceMetricFunc = std::function<torch::Tensor(const std::vector<int> &trk_ids, const std::vector<int> &det_ids)>;

const float INVALID_DIST = 1E3f;

void AssociateDetectionsToTrackersIdx(const DistanceMetricFunc &metric,
                                      std::vector<int> &unmatched_trks,
                                      std::vector<int> &unmatched_dets,
                                      std::vector<std::tuple<int, int>> &matched);

template<typename T>
class TrackerManager {
public:
    explicit TrackerManager(std::vector<T> &data, const std::array<int64_t, 2> &dim)
    : data(data), img_box(0, 0, dim[1], dim[0]) {}

    void predict() {
        for (auto &t:data) {
            t.kalman.predict();
        }
    }

    void remove_nan() {
        data.erase(remove_if(data.begin(), data.end(),[](const T &t) {
            auto bbox = t.kalman.rect();
            return std::isnan(bbox.x) || std::isnan(bbox.y) ||
            std::isnan(bbox.width) || std::isnan(bbox.height);
        }),data.end());
    }

    void remove_deleted() {
        data.erase(remove_if(data.begin(), data.end(),
                             [this](const T &t) {
            return t.kalman.state() == TrackState::Deleted;
        }), data.end());
    }


    /**
     * 执行关联
     * @param dets
     * @param confirmed_metric 外观代价计算函数
     * @param unconfirmed_metric IoU代价计算函数
     * @return 匹配结果v，其中v[k]是一对匹配{i,j}，表示第i个轨迹匹配第j个检测框
     */
    std::vector<std::tuple<int, int>> update(const std::vector<Box2D::Ptr> &dets,
                                             const DistanceMetricFunc &confirmed_metric,
                                             const DistanceMetricFunc &unconfirmed_metric) {
        ///卡尔滤波器中的轨迹
        std::vector<int> unmatched_trks;
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i].kalman.state() == TrackState::Confirmed) {
                unmatched_trks.emplace_back(i);
            }
        }

        ///给每个输入的检测框一个编号
        std::vector<int> unmatched_dets(dets.size());
        iota(unmatched_dets.begin(), unmatched_dets.end(), 0);

        std::vector<std::tuple<int, int>> matched;

        ///根据confirmed_metric计算的代价矩阵，对unmatched_trks和unmatched_dets进行关联，得到关联结果matched。并更新unmatched_trks和unmatched_dets
        AssociateDetectionsToTrackersIdx(confirmed_metric, unmatched_trks, unmatched_dets, matched);

        ///将试用期的轨迹也加入unmatched_trks
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i].kalman.state() == TrackState::Tentative) {
                unmatched_trks.emplace_back(i);
            }
        }

        ///根据unconfirmed_metricIoU代价再次匹配剩下的
        AssociateDetectionsToTrackersIdx(unconfirmed_metric, unmatched_trks, unmatched_dets, matched);

        ///若还是匹配不上，则miss
        for (auto i : unmatched_trks) {
            data[i].kalman.miss();
        }

        ///更新卡尔曼滤波器的轨迹
        for (auto[x, y] : matched) {
            data[x].kalman.update(dets[y]->rect);
            data[x].info = dets[y];
        }

        ///对未匹配的检测框创建新的轨迹
        for (auto umd : unmatched_dets) {
            matched.emplace_back(data.size(), umd);
            auto t = T{};
            t.kalman.init(dets[umd]->rect);
            t.info = dets[umd];
            data.emplace_back(t);
        }
        return matched;
    }


    std::vector<Track> visible_tracks() {
        std::vector<Track> ret;
        for (auto &t : data) {
            auto bbox = t.kalman.rect();
            if (t.kalman.state() == TrackState::Confirmed &&
            img_box.contains(bbox.tl()) && img_box.contains(bbox.br())) {
                Track res{t.kalman.id(), bbox};
                ret.push_back(res);
            }
        }
        return ret;
    }


    std::vector<Box2D::Ptr> visible_tracks_info() {
        std::vector<Box2D::Ptr> ret;
        for (auto &t : data) {
            if(t.kalman.get_time_since_update()>0){ //若当前无观测，则跳过
                continue;
            }

            auto bbox = t.kalman.rect();
            /*if (t.kalman.state() == TrackState::Confirmed &&
                img_box.contains(bbox.tl()) && img_box.contains(bbox.br())) {
                t.info->track_id = t.kalman.id();
                ret.push_back(t.info);
            }*/

            if (t.kalman.state() == TrackState::Confirmed) {
                t.info->track_id = t.kalman.id();
                ret.push_back(t.info);
            }
        }
        return ret;
    }

private:
    std::vector<T> &data;
    const cv::Rect2f img_box;//整个图像的大小
};


}

#endif //DYNAMIC_VINS_TRACKER_H
