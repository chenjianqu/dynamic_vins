//
// Created by chen on 2021/11/30.
//

#ifndef DYNAMIC_VINS_TRACKERMANAGER_H
#define DYNAMIC_VINS_TRACKERMANAGER_H

#include <memory>
#include <torch/torch.h>

#include "track.h"
#include "KalmanTracker.h"
#include "utility/utils.h"



using DistanceMetricFunc = std::function<torch::Tensor(const std::vector<int> &trk_ids, const std::vector<int> &det_ids)>;

const float INVALID_DIST = 1E3f;

void associate_detections_to_trackers_idx(const DistanceMetricFunc &metric,
                                          std::vector<int> &unmatched_trks,
                                          std::vector<int> &unmatched_dets,
                                          std::vector<std::tuple<int, int>> &matched);


template<typename TrackData>
class TrackerManager {
public:
    using Ptr=std::unique_ptr<TrackerManager>;
    explicit TrackerManager(std::vector<TrackData> &data, const std::array<int64_t, 2> &dim)
    : data(data), img_box(0, 0, dim[1], dim[0]) {}

    /**
     * 执行卡尔曼滤波的预测
     */
    void predict() {
        for (auto &t:data) {
            t.kalman.predict();
        }
    }

    /**
     * 去除跟踪失败的实例
     */
    void remove_nan() {
        data.erase(std::remove_if(data.begin(),data.end(),[](const TrackData &t) {
                            auto bbox = t.kalman.rect();
                            return std::isnan(bbox.x) || std::isnan(bbox.y) || std::isnan(bbox.width) || std::isnan(bbox.height);
                    }),
                   data.end());
    }

    /**
     * 去除跟踪失败的实例
     */
    void remove_deleted() {
        data.erase(
                remove_if(data.begin(), data.end(),[this](const TrackData &t) {
                    return t.kalman.state() == TrackState::Deleted;
                }),
                data.end());
    }


    /**
     * 更新跟踪实例
     * @param dets
     * @param confirmed_metric
     * @param unconfirmed_metric
     * @return 返回vector<x,y>,其中x是跟踪列表索引，y是匹配的当前检测的索引
     */
    std::vector<std::tuple<int, int>> update(std::vector<InstInfo> &dets,
                                             const DistanceMetricFunc &confirmed_metric,
                                             const DistanceMetricFunc &unconfirmed_metric){

        //获得跟踪良好的实例的索引
        std::vector<int> unmatched_trks;
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i].kalman.state() == TrackState::Confirmed) {
                unmatched_trks.emplace_back(i);
            }
        }

        //每个新检测分配一个索引
        std::vector<int> unmatched_dets(dets.size());
        std::iota(unmatched_dets.begin(), unmatched_dets.end(), 0);

        std::vector<std::tuple<int, int>> matched;
        associate_detections_to_trackers_idx(confirmed_metric, unmatched_trks, unmatched_dets, matched);


        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i].kalman.state() == TrackState::Tentative) {
                unmatched_trks.emplace_back(i);
            }
        }

        associate_detections_to_trackers_idx(unconfirmed_metric, unmatched_trks, unmatched_dets, matched);

        for (auto i : unmatched_trks) {
            data[i].kalman.miss();
        }

        ///更新成功跟踪的实例
        for (auto[x, y] : matched) {
            data[x].kalman.update(dets[y].rect);
            data[x].info = dets[y];
        }

        ///创建新的跟踪实例
        for (auto umd : unmatched_dets) {
            matched.emplace_back(data.size(), umd);
            auto t = TrackData{};
            t.kalman.init(dets[umd].rect);
            t.info = dets[umd];
            data.emplace_back(t);
        }

        return matched;
    }



    std::vector<Track> visible_tracks() {
        std::vector<Track> ret;
        for (auto &t : data) {
            auto bbox = t.kalman.rect();
            if (t.kalman.state() == TrackState::Confirmed && img_box.contains(bbox.tl()) && img_box.contains(bbox.br())) {
                Track res{t.kalman.id(), bbox};
                ret.push_back(res);
            }
        }
        return ret;
    }



    std::vector<InstInfo> visible_tracks_info() {
        std::vector<InstInfo> ret;
        for (auto &t : data) {
            auto bbox = t.kalman.rect();
            if (t.kalman.state() == TrackState::Confirmed && img_box.contains(bbox.tl()) && img_box.contains(bbox.br())) {
                t.info.track_id = t.kalman.id();
                ret.push_back(t.info);
            }
        }
        return ret;
    }


private:
    std::vector<TrackData> &data;
    const cv::Rect2f img_box;
};



#endif //DYNAMIC_VINS_TRACKERMANAGER_H
