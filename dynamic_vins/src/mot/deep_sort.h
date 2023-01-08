/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_DEEPSORT_H
#define DYNAMIC_VINS_DEEPSORT_H

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "extractor.h"
#include "tracker_manager.h"
#include "feature_bundle.h"
#include "feature_metric.h"
#include "basic/def.h"
#include "basic/box2d.h"

namespace dynamic_vins{\

torch::Tensor CalIouDist(const std::vector<cv::Rect2f> &dets, const std::vector<cv::Rect2f> &trks);

struct TrackData {
    KalmanTracker kalman;
    FeatureBundle feats;
    Box2D::Ptr info;
};

class DeepSORT {
public:
    using Ptr=std::unique_ptr<DeepSORT>;
    explicit DeepSORT(const std::string& config_path,const std::array<int64_t, 2> &dim);

    std::vector<Box2D::Ptr> update(const std::vector<Box2D::Ptr> &detections, cv::Mat ori_img);

private:
    std::vector<TrackData> data;

    std::unique_ptr<Extractor> extractor;
    std::unique_ptr<TrackerManager<TrackData>> manager;
    std::unique_ptr<FeatureMetric<TrackData>> feat_metric;
};


}

#endif //DYNAMIC_VINS_DEEPSORT_H
