//
// Created by chen on 2021/11/30.
//

#ifndef DYNAMIC_VINS_DEEPSORT_H
#define DYNAMIC_VINS_DEEPSORT_H

#include <memory>
#include <array>
#include <utility>

#include <torch/torch.h>

#include "Extractor.h"
#include "FeatureMetric.h"
#include "KalmanTracker.h"
#include "TrackerManager.h"
#include "Hungarian.h"
#include "utility/parameters.h"
#include "utility/utils.h"



class DeepSORT {
public:
    using Ptr=std::unique_ptr<DeepSORT>;
    explicit DeepSORT(const std::array<int64_t, 2> &dim);

    std::vector<InstInfo> update( std::vector<InstInfo> &detections, cv::Mat &ori_img);

private:
    struct InstanceTrackData {
        KalmanTracker kalman;
        FeatureBundle feats;
        InstInfo info;
    };

    std::vector<InstanceTrackData> data; //正在跟踪的实例

    Extractor::Ptr extractor;
    FeatureMetric<InstanceTrackData>::Ptr featureMetric;
    TrackerManager<InstanceTrackData>::Ptr trackerManager;
};


#endif //DYNAMIC_VINS_DEEPSORT_H
