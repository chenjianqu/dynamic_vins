/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_KALMAN_H
#define DYNAMIC_VINS_KALMAN_H

#include <opencv2/video/tracking.hpp>

namespace dynamic_vins{\


enum class TrackState {
    Tentative,
    Confirmed,
    Deleted
};



class KalmanTracker {
public:
    KalmanTracker();

    explicit KalmanTracker(cv::Rect2f initRect) : KalmanTracker() { init(initRect); }

    void init(cv::Rect2f initRect);

    void predict();

    void update(cv::Rect2f stateMat);

    void miss();

    [[nodiscard]] cv::Rect2f rect() const;

    [[nodiscard]] TrackState state() const { return _state; }

    [[nodiscard]] int id() const { return _id; }

    [[nodiscard]] int get_time_since_update() const{ return time_since_update;}
private:

    inline static int global_instances_id=1;

    TrackState _state = TrackState::Tentative;

    int _id = -1;

    int time_since_update = 0;
    int hits = 0;

    cv::KalmanFilter kf;
    cv::Mat measurement;
};

}

#endif //DYNAMIC_VINS_KALMAN_H