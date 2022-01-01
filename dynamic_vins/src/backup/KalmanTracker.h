//
// Created by chen on 2021/11/30.
//

#ifndef DYNAMIC_VINS_KALMANTRACKER_H
#define DYNAMIC_VINS_KALMANTRACKER_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "track.h"

/**
 * 跟踪的状态
 */
enum class TrackState {
    Tentative,//初始状态
    Confirmed,//当跟踪次数>N_INIT时，转入该状态
    Deleted //删除状态
};


// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker {
public:
    explicit KalmanTracker(cv::Rect2f initRect) : KalmanTracker() {
        init(initRect);
    }


    KalmanTracker() {
        kf = cv::KalmanFilter(STATE_NUM, MEASURE_NUM, 0);

        measurement = cv::Mat::zeros(MEASURE_NUM, 1, CV_32F);
        //设置状态转移矩阵
        kf.transitionMatrix = (cv::Mat_<float>(STATE_NUM, STATE_NUM)
                <<
                1, 0, 0, 0, 1, 0, 0,
                0, 1, 0, 0, 0, 1, 0,
                0, 0, 1, 0, 0, 0, 1,
                0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 1);

        //测量矩阵H
        cv::setIdentity(kf.measurementMatrix);
        //系统误差矩阵Q
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
        //测量误差矩阵R
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
        //最小均方误差的
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
    }

    void init(cv::Rect2f &initRect) {
        // initialize state vector with bounding box in [cx,cy,s,r] style
        kf.statePost.at<float>(0, 0) = initRect.x + initRect.width / 2;
        kf.statePost.at<float>(1, 0) = initRect.y + initRect.height / 2;
        kf.statePost.at<float>(2, 0) = initRect.area();
        kf.statePost.at<float>(3, 0) = initRect.width / initRect.height;
    }

    // Predict the estimated bounding box.
    void predict() {
        ++time_since_update;
        kf.predict();
    }


    void miss() {
        if (_state == TrackState::Tentative) {
            _state = TrackState::Deleted;
        }
        else if (time_since_update > Config::TRACKING_MAX_AGE) {
            _state = TrackState::Deleted;
        }
    }

    // Return the current state vector
    [[nodiscard]] cv::Rect2f rect() const {
        return get_rect_xysr(kf.statePost);
    }

    [[nodiscard]] TrackState state() const {
        return _state;
    }


    [[nodiscard]] int id() const {
        return _id;
    }


    // Update the state vector with observed bounding box.
    void update(cv::Rect2f &stateMat) {
        time_since_update = 0;
        ++hits;

        if (_state == TrackState::Tentative && hits > Config::TRACKING_N_INIT) {
            _state = TrackState::Confirmed;
            _id = count++;
        }

        //设置测量值: 中心坐标x 中心坐标y box面积 box长宽比
        measurement.at<float>(0, 0) = stateMat.x + stateMat.width / 2;
        measurement.at<float>(1, 0) = stateMat.y + stateMat.height / 2;
        measurement.at<float>(2, 0) = stateMat.area();
        measurement.at<float>(3, 0) = stateMat.width / stateMat.height;

        //滤波器修正
        kf.correct(measurement);
    }


    static constexpr int STATE_NUM = 7;//状态维度
    static constexpr int MEASURE_NUM = 4;//测量维度
private:
    inline static int count = 0;

    TrackState _state{TrackState::Tentative};

    int _id{-2};
    int time_since_update {0};//距离上次检测到的次数
    int hits{0};

    /**
     * 卡尔曼滤波
     * 状态维度：7维，表示u,v,s,r,u_d,v_d,s_d
     * 其中：
     * u,v是box中心坐标
     * s是box面积
     * r是box宽高比
     * u_d,v_d,s_d分别表示导数
     */
    cv::KalmanFilter kf;
    cv::Mat measurement;
};


#endif //DYNAMIC_VINS_KALMANTRACKER_H
