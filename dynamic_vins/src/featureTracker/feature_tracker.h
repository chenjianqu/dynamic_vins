/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/
/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <thread>
#include <memory>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <camodocal/camera_models/CameraFactory.h>
#include <camodocal/camera_models/CataCamera.h>
#include <camodocal/camera_models/PinholeCamera.h>

#include "parameters.h"
#include "utils.h"
#include "estimator/dynamic.h"
#include "instance_tracker.h"
#include "InstanceSegment/instance_segmentor.h"
#include "feature_utils.h"

namespace dynamic_vins{\

class FeatureTracker
{
public:
    using Ptr=std::unique_ptr<FeatureTracker>;
    FeatureTracker();
    std::map<int, vector<pair<int, Vec7d>>> TrackImage(SegImage &img);
    std::map<int, vector<pair<int, Vec7d>>> TrackImageNaive(SegImage &img);
    FeatureMap TrackSemanticImage(SegImage &img);

    void ReadIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    static vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &id_vec, vector<cv::Point2f> &pts,
                                    std::map<int, cv::Point2f> &cur_id_pts,
                                    std::map<int, cv::Point2f> &prev_id_pts) const;
    void drawTrack(const SegImage &img,
                   vector<int> &curLeftIds,
                   vector<cv::Point2f> &curLeftPts,
                   vector<cv::Point2f> &curRightPts,
                   std::map<int, cv::Point2f> &prevLeftPts);
    void setPrediction(std::map<int, Eigen::Vector3d> &predictPts);
    void removeOutliers(std::set<int> &removePtsIds);
    std::map<int, vector<pair<int, Vec7d>>> SetOutputFeats();

    cv::Mat img_track(){return img_track_;}

    InstsFeatManager::Ptr insts_tracker;
private:
    void SortPoints(std::vector<cv::Point2f> &cur_pts, std::vector<int> &track_cnt, std::vector<int> &ids);

    int row{}, col{};
    cv::Mat img_track_;
    cv::Mat mask,semantic_mask;
    cv::cuda::GpuMat mask_gpu,semantic_mask_gpu;
    cv::Mat fisheye_mask;
    SegImage prev_img, cur_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> track_cnt;
    std::map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    std::map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    std::map<int, cv::Point2f> prev_left_map;
    vector<camodocal::CameraPtr> m_camera;
    double cur_time{};
    double prev_time{};
    int n_id;

    std::vector<cv::Point2f> visual_new_pts;

    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_optical_flow;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_optical_flow_back;
    cv::Ptr<cv::cuda::CornersDetector> detector;
};


}
