/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

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


#pragma once

#include <cstdio>
#include <execinfo.h>
#include <csignal>

#include "utils/def.h"
#include "utils/parameters.h"
#include "instance_tracker.h"
#include "det2d/detector2d.h"
#include "feature_utils.h"
#include "estimator/vio_util.h"

namespace dynamic_vins{\

class FeatureTracker
{
public:
    using Ptr=std::unique_ptr<FeatureTracker>;
    explicit FeatureTracker(const string& config_path);

    FeatureBackground TrackImage(SemanticImage &img);
    FeatureBackground TrackImageNaive(SemanticImage &img);
    FeatureBackground TrackSemanticImage(SemanticImage &img);

    cv::Mat img_track(){return img_track_;}

    InstsFeatManager::Ptr insts_tracker;
    SemanticImage prev_img, cur_img;

private:
    FeatureBackground SetOutputFeats();
    void SetPrediction(std::map<int, Eigen::Vector3d> &predictPts);
    void RemoveOutliers(std::set<int> &removePtsIds);

    void ShowUndistortion(const string &name);
    void RejectWithF();

    vector<cv::Point2f> PtsVelocity(vector<int> &id_vec, vector<cv::Point2f> &pts,
                                    std::map<int, cv::Point2f> &cur_id_pts,
                                    std::map<int, cv::Point2f> &prev_id_pts) const;
    void DrawTrack(const SemanticImage &img,
                   vector<int> &curLeftIds,
                   vector<cv::Point2f> &curLeftPts,
                   vector<cv::Point2f> &curRightPts,
                   std::map<int, cv::Point2f> &prevLeftPts);

    int row{}, col{};
    cv::Mat img_track_;
    cv::Mat mask;
    cv::cuda::GpuMat mask_gpu;
    cv::Mat fisheye_mask;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;
    vector<cv::Point2f> pts_velocity, right_pts_velocity;
    vector<int> ids, ids_right;
    vector<int> track_cnt;
    std::map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    std::map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    std::map<int, cv::Point2f> prev_left_map;
    double cur_time{};
    double prev_time{};
    int n_id;

    std::vector<cv::Point2f> visual_new_pts;

    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_optical_flow;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_optical_flow_back;
    cv::Ptr<cv::cuda::CornersDetector> detector;
};


}
