/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
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

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "utils/def.h"
#include "utils/parameters.h"
#include "dynamic_tracker.h"
#include "det2d/detector2d.h"
#include "feature_utils.h"
#include "estimator/vio_util.h"
#include "line_detector/line_detector.h"

namespace dynamic_vins{\

class FeatureTracker{
public:
    using Ptr=std::unique_ptr<FeatureTracker>;
    explicit FeatureTracker(const string& config_path);

    FeatureBackground TrackImage(SemanticImage &img);
    FeatureBackground TrackImageLine(SemanticImage &img);
    FeatureBackground TrackImageNaive(SemanticImage &img);
    FeatureBackground TrackSemanticImage(SemanticImage &img);

    cv::Mat& img_track(){return img_vis;}

    SemanticImage prev_img, cur_img;

private:
    FeatureBackground SetOutputFeats();
    void SetPrediction(std::map<int, Eigen::Vector3d> &predictPts);

    void ShowUndistortion(const string &name);
    void RejectWithF();

    void TrackLine(cv::Mat gray0, cv::Mat gray1,cv::Mat mask);

    void DrawTrack(const SemanticImage &img,
                   vector<unsigned int> &curLeftIds,
                   vector<cv::Point2f> &curLeftPts,
                   vector<cv::Point2f> &curRightPts,
                   std::map<unsigned int, cv::Point2f> &prevLeftPts);

    void DrawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                   vector<unsigned int> &curLeftIds,
                   vector<cv::Point2f> &curLeftPts,
                   vector<cv::Point2f> &curRightPts,
                   map<unsigned int, cv::Point2f> &prevLeftPtsMap);

    cv::Mat img_vis;

    std::map<unsigned int, cv::Point2f> last_id_pts_map;
    double cur_time{}, prev_time{};

    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_optical_flow;
    cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_optical_flow_back;
    cv::Ptr<cv::cuda::CornersDetector> corners_detector;
    LineDetector::Ptr line_detector;
    InstFeat bg;
};


}
