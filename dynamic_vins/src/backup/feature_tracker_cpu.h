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
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <thread>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../parameters.h"
#include "../estimator/dynamic.h"
#include "../featureTracker/instance_tracker.h"
#include "../InstanceSegmentation/infer.h"
#include "../utils.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;


class FeatureTracker
{
public:
    using Ptr=std::shared_ptr<FeatureTracker>;
    FeatureTracker();
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(SegImage &img);
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImageNaive(SegImage &img);
    FeatureMap trackSemanticImage(SegImage &img);

    void setMask();
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    static vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &id_vec, vector<cv::Point2f> &pts,
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts) const;
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const SegImage &img,
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPts);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    void removeOutliers(set<int> &removePtsIds);
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> setOutputFeats();


    int row{}, col{};
    cv::Mat imTrack;
    cv::Mat mask,semantic_mask;
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
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map;
    map<int, cv::Point2f> prevLeftPtsMap;
    vector<camodocal::CameraPtr> m_camera;
    double cur_time{};
    double prev_time{};
    bool stereo_cam;
    int n_id;

    InstsFeatManager::Ptr insts_tracker;


private:
    static void superpositionMask(cv::Mat &mask1, const cv::Mat &mask2);

};
