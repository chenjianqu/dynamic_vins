/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_INSTANCEFEATURE_H
#define DYNAMIC_VINS_INSTANCEFEATURE_H


#include <queue>
#include <vector>
#include <unordered_map>
#include <map>
#include <memory>
#include <thread>
#include <random>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include <torch/torch.h>

#include "camodocal/camera_models/CameraFactory.h"

#include "semantic_image.h"
#include "utils/parameters.h"
#include "feature_utils.h"
#include "utils/box3d.h"
#include "utils/box2d.h"
#include "line_detector/line.h"
#include "line_detector/line_descriptor/include/line_descriptor_custom.hpp"
#include "line_detector/line_detector.h"

namespace dynamic_vins{\

struct InstFeat{
    using Ptr=std::shared_ptr<InstFeat>;
    InstFeat():color(color_rd(random_engine), color_rd(random_engine), color_rd(random_engine)),
    roi(std::make_shared<InstRoi>())
    {}

    InstFeat(unsigned int id_): id(id_),color(color_rd(random_engine), color_rd(random_engine), color_rd(random_engine)),
    roi(std::make_shared<InstRoi>())
    {}

    void SortPoints();

    ///检测特征点
    void DetectNewFeature(SemanticImage &img,bool use_gpu,int min_dist = 20,const cv::Mat &mask = cv::Mat());

    void RemoveOutliers(std::set<unsigned int> &removePtsIds);

    void PtsVelocity(double dt);//计算像素的速度
    void RightPtsVelocity(double dt);

    void UndistortedPts(camodocal::CameraPtr &cam);
    void RightUndistortedPts(camodocal::CameraPtr &cam);

    void UndistortedPoints(camodocal::CameraPtr &cam,vector<cv::Point2f>& point_cam,vector<cv::Point2f>& point_un);
    void UndistortedPointsWithAddOffset(camodocal::CameraPtr &cam,vector<cv::Point2f>& point_cam,vector<cv::Point2f>& point_un);


    ///跟踪图像
    void TrackLeft(cv::Mat &curr_img,cv::Mat &last_img,const cv::Mat &mask=cv::Mat());
    void TrackLeftGPU(SemanticImage &img,SemanticImage &prev_img,
                      cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_forward,
                      cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_backward,
                      const cv::Mat &mask=cv::Mat());

    void TrackRight(SemanticImage &img);
    void TrackRightGPU(SemanticImage &img,
                       cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_forward,
                       cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_backward);


    ///后处理
    void PostProcess(){
        last_points=curr_points;
        last_un_points = curr_un_points;

        prev_id_pts=curr_id_pts;
        right_prev_id_pts=right_curr_id_pts;

        visual_new_points.clear();

        prev_lines = curr_lines;

        roi->prev_roi_gpu = roi->roi_gpu;
        roi->prev_roi_gray = roi->roi_gray;
    }

    unsigned int id{1};
    cv::Scalar color;

    vector<unsigned int> ids, right_ids;
    vector<int> track_cnt;//每个特征点的跟踪次数

    vector<cv::Point2f> curr_points, curr_un_points;
    vector<cv::Point2f> last_points,last_un_points;
    vector<cv::Point2f> right_points, right_un_points;

    std::map<unsigned int, cv::Point2f> prev_id_pts, curr_id_pts;
    std::map<unsigned int, cv::Point2f> right_prev_id_pts, right_curr_id_pts;

    vector<cv::Point2f> pts_velocity, right_pts_velocity;

    std::list<std::pair<cv::Point2f,cv::Point2f>> visual_points_pair;
    std::list<std::pair<cv::Point2f,cv::Point2f>> visual_right_points_pair;
    std::list<cv::Point2f> visual_new_points;

    FrameLines::Ptr curr_lines,prev_lines;
    FrameLines::Ptr curr_lines_right;

    int lost_num{0};//无法被跟踪的帧数,超过一定数量该实例将被删除

    bool is_curr_visible{false};

    Box2D::Ptr box2d;
    Box3D::Ptr box3d;

    InstRoi::Ptr roi;//动态物体的ROI

    inline static std::default_random_engine random_engine;
    inline static std::uniform_int_distribution<unsigned int> color_rd{0,255};

    inline static unsigned long global_id_count{1};//全局特征序号

    vector<cv::Point2f> extra_points;
    vector<cv::Point2f> extra_un_points;
    vector<unsigned int> extra_ids;

};







}


#endif //DYNAMIC_VINS_INSTANCEFEATURE_H
