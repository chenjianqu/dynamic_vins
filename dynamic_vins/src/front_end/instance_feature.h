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

#include "semantic_image.h"
#include "utils/parameters.h"
#include "feature_utils.h"
#include "utils/box3d.h"
#include "utils/box2d.h"

namespace dynamic_vins{\

struct InstFeat{
    using Ptr=std::shared_ptr<InstFeat>;
    InstFeat():color(color_rd(randomEngine),color_rd(randomEngine),color_rd(randomEngine))
    {}

    InstFeat(unsigned int id_): id(id_),color(color_rd(randomEngine),color_rd(randomEngine),color_rd(randomEngine))
    {}

    void SortPoints();

    ///检测特征点
    void DetectNewFeature(SemanticImage &img,bool use_gpu,const cv::Mat &mask = cv::Mat());

    void RemoveOutliers(std::set<unsigned int> &removePtsIds);

    void PtsVelocity(double dt);//计算像素的速度
    void RightPtsVelocity(double dt);

    void UndistortedPts(PinHoleCamera::Ptr &cam);
    void RightUndistortedPts(PinHoleCamera::Ptr &cam);

    ///跟踪图像
    void TrackLeft(SemanticImage &img,SemanticImage &prev_img,bool dense_flow=cfg::use_dense_flow);
    void TrackLeftGPU(SemanticImage &img,SemanticImage &prev_img,
                      cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_forward,
                      cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_backward);

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
    }

    unsigned int id{0};
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

    cv::Point2f feats_center_pt;//当前跟踪的特征点的中心坐标

    int lost_num{0};//无法被跟踪的帧数,超过一定数量该实例将被删除

    unsigned int last_frame_cnt{0};

    Box2D::Ptr box2d;
    Box3D::Ptr box3d;

    inline static std::default_random_engine randomEngine;
    inline static std::uniform_int_distribution<unsigned int> color_rd{0,255};

    inline static unsigned long global_id_count{0};//全局特征序号

};







}


#endif //DYNAMIC_VINS_INSTANCEFEATURE_H
