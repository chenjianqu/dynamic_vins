/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_INSTANCE_TRACKER_H
#define DYNAMIC_VINS_INSTANCE_TRACKER_H


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
#include "estimator/landmark.h"
#include "mot/deep_sort.h"
#include "feature_utils.h"
#include "estimator/feature_queue.h"
#include "utils/box3d.h"

namespace dynamic_vins{\


struct InstFeat{
    using Ptr=std::shared_ptr<InstFeat>;
    InstFeat();
    InstFeat(unsigned int id_, int class_id_);

    unsigned int id{0};
    int class_id{0};
    cv::Scalar color;

    vector<cv::Point2f> last_points;
    std::map<unsigned int, cv::Point2f> prev_id_pts;

    vector<cv::Point2f> curr_points;
    vector<cv::Point2f> curr_un_points;
    vector<int> track_cnt;//每个特征点的跟踪次数

    std::map<unsigned int, cv::Point2f> curr_id_pts;

    vector<unsigned int> ids;

    vector<cv::Point2f> right_points;
    vector<cv::Point2f> right_un_points;
    std::map<unsigned int, cv::Point2f> right_prev_id_pts;
    std::map<unsigned int, cv::Point2f> right_curr_id_pts;

    vector<unsigned int> right_ids;

    vector<cv::Point2f> pts_velocity, right_pts_velocity;

    std::list<std::pair<cv::Point2f,cv::Point2f>> visual_points_pair;
    std::list<std::pair<cv::Point2f,cv::Point2f>> visual_right_points_pair;
    std::list<cv::Point2f> visual_new_points;

    cv::Point2f feats_center_pt;//当前跟踪的特征点的中心坐标
    int row{},col{};

    cv::Point2f box_vel;
    double last_time{-1.},delta_time{};

    int lost_num{0};//无法被跟踪的帧数,超过一定数量该实例将被删除

    cv::Mat mask_img;//物体在当前帧的mask
    cv::cuda::GpuMat mask_img_gpu;//物体在当前帧的mask
    torch::Tensor mask_tensor;
    float mask_area{0.};//当前帧中属于该物体的像素数量

    unsigned int last_frame_cnt{0};

    Box2D::Ptr box2d;
    Box3D::Ptr box3d;
};






class InstsFeatManager {
public:
    using Ptr=std::shared_ptr<InstsFeatManager>;
    explicit InstsFeatManager(const string& config_path);

    void InstsTrack(SemanticImage img);
    void InstsTrackByMatching(SemanticImage img);

    std::map<unsigned int,FeatureInstance> GetOutputFeature();
    void AddViodeInstances(SemanticImage &img);
    cv::Mat AddInstancesByIoU(SemanticImage &img);
    void AddInstancesByIouWithGPU(const SemanticImage &img);
    void AddInstancesByTracking(SemanticImage &img);
    void DrawInsts(cv::Mat& img);

    void set_vel_map(const std::unordered_map<unsigned int,Vel3d>& vel_map){vel_map_ = vel_map;}

private:
    void ManageInstances();

    void ClearState();

    void BoxAssociate2Dto3D(std::vector<Box3D::Ptr> &boxes);


    vector<uchar> RejectWithF(InstFeat &inst, int col, int row) const;

    std::tuple<int,float,float> GetMatchInst(InstInfo &instInfo, torch::Tensor &inst_mask_tensor);

    void ExecInst(std::function<void(unsigned int, InstFeat&)> func){
        for(auto & [ key,inst] : instances_){
            if(inst.lost_num>0)
                continue;
            func(key,inst);
        }
    }

    std::unordered_map<unsigned int,InstFeat> instances_;
    std::unordered_map<unsigned int,Vel3d> vel_map_;

    PinHoleCamera::Ptr camera_,right_camera_;

    unsigned int global_frame_id{0};
    cv::Mat mask_background;
    cv::cuda::GpuMat mask_background_gpu;

    std::vector<cv::Point2f> visual_new_points_;
    SemanticImage prev_img;
    bool is_exist_inst_{false};

    unsigned long global_id_count{0};//全局特征序号，注意与静态物体上的特征id不共用
    unsigned int global_instance_id{0};

    double curr_time{},last_time{};

    cv::Ptr<cv::cuda::CornersDetector> detector;

    DeepSORT::Ptr mot_tracker;
    //FlowEstimator::Ptr flow_estimator_;

    cv::Ptr<cv::FeatureDetector> orb_detector_;
    cv::Ptr<cv::DescriptorExtractor> orb_descriptor_;
    cv::Ptr<cv::DescriptorMatcher> orb_matcher_;
};

}

#endif //DYNAMIC_VINS_INSTANCE_TRACKER_H
