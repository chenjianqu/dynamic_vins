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
#include "instance_feature.h"

namespace dynamic_vins{\


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

    SemanticImage prev_img;
    bool is_exist_inst_{false};

    unsigned int global_instance_id{0};

    double curr_time{},last_time{};

    cv::Ptr<cv::cuda::CornersDetector> detector;

    DeepSORT::Ptr mot_tracker;
    //FlowEstimator::Ptr flow_estimator_;
};

}

#endif //DYNAMIC_VINS_INSTANCE_TRACKER_H
