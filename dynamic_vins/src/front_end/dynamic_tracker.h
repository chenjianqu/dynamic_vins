/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_DYNAMIC_TRACKER_H
#define DYNAMIC_VINS_DYNAMIC_TRACKER_H

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
#include <camodocal/camera_models/CameraFactory.h>

#include "basic/semantic_image.h"
#include "utils/parameters.h"
#include "basic/point_landmark.h"
#include "mot/deep_sort.h"
#include "feature_utils.h"
#include "basic/frontend_feature.h"
#include "basic/box3d.h"
#include "basic/inst_estimated_info.h"
#include "instance_feature.h"

namespace dynamic_vins{\


class InstsFeatManager {
public:
    using Ptr=std::shared_ptr<InstsFeatManager>;
    explicit InstsFeatManager(const string& config_path);

    void InstsTrack(SemanticImage img);

    void ProcessExtraPoints();

    std::map<unsigned int,FeatureInstance> Output();
    void AddViodeInstances(SemanticImage &img);
    cv::Mat AddInstancesByIoU(SemanticImage &img);
    void AddInstancesByIouWithGPU(const SemanticImage &img);
    void AddInstancesByTracking(SemanticImage &img);
    void DrawInsts(cv::Mat& img);

    void SetEstimatedInstancesInfo(const std::unordered_map<unsigned int,InstEstimatedInfo>& estimated_info_){
        estimated_info = estimated_info_;
    }

    void ExecInst(std::function<void(unsigned int, InstFeat&)> func){
        for(auto & [ key,inst] : instances){
            if(inst.lost_num>0)
                continue;
            func(key,inst);
        }
    }

private:
    void ManageInstances();

    void ClearState();

    void BoxAssociate2Dto3D(std::vector<Box3D::Ptr> &boxes);

    vector<uchar> RejectWithF(InstFeat &inst, int col, int row) const;

    std::tuple<int,float,float> GetMatchInst(Box2D &instInfo, torch::Tensor &inst_mask_tensor);

public:
    std::unordered_map<unsigned int,InstFeat> instances;
    std::unordered_map<unsigned int,InstEstimatedInfo> estimated_info;

private:
    unsigned int global_frame_id{0};
    cv::Mat mask_background;
    cv::cuda::GpuMat mask_background_gpu;

    SemanticImage prev_img,curr_img;
    bool is_exist_inst_{false};

    double curr_time{},last_time{};

    cv::Ptr<cv::cuda::CornersDetector> detector;

    DeepSORT::Ptr mot_tracker;
    //FlowEstimator::Ptr flow_estimator_;
};

}

#endif //DYNAMIC_VINS_DYNAMIC_TRACKER_H
