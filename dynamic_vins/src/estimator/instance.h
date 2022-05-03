/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_INSTANCE_H
#define DYNAMIC_VINS_INSTANCE_H

#include <unordered_map>
#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include "dynamic.h"
#include "landmark.h"
#include "vio_parameters.h"
#include "utils/parameters.h"

namespace dynamic_vins{\


class Estimator;



class Instance{
public:
    using Ptr=std::shared_ptr<Instance>;
    Instance()=default;
    Instance(const unsigned int frame_id,const unsigned int &id,Estimator* estimator): id(id),e(estimator){
    }

    int SlideWindowOld();
    int SlideWindowNew();

    void InitialPose();
    void SetCurrentPoint3d();

    void SetOptimizeParameters();
    void GetOptimizationParameters();
    void SetWindowPose();
    void OutlierRejection();

    double ReprojectTwoFrameError(FeaturePoint &feat_j, FeaturePoint &feat_i, double depth, bool isStereo);
    void GetBoxVertex(EigenContainer<Eigen::Vector3d> &vertex);

    void SetDynamicOrStatic();

    double AverageDepth() const{
        if(triangle_num>0){
            return depth_sum/double(triangle_num);
        }
        else{
            return 0;
        }
    }

    vector<Eigen::Vector3d> point3d_curr;
    std::list<LandmarkPoint> landmarks;

    unsigned int id{0};

    bool is_initial{false};//是否已经初始化位姿了
    bool is_tracking{true};//是否在滑动窗口中
    bool opt_vel{false};

    State state[(kWinSize + 1)]{}; //物体的位姿
    Vel3d vel,last_vel;//物体的速度

    Eigen::Vector3d box;
    cv::Scalar color;

    //优化过程中的变量
    double para_state[kWinSize + 1][kSizePose]{};
    double para_speed[1][kSpeedSize]{};
    double para_box[1][kBoxSize]{};
    double para_inv_depth[kInstFeatSize][kSizeFeature]{};//逆深度参数数组

    Estimator* e{nullptr};

    int triangle_num{0};//已经三角化的路标点的数量
    double depth_sum{0};//所有路标点的平均深度

    bool is_static{false};//物体是运动的还是静态的
    int static_cnt{0};//辅助判断物体是否运动,
};

}

#endif //DYNAMIC_VINS_DYNAMICFEATURE_H
