/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_INSTANCE_MANAGER_H
#define DYNAMIC_VINS_INSTANCE_MANAGER_H

#include <unordered_map>
#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include "factor/pose_local_parameterization.h"
#include "factor/projection_instance_factor.h"
#include "factor/projection_speed_factor.h"
#include "factor/projection_box_factor.h"
#include "factor/projection_factor_simple.h"
#include "utils/parameters.h"
#include "estimator/dynamic.h"
#include "instance.h"
#include "landmark.h"


namespace dynamic_vins{\


class Estimator;

class InstanceManager{
public:
    using Ptr=std::shared_ptr<InstanceManager>;
    InstanceManager(){
        ProjectionInstanceFactor::sqrt_info = kFocalLength / 1.5 * Eigen::Matrix2d::Identity();//初始化因子的信息矩阵
    }

    void PushBack(unsigned int  frame_id, InstancesFeatureMap &input_insts);
    void Triangulate(int frame_cnt);
    void PredictCurrentPose();
    void GetOptimizationParameters();
    void SlideWindow();
    void AddInstanceParameterBlock(ceres::Problem &problem);
    void AddResidualBlock(ceres::Problem &problem, ceres::LossFunction *loss_function);

    /**
    * 获得优化完成的参数，并重新设置窗口内物体的位姿
    */
    void SetOptimizationParameters(){
        InstExec([](int key,Instance& inst){
            inst.SetOptimizeParameters();
        });
    }

    /**
 * 根据速度设置物体的位姿
 */
    void SetWindowPose(){
        InstExec([](int key,Instance& inst){
            inst.SetWindowPose();
        });
    }

    /**
 * 进行物体的位姿初始化
 */
    void InitialInstance(){
        InstExec([](int key,Instance& inst){
            inst.InitialPose();
        });
    }

    void SetInstanceCurrentPoint3d(){
        InstExec([](int key,Instance& inst){
            inst.SetCurrentPoint3d();
        });
    }

    void OutliersRejection(){
        InstExec([](int key,Instance& inst){
            inst.OutlierRejection();
        });
    }

    std::unordered_map<unsigned int,Vel3d> vel_map(){
        std::unique_lock<std::mutex> lk(vel_mutex_);
        return vel_map_;
    }

    void set_estimator(Estimator* estimator);
    int tracking_number() const {return tracking_number_;}

    std::unordered_map<unsigned int,Instance> instances;
private:
    void InstExec(std::function<void(unsigned int,Instance&)> function){
        if(tracking_number_ < 1)
            return;
        for(auto &[key,inst] : instances){
            if(!inst.is_initial || !inst.is_tracking) continue;
            function(key,inst);
        }
    }

    void SetVelMap(){
        std::unique_lock<std::mutex> lk(vel_mutex_);
        vel_map_.clear();
        InstExec([this](int key,Instance& inst){
            if(inst.vel.v.norm() > 0)vel_map_.insert({inst.id, inst.vel});;
        });
    }

    std::mutex vel_mutex_;
    std::unordered_map<unsigned int,Vel3d> vel_map_;

    Estimator* e_{nullptr};
    int opt_inst_num_{0};//优化位姿的数量
    int tracking_number_{0};//正在跟踪的物体数量
};

}

#endif //DYNAMIC_VINS_INSTANCE_MANAGER_H
