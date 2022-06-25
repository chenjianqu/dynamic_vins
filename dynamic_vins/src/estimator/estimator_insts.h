/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_ESTIMATOR_INSTS_H
#define DYNAMIC_VINS_ESTIMATOR_INSTS_H

#include <unordered_map>
#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include "utils/parameters.h"
#include "estimator/vio_util.h"
#include "instance.h"
#include "landmark.h"
#include "semantic_feature.h"


namespace dynamic_vins{\



class InstanceManager{
public:
    using Ptr=std::shared_ptr<InstanceManager>;

    InstanceManager()= default;

    void PushBack(unsigned int  frame_id, std::map<unsigned int,FeatureInstance> &input_insts);

    void Triangulate();

    void ManageTriangulatePoint();

    void PropagatePose();

    void GetOptimizationParameters();

    void SlideWindow(const MarginFlag &flag);

    void AddInstanceParameterBlock(ceres::Problem &problem);

    void AddResidualBlockForJointOpt(ceres::Problem &problem, ceres::LossFunction *loss_function);

    void SetOutputInstInfo();


    void InitialInstance(std::map<unsigned int,FeatureInstance> &input_insts);

    void InitialInstanceVelocity();

    void AddResidualBlockForInstOpt(ceres::Problem &problem, ceres::LossFunction *loss_function);

    void Optimization();


    /**
    * 获得优化完成的参数，并重新设置窗口内物体的位姿
    */
    void SetOptimizationParameters(){
        InstExec([](int key,Instance& inst){
            inst.SetOptimizeParameters();
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

    void DeleteBadLandmarks();

    void SetDynamicOrStatic();

    std::unordered_map<unsigned int,InstEstimatedInfo> GetOutputInstInfo(){
        std::unique_lock<std::mutex> lk(vel_mutex_);
        return insts_output;
    }

    void InstExec(std::function<void(unsigned int,Instance&)> function,bool exec_all=false){
        if(tracking_number_ < 1)
            return;
        if(exec_all){
            for(auto &[key,inst] : instances){
                function(key,inst);
            }
        }
        else{
            for(auto &[key,inst] : instances){
                if(!inst.is_initial || !inst.is_tracking) continue;
                function(key,inst);
            }
        }
    }

    int tracking_number() const {return tracking_number_;}

    std::unordered_map<unsigned int,Instance> instances;
private:


    std::mutex vel_mutex_;
    std::unordered_map<unsigned int,InstEstimatedInfo> insts_output;

    int opt_inst_num_{0};//优化位姿的数量
    int tracking_number_{0};//正在跟踪的物体数量

    int frame{0};

};

}

#endif //DYNAMIC_VINS_ESTIMATOR_INSTS_H
