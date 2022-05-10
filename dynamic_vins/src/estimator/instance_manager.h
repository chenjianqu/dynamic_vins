/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
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

#include "utils/parameters.h"
#include "estimator/vio_util.h"
#include "instance.h"
#include "landmark.h"
#include "feature_queue.h"


namespace dynamic_vins{\


class Estimator;

class InstanceManager{
public:
    using Ptr=std::shared_ptr<InstanceManager>;

    InstanceManager()= default;

    void PushBack(unsigned int  frame_id, std::map<unsigned int,FeatureInstance> &input_insts);

    void Triangulate(int frame_cnt);

    void PropagatePose();

    void GetOptimizationParameters();

    void SlideWindow();

    void AddInstanceParameterBlock(ceres::Problem &problem);

    void AddResidualBlock(ceres::Problem &problem, ceres::LossFunction *loss_function);

    void SetVelMap();

    string PrintInstanceInfo(bool output_lm,bool output_stereo=false);
    string PrintInstancePoseInfo(bool output_lm);

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

    void SetDynamicOrStatic(){
        InstExec([](int key,Instance& inst){
            inst.DetermineStatic();
        },true);
    }

    std::unordered_map<unsigned int,Vel3d> vel_map(){
        std::unique_lock<std::mutex> lk(vel_mutex_);
        return vel_map_;
    }

    void set_estimator(Estimator* estimator);

    int tracking_number() const {return tracking_number_;}

    std::unordered_map<unsigned int,Instance> instances;
private:
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

    std::mutex vel_mutex_;
    std::unordered_map<unsigned int,Vel3d> vel_map_;

    Estimator* e{nullptr};
    int opt_inst_num_{0};//优化位姿的数量
    int tracking_number_{0};//正在跟踪的物体数量

    int frame{0};
};

}

#endif //DYNAMIC_VINS_INSTANCE_MANAGER_H
