//
// Created by chen on 2021/10/8.
//

#ifndef DYNAMIC_VINS_INSTANCE_MANAGER_H
#define DYNAMIC_VINS_INSTANCE_MANAGER_H


#include <unordered_map>
#include <memory>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include "../factor/pose_local_parameterization.h"
#include "../factor/projectionInstanceFactor.h"
#include "../factor/projectionSpeedFactor.h"
#include "../factor/projectionBoxFactor.h"
#include "../factor/projectionFactorSimple.h"
#include "../parameters.h"
#include "../estimator/dynamic.h"

#include "Instance.h"

class Estimator;



class InstanceManager{
public:
    using Ptr=std::shared_ptr<InstanceManager>;
    InstanceManager(){
        ProjectionInstanceFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Eigen::Matrix2d::Identity();//初始化因子的信息矩阵
    }


    void setEstimator(Estimator* estimator_);

    void push_back(unsigned int  frame_id, InstancesFeatureMap &input_insts);
    void triangulate(int frameCnt);

    void predictCurrentPose();

    void getOptimizationParameters();

    void slideWindow();

    void addInstanceParameterBlock(ceres::Problem &problem);
    void addResidualBlock(ceres::Problem &problem, ceres::LossFunction *loss_function);


    /**
* 获得优化完成的参数，并重新设置窗口内物体的位姿
*/
    void setOptimizationParameters(){
        if(tracking_number<1) return;
        for(auto &[key,inst] : instances){
            if(!inst.isInitial || !inst.isTracking) continue;
            inst.setOptimizationParameters();
        }
    }

    /**
 * 根据速度设置物体的位姿
 */
    void setWindowPose(){
        if(tracking_number<1) return;
        for(auto &[key,inst] : instances){
            if(!inst.isInitial || !inst.isTracking) continue;
            inst.setWindowPose();
        }
    }

    /**
 * 进行物体的位姿初始化
 */
    void initialInstance(){
        if(tracking_number<1) return;
        for(auto &[key,inst] : instances){
            if(!inst.isTracking) continue;
            inst.initialPose();
        }
    }

    void setInstanceCurrentPoint3d(){
        if(tracking_number<1) return;
        for(auto &[key,inst] : instances){
            if(!inst.isInitial || !inst.isTracking) continue;
            inst.setCurrentPoint3d();
        }
    }

    void outliersRejection(){
        if(tracking_number<1) return;
        for(auto &[key,inst] : instances){
            if(!inst.isInitial || !inst.isTracking) continue;
            inst.outlierRejection();
        }
    }


    std::unordered_map<unsigned int,Vel3d> getInstancesVelocity(){
        std::unique_lock<std::mutex> lk(velMutex);
        return vel_map;
    }


    Estimator* e{nullptr};
    std::unordered_map<unsigned int,Instance> instances;
    int opt_inst_num{0};//优化位姿的数量
    int tracking_number{0};//正在跟踪的物体数量

private:




    std::mutex velMutex;
    std::unordered_map<unsigned int,Vel3d> vel_map;
    void setVelMap(){
        std::unique_lock<std::mutex> lk(velMutex);
        vel_map.clear();
        for(auto &[key,inst] : instances){
            if(inst.isTracking && inst.isInitial && inst.vel.v.norm() >0){
                vel_map.insert({inst.id,inst.vel});
            }
        }
    }


};



#endif //DYNAMIC_VINS_INSTANCE_MANAGER_H
