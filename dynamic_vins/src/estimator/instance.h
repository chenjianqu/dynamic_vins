/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
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

#include "vio_util.h"
#include "landmark.h"
#include "vio_parameters.h"
#include "utils/parameters.h"
#include "utils/box2d.h"
#include "utils/box3d.h"

namespace dynamic_vins{\


class Estimator;

struct State{
    State():R(Mat3d::Identity()),P(Vec3d::Zero()){
    }

    /**
     * 拷贝构造函数
     */
    State(const State& rhs):R(rhs.R),P(rhs.P),time(rhs.time){}

    /**
     * 拷贝赋值运算符
     * @param rhs
     * @return
     */
    State& operator=(const State& rhs){
        R = rhs.R;
        P = rhs.P;
        time = rhs.time;
        return *this;
    }

    void swap(State &rstate){
        State temp=rstate;
        rstate.R=std::move(R);
        rstate.P=std::move(P);
        rstate.time=time;
        R=std::move(temp.R);
        P=std::move(temp.P);
        time=temp.time;
    }
    Mat3d R;
    Vec3d P;
    double time{0};
};

class Instance{
public:
    using Ptr=std::shared_ptr<Instance>;
    Instance()=default;

    Instance(const unsigned int frame_id,const unsigned int &id,Estimator* estimator)
    : id(id),e(estimator){
    }

    int SlideWindowOld();
    int SlideWindowNew();

    void InitialPose();
    void SetCurrentPoint3d();

    void SetOptimizeParameters();
    void GetOptimizationParameters();
    void SetWindowPose();
    void OutlierRejection();

    double ReprojectTwoFrameError(FeaturePoint &feat_j, FeaturePoint &feat_i, double depth, bool isStereo) const;
    void GetBoxVertex(EigenContainer<Eigen::Vector3d> &vertex);

    void DetermineStatic();

    [[nodiscard]] double AverageDepth() const;

    void ClearState(){
        is_init_velocity=false;
        is_initial = false;
        is_tracking = false;
        age=0;
    }

    vector<Eigen::Vector3d> point3d_curr;
    std::list<LandmarkPoint> landmarks;

    unsigned int id{0};

    cv::Scalar color;
    Box3D::Ptr box3d;
    Box2D::Ptr box2d;

    bool is_initial{false};//是否已经初始化位姿了
    bool is_tracking{true};//是否在滑动窗口中
    bool is_curr_visible{false};//当前帧是否可见

    State state[(kWinSize + 1)]{}; //物体的位姿
    Vel3d vel,last_vel;//物体的速度
    bool is_init_velocity{false};

    //优化过程中的变量
    double para_state[kWinSize + 1][kSizePose]{};
    double para_speed[1][kSpeedSize]{};
    double para_box[1][kBoxSize]{};
    double para_inv_depth[kInstFeatSize][kSizeFeature]{};//逆深度参数数组

    Estimator* e{nullptr};

    int triangle_num{0};//已经三角化的路标点的数量

    bool is_static{false};//物体是运动的还是静态的

    int age{0};//初始化后走过了多少帧
    int lost_number{0};//已经连续多少帧未检测到特征了

    Box3D::Ptr boxes3d[(kWinSize + 1)]{};//实例关联的3d box

    std::list<State> history_pose;
};

}

#endif //DYNAMIC_VINS_DYNAMICFEATURE_H
