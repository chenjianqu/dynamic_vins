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
#include "vio_parameters.h"
#include "utils/parameters.h"
#include "body.h"
#include "basic/point_landmark.h"
#include "basic/box2d.h"
#include "basic/box3d.h"
#include "basic/velocity.h"
#include "basic/state.h"


namespace dynamic_vins{\


class Instance{
public:
    using Ptr=std::shared_ptr<Instance>;
    Instance()=default;

    Instance(const unsigned int frame_id,const unsigned int &id)
    : id(id){
    }

    int SlideWindowOld();
    int SlideWindowNew();

    void SetCurrentPoint3d();

    void SetOptimizeParameters();
    void GetOptimizationParameters();
    void SetWindowPose();
    void OutlierRejection();

    int OutlierRejectionByBox3d();

    void DeleteOutdatedLandmarks(int critical_frame);

    int DeleteBadLandmarks();

    [[nodiscard]] Vec3d WorldToObject(const Vec3d& pt,int frame_idx) const{
        return state[frame_idx].R.transpose() * ( pt - state[frame_idx].P);
    }

    [[nodiscard]] Vec3d ObjectToWorld(const Vec3d& pt,int frame_idx) const{
        return  state[frame_idx].R * pt + state[frame_idx].P;
    }

    [[nodiscard]] Vec3d CamToObject(const Vec3d& pt,int frame_idx,int cam_idx=0) const{
        return WorldToObject(body.CamToWorld(pt,frame_idx,cam_idx),frame_idx);
    }

    [[nodiscard]] Vec3d  ObjectToCam(const Vec3d& pt,int frame_idx,int cam_idx=0) const{
        return body.WorldToCam(ObjectToWorld(pt,frame_idx),frame_idx,cam_idx);
    }

    [[nodiscard]] double AverageDepth() const;

    /**
     * 所有非bad的路标点的数量
     * @return
     */
    [[nodiscard]] int valid_size(){
        int cnt=0;
        for(auto &lm:landmarks){
            if(!lm.bad){
                cnt++;
            }
        }
        return cnt;
    }

    /**
     * 清空物体
     */
    void ClearState(){
        is_init_velocity=false;
        is_initial = false;
        is_tracking = false;
        is_curr_visible=false;
        is_static=false;
        age=0;

        vel.SetZero();

        Debugv("ClearState() inst:{}",id);
    }

    /**
     * 滑动窗口内有点云的帧的数量
     * @return
     */
    [[nodiscard]] int GetPointsExtraFrames(){
        int num=0;
        for(int i=0;i<=kWinSize;++i){
            if(!points_extra[i].empty()){
                num++;
            }
        }
        return num;
    }


    /**
     * 判断3D点pw是否在边界框内
     * @param pw
     * @param factor
     * @return
     */
    bool IsInBoxPw(const Vec3d &pw,int frame=body.frame,double factor=4.){
        Vec3d pts_oi = WorldToObject(pw,frame);
        return  (std::abs(pts_oi.x()) < factor * box3d->dims.x()) &&
                (std::abs(pts_oi.y()) < factor * box3d->dims.y()) &&
                (std::abs(pts_oi.z()) < factor * box3d->dims.z());
    }

    bool IsInBoxPc(const Vec3d &pw,int frame=body.frame,double factor=4.){
        Vec3d pts_oi = CamToObject(pw,frame);
        return  (std::abs(pts_oi.x()) < factor * box3d->dims.x()) &&
        (std::abs(pts_oi.y()) < factor * box3d->dims.y()) &&
        (std::abs(pts_oi.z()) < factor * box3d->dims.z());
    }


    /**
     * 设置并返回路标点中具有深度的点的数量
     * @return
     */
    int set_triangle_num(){
        triangle_num=0;
        for(auto &lm:landmarks){
            if(lm.bad)
                continue;
            else if(lm.depth>0){
                triangle_num++;
            }
        }
        return triangle_num;
    }

    vector<Eigen::Vector3d> point3d_curr;
    std::list<LandmarkPoint> landmarks;

    vector<Vec3d> points_extra[kWinSize+1];
    PointCloud::Ptr points_extra_pcl[kWinSize+1];

    unsigned int id{0};

    cv::Scalar color;
    Box3D::Ptr box3d;
    Box2D::Ptr box2d;

    bool is_initial{false};//是否已经初始化位姿了
    bool is_tracking{true};//是否在滑动窗口中
    bool is_curr_visible{false};//当前帧是否可见
    bool is_static{false};//物体是运动的还是静态的
    bool is_init_velocity{false};

    State state[(kWinSize + 1)]{}; //物体的位姿

    Velocity vel,last_vel;//物体的速度
    Velocity point_vel;
    std::list<Velocity> history_vel;


    //优化过程中的变量
    double para_state[kWinSize + 1][kSizePose]{};
    double para_speed[1][kSpeedSize]{};
    double para_box[1][kBoxSize]{};
    double para_inv_depth[kInstFeatSize][kSizePoint]{};//逆深度参数数组


    int triangle_num{0};//已经三角化的路标点的数量
    int static_frame{1};//连续静止了多少帧

    int age{0};//初始化后走过了多少帧
    int lost_number{0};//已经连续多少帧未检测到特征了

    Box3D::Ptr boxes3d[(kWinSize + 1)]{};//实例关联的3d box

    std::list<State> history_pose;
};

}

#endif //DYNAMIC_VINS_DYNAMICFEATURE_H
