/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_BODY_H
#define DYNAMIC_VINS_BODY_H

#include <camodocal/camera_models/CameraFactory.h>

#include "basic/def.h"
#include "utils/parameters.h"
#include "vio_parameters.h"


namespace dynamic_vins{\

class BodyState{
public:

    [[nodiscard]] Vec3d CamToWorld(const Vec3d& pt,int frame_idx,int cam_idx=0) const{
        return  Rs[frame_idx] * ( ric[cam_idx] * pt + tic[cam_idx]) + Ps[frame_idx];
    }

    [[nodiscard]] Vec3d WorldToCam(const Vec3d& pt,int frame_idx,int cam_idx=0) const{
        return ric[cam_idx].transpose() * (Rs[frame_idx].transpose() * (pt - Ps[frame_idx]) - tic[cam_idx]);
    }

    Eigen::Matrix4d GetPoseInWorldFrame(){
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = Rs[frame];
        T.block<3, 1>(0, 3) = Ps[frame];
        return T;
    }

    Eigen::Matrix4d GetPoseInWorldFrame(int index){
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.block<3, 3>(0, 0) = Rs[index];
        T.block<3, 1>(0, 3) = Ps[index];
        return T;
    }

    /**
     * 获取某一时刻的相机位姿的3x4矩阵
     * @param index
     * @param cam_id
     * @return
     */
    Mat34d GetCamPose34d(int index,int cam_id){
        assert(cam_id == 0 || cam_id == 1);
        Vec3d t0 = Ps[index] + Rs[index] * tic[cam_id];
        Mat3d R0 = Rs[index] * ric[cam_id];
        Mat34d pose;
        pose.leftCols<3>() = R0.transpose();
        pose.rightCols<1>() = -R0.transpose() * t0;
        return pose;
    };

    void SetOptimizeParameters();
    void GetOptimizationParameters(Vec3d &origin_R0,Vec3d &origin_P0);

    Mat3d ric[2];
    Vec3d tic[2];
    Vec3d Ps[(kWinSize + 1)];
    Vec3d Vs[(kWinSize + 1)];
    Mat3d Rs[(kWinSize + 1)];
    Vec3d Bas[(kWinSize + 1)];
    Vec3d Bgs[(kWinSize + 1)];

    Vec3d g;

    double td{};
    double headers[(kWinSize + 1)]{};
    int frame{};

    double para_ex_pose[2][kSizePose]{};
    double para_pose[kWinSize + 1][kSizePose]{};
    double para_speed_bias[kWinSize + 1][kSizeSpeedBias]{};
    double para_point_features[kNumFeat][kSizePoint]{};
    double para_line_features[kNumFeat][kSizeLine];
    double para_Retrive_Pose[kSizePose]{};
    double para_td[1][1]{};
    double para_Tr[1][1]{};

    double frame_time;
    unsigned int seq_id;
};

extern BodyState body;


}

#endif //DYNAMIC_VINS_BODY_H
