/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/


#ifndef DYNAMIC_VINS_LINE_LANDMARK_H
#define DYNAMIC_VINS_LINE_LANDMARK_H

#include "utils/def.h"

namespace dynamic_vins{\



class LineFeature{
public:
    explicit LineFeature(const Vec4d &line){
        line_obs = line;
    }

    explicit LineFeature(const Eigen::Matrix<double, 8, 1> &line){
        line_obs = line.head<4>();
        line_obs_right = line.tail<4>();
        is_stereo=true;
    }

    explicit LineFeature(const Line &line){
        line_obs << line.StartPt.x,line.StartPt.y,line.EndPt.x,line.EndPt.y;
    }

    LineFeature(const Line &line,const Line &line_right){
        line_obs << line.StartPt.x,line.StartPt.y,line.EndPt.x,line.EndPt.y;
        line_obs_right << line_right.StartPt.x,line_right.StartPt.y,line_right.EndPt.x,line_right.EndPt.y;
        is_stereo=true;
    }

    Vec4d line_obs;   // 每一帧上的观测
    Vec4d line_obs_right; //右观测
    bool is_stereo{false};
    double z{};
    bool is_used{};
    double parallax{};
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    double dep_gradient{};
};


class LineLandmark{
public:
    LineLandmark(int _feature_id, int _start_frame):
        feature_id(_feature_id), start_frame(_start_frame),
        used_num(0), is_triangulation(false),solve_flag(0)
    {
        removed_cnt = 0;
        all_obs_cnt = 1;
    }

    [[nodiscard]] int endFrame() const{
        return start_frame + feats.size() - 1;
    }


    const int feature_id;
    int start_frame;

    //feature_per_frame 是个向量容器，存着这个特征在每一帧上的观测量。
    //如：feature_per_frame[0]，存的是ft在start_frame上的观测值; feature_per_frame[1]存的是start_frame+1上的观测
    vector<LineFeature> feats;

    int used_num;
    bool is_outlier{};
    bool is_margin{};
    bool is_triangulation;
    Vec6d line_plucker;

    Vec4d obs_init;
    Vec4d obs_j;
    Vec6d line_plk_init; // used to debug
    Vec3d ptw1;  // used to debug
    Vec3d ptw2;  // used to debug
    Eigen::Vector3d tj_;   // tij
    Eigen::Matrix3d Rj_;
    Eigen::Vector3d ti_;   // tij
    Eigen::Matrix3d Ri_;
    int removed_cnt;
    int all_obs_cnt;    // 总共观测多少次了？

    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;
};



}

#endif //DYNAMIC_VINS_LINE_LANDMARK_H
