/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_STATE_H
#define DYNAMIC_VINS_STATE_H

#include "utils/def.h"

namespace dynamic_vins{\


/**
 * 位姿
 */
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

    /**
     * 获的变换矩阵
     * @return
     */
    [[nodiscard]] Eigen::Isometry3d GetTransform() const{
        Eigen::Isometry3d m = Eigen::Isometry3d::Identity();
        m.rotate(R);
        m.pretranslate(P);
        return m;
    }

    void SetPose(const Eigen::Isometry3d &m){
        R = m.rotation();
        P = m.translation();
    }

    Mat3d R;
    Vec3d P;
    double time{0};
};





}


#endif //DYNAMIC_VINS_STATE_H
