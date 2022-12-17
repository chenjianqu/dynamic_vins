/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_VELOCITY_H
#define DYNAMIC_VINS_VELOCITY_H

#include "utils/def.h"


namespace dynamic_vins{\


struct Velocity{
    Velocity()=default;

    void SetZero(){
        v.setZero();
        a.setZero();
    }

    /**
     * 返回 {Roioj,Poioj}
     * @param time_ij
     * @return
     */
    [[nodiscard]] tuple<Mat3d,Vec3d> RelativePose(double time_ij){
        return {Sophus::SO3d::exp(a*time_ij).matrix(),v*time_ij};
    }

    void SetVel(const Eigen::Isometry3d &m,double time_ij){
        v = m.translation() / time_ij;
        a = Sophus::SO3d(m.rotation()).log() / time_ij;
    }

    Vec3d v{0,0,0};
    Vec3d a{0,0,0};
};


struct Vel : Velocity{
    Vel()=default;
    Vec2d img_v;
};


}

#endif //DYNAMIC_VINS_VELOCITY_H
