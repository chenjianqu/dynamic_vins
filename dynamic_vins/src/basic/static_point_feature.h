/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_STATIC_POINT_FEATURE_H
#define DYNAMIC_VINS_STATIC_POINT_FEATURE_H

#include "def.h"

namespace dynamic_vins{\

class StaticPointFeature{
public:
    StaticPointFeature(const Eigen::Matrix<double, 7, 1> &_point, double td){
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5);
        velocity.y() = _point(6);
        cur_td = td;
        is_stereo = false;
    }

    void rightObservation(const Eigen::Matrix<double, 7, 1> &_point){
        point_right.x() = _point(0);
        point_right.y() = _point(1);
        point_right.z() = _point(2);
        uvRight.x() = _point(3);
        uvRight.y() = _point(4);
        velocity_right.x() = _point(5);
        velocity_right.y() = _point(6);
        is_stereo = true;
    }

    double cur_td;
    Vec3d point, point_right;
    Vec2d uv, uvRight;
    Vec2d velocity, velocity_right;
    bool is_stereo;
};


}

#endif //DYNAMIC_VINS_STATIC_POINT_FEATURE_H
