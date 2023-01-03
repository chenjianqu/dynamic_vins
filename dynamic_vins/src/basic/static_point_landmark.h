/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_STATIC_POINT_LANDMARK_H
#define DYNAMIC_VINS_STATIC_POINT_LANDMARK_H

#include "def.h"
#include "static_point_feature.h"

namespace dynamic_vins{\


class StaticPointLandmark{
public:
    const int feature_id;
    int start_frame;
    vector<StaticPointFeature> feats;
    size_t used_num;
    double depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    StaticPointLandmark(int _feature_id, int _start_frame)
    : feature_id(_feature_id), start_frame(_start_frame),
    used_num(0), depth(-1.0), solve_flag(0)
    {}

    int endFrame(){
        return start_frame + feats.size() - 1;
    }
};



}

#endif //DYNAMIC_VINS_STATIC_POINT_LANDMARK_H
