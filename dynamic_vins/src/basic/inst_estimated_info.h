/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_INST_ESTIMATED_INFO_H
#define DYNAMIC_VINS_INST_ESTIMATED_INFO_H

#include "def.h"

namespace dynamic_vins{\

class InstEstimatedInfo{
public:
    double time;
    Mat3d R;
    Vec3d P{0,0,0};
    Vec3d v{0,0,0};
    Vec3d a{0,0,0};

    Vec3d dims{0,0,0};
    Vec3d avg_point{0,0,0};

    bool is_init{false};
    bool is_init_velocity{false};
    bool is_static{false};
};



}


#endif //DYNAMIC_VINS_INST_ESTIMATED_INFO_H
