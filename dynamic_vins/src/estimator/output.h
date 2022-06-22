/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_OUTPUT_H
#define DYNAMIC_VINS_OUTPUT_H

#include "utils/def.h"
#include "estimator_insts.h"

namespace dynamic_vins{\

string PrintInstanceInfo(InstanceManager& im,bool output_lm,bool output_stereo);

void SaveTrajectory(InstanceManager& im);

string PrintInstancePoseInfo(InstanceManager& im,bool output_lm);



}

#endif //DYNAMIC_VINS_OUTPUT_H
