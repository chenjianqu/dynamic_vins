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

string PrintFeaturesInfo(InstanceManager& im, bool output_lm, bool output_stereo);

void SaveTrajectory(InstanceManager& im);

string PrintInstancePoseInfo(InstanceManager& im,bool output_lm);

cv::Mat DrawTopView(InstanceManager& im,cv::Size size=cv::Size(600,600));

string PrintFactorDebugMsg(InstanceManager& im);


}

#endif //DYNAMIC_VINS_OUTPUT_H
