//
// Created by chen on 2022/10/28.
//

#ifndef DYNAMIC_VINS_DEEPLEARNING_UTILS_H
#define DYNAMIC_VINS_DEEPLEARNING_UTILS_H

#include "camodocal/camera_models/CameraFactory.h"
#include "utils/def.h"


namespace dynamic_vins{\

void InitDeepLearningUtils(const string& config_path);


extern camodocal::CameraPtr left_cam_dl;
extern camodocal::CameraPtr right_cam_dl;

}


#endif //DYNAMIC_VINS_DEEPLEARNING_UTILS_H
