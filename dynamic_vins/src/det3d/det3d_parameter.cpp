/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "det3d_parameter.h"

#include <opencv2/opencv.hpp>

namespace dynamic_vins{\



void Det3dParameter::SetParameters(const std::string &config_path)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(std::string("ERROR: Wrong path to settings:" + config_path));
    }

    fs["det3d_preprocess_path"] >> kDet3dPreprocessPath;
    fs["det3d_score_threshold"] >> kDet3dScoreThreshold;

    fs["use_offline_det3d"] >> use_offline;

    fs.release();
}


}

