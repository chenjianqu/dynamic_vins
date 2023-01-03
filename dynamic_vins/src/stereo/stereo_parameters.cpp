/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/


#include "stereo_parameters.h"

#include "basic/def.h"


namespace dynamic_vins{\



void StereoParameter::SetParameters(const std::string &config_path,const std::string &seq_name)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(std::string("ERROR: Wrong path to settings:" + config_path));
    }

    kDatasetSequence = seq_name;

    fs["stereo_preprocess_path"] >> kStereoPreprocessPath;


    fs.release();
}


}


