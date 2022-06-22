/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/


#ifndef DYNAMIC_VINS_STEREO_PARAMETERS_H
#define DYNAMIC_VINS_STEREO_PARAMETERS_H


#include <string>

namespace dynamic_vins{\


class StereoParameter{
public:

    inline static std::string kStereoPreprocessPath;
    inline static std::string kDatasetSequence;

    static void SetParameters(const std::string &config_path);
};

using stereo_para=StereoParameter;


}

#endif //DYNAMIC_VINS_STEREO_PARAMETERS_H
