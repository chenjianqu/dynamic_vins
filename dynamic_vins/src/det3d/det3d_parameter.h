/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_DET3D_PARAMETER_H
#define DYNAMIC_VINS_DET3D_PARAMETER_H

#include <string>

namespace dynamic_vins{\


class Det3dParameter{
public:

    inline static std::string kDet3dPreprocessPath;
    inline static double kDet3dScoreThreshold;

    inline static std::string kGroundTruthPath;

    inline static bool use_offline;

    static void SetParameters(const std::string &config_path,const std::string &seq_name);
};

using det3d_para=Det3dParameter;


}

#endif //DYNAMIC_VINS_DET3D_PARAMETER_H
