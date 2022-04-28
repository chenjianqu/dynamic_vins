/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_MOT_PARAMETER_H
#define DYNAMIC_VINS_MOT_PARAMETER_H

#include <string>

namespace dynamic_vins{\

class MotParameter{
public:
    inline static std::string kExtractorModelPath;
    inline static int kTrackingMaxAge;
    inline static int kTrackingNInit;


    static void SetParameters(const std::string &config_path);
};

using mot_para = MotParameter;





}



#endif //DYNAMIC_VINS_MOT_PARAMETER_H
