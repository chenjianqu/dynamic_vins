/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_STEREO_H
#define DYNAMIC_VINS_STEREO_H

#include "utils/def.h"

namespace dynamic_vins{\

class MyStereoMatcher{
public:
    using Ptr=std::shared_ptr<MyStereoMatcher>;

    MyStereoMatcher(const std::string &config_path);

    void Launch(int seq);

    cv::Mat WaitResult();

    cv::Mat StereoMatch();


private:
    int img_seq_id{};
};




}

#endif //DYNAMIC_VINS_STEREO_H
