/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_MOT_DEF_H
#define DYNAMIC_VINS_MOT_DEF_H

#include <opencv2/opencv.hpp>

namespace dynamic_vins{\


struct Track {
    int id;
    cv::Rect2f box;
};

}

#endif //DYNAMIC_VINS_MOT_DEF_H
