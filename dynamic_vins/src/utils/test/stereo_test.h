/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_STEREO_TEST_H
#define DYNAMIC_VINS_STEREO_TEST_H

#include "basic/semantic_image.h"


namespace dynamic_vins{\

/**
 * 根据视差图构建点云，并发布到rviz
 * @param img
 */
void StereoTest(const SemanticImage &img);


}

#endif //DYNAMIC_VINS_STEREO_TEST_H
