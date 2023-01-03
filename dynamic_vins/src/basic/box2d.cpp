/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "box2d.h"

namespace dynamic_vins{\

/**
 * 计算两个box之间的IOU
 * @param bb_test
 * @param bb_gt
 * @return
 */
float Box2D::IoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt){
    auto in = (bb_test & bb_gt).area();
    auto un = bb_test.area() + bb_gt.area() - in;
    if (un <  DBL_EPSILON)
        return 0;
    return in / un;
}



}
