/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_LINE_H
#define DYNAMIC_VINS_LINE_H

#include <opencv2/opencv.hpp>

namespace dynamic_vins{\

struct Line{
    cv::Point2f StartPt;
    cv::Point2f EndPt;
    float lineWidth;
    cv::Point2f Vp;

    cv::Point2f Center;
    cv::Point2f unitDir; // [cos(theta), sin(theta)]
    float length;
    float theta;

    // para_a * x + para_b * y + c = 0
    float para_a;
    float para_b;
    float para_c;

    float image_dx;
    float image_dy;
    float line_grad_avg;

    float xMin;
    float xMax;
    float yMin;
    float yMax;
    unsigned int id;
    int colorIdx;
};

}

#endif //DYNAMIC_VINS_LINE_H
