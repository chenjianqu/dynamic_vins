/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_LINE_FEATURE_H
#define DYNAMIC_VINS_LINE_FEATURE_H

#include "utils/def.h"


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


class LineFeature{
public:
    explicit LineFeature(const Vec4d &line){
        line_obs = line;
    }

    explicit LineFeature(const Eigen::Matrix<double, 8, 1> &line){
        line_obs = line.head<4>();
        line_obs_right = line.tail<4>();
        is_stereo=true;
    }

    explicit LineFeature(const Line &line){
        line_obs << line.StartPt.x,line.StartPt.y,line.EndPt.x,line.EndPt.y;
    }

    LineFeature(const Line &line,const Line &line_right){
        line_obs << line.StartPt.x,line.StartPt.y,line.EndPt.x,line.EndPt.y;
        line_obs_right << line_right.StartPt.x,line_right.StartPt.y,line_right.EndPt.x,line_right.EndPt.y;
        is_stereo=true;
    }

    Vec4d line_obs;   // 每一帧上的观测
    Vec4d line_obs_right; //右观测
    bool is_stereo{false};
    double z{};
    bool is_used{};
    double parallax{};
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    double dep_gradient{};
};


}

#endif //DYNAMIC_VINS_LINE_FEATURE_H
