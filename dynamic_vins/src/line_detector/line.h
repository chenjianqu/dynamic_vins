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

#include <memory>
#include <vector>

#include <opencv2/features2d.hpp>
#include "camodocal/camera_models/CameraFactory.h"

#include "line_descriptor/include/line_descriptor_custom.hpp"
#include "utils/camera_model.h"


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


class FrameLines{
public:
    using Ptr=std::shared_ptr<FrameLines>;

    void SetLines();

    void UndistortedLineEndPoints(camodocal::CameraPtr &cam);


    int frame_id;
    cv::Mat img;

    std::vector<Line> lines,un_lines;
    std::vector<unsigned int> line_ids;

    std::map<unsigned int,int> track_cnt;//每条线的跟踪次数

    // opencv3 lsd+lbd
    std::vector<cv::line_descriptor::KeyLine> keylsd;
    cv::Mat lbd_descr;
};


}



#endif //DYNAMIC_VINS_LINE_H
