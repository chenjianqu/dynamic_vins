/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_FRAME_LINES_H
#define DYNAMIC_VINS_FRAME_LINES_H

#include <memory>
#include <vector>

#include <opencv2/features2d.hpp>
#include <camodocal/camera_models/CameraFactory.h>

#include "line_descriptor/include/line_descriptor_custom.hpp"
#include "utils/camera_model.h"
#include "basic/line_feature.h"


namespace dynamic_vins{\


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



#endif //DYNAMIC_VINS_FRAME_LINES_H
