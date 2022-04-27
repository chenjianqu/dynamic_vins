//
// Created by chen on 2022/4/26.
//

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
