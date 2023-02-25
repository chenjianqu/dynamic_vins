/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_IMAGE_VIEWER_H
#define DYNAMIC_VINS_IMAGE_VIEWER_H

#include "basic/def.h"

namespace dynamic_vins{\


/**
 * 图像可视化类
 */
class ImageViewer{
public:
    ImageViewer(){
        tt.Tic();
    }

    void ImageShow(cv::Mat &img,int period, int delay_frames=0);

    void Delay(int period);

private:
    std::queue<cv::Mat> img_queue;

    TicToc tt;
};

}

#endif //DYNAMIC_VINS_IMAGE_VIEWER_H
