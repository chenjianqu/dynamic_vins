/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "image_viewer.h"
#include <ros/ros.h>
#include "utils/parameters.h"

namespace dynamic_vins{\


/**
 * 阻塞延时
 * @param period 延时时长
 */
void ImageViewer::Delay(int period){
    int delta_t =(int) tt.TocThenTic();
    int wait_time = period - delta_t;
    if(wait_time>0){
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
    }
}


/**
 * 图像显示
 * @param img 图像
 * @param period 显示时长(ms)
 * @param delay_frames 延时帧数
 */
void ImageViewer::ImageShow(cv::Mat &img,int period, int delay_frames){
    int delta_t =(int) tt.TocThenTic();
    int wait_time = period - delta_t;
    wait_time = std::max(wait_time,1);

    img_queue.push(img);//使用队列存储图片

    if(img_queue.size()>delay_frames){
        img = img_queue.front();
        img_queue.pop();
        bool pause=false;
        do{
            cv::imshow("ImageViewer",img);
            int key = cv::waitKey(wait_time);
            if(key ==' '){
                pause = !pause;
            }
            else if(key== 27){ //ESC
                cfg::ok=false;
                ros::shutdown();
                pause=false;
            }
            wait_time=period;
        } while (pause);
    }
}


}

