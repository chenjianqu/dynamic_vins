/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_CALLBACK_H
#define DYNAMIC_VINS_CALLBACK_H

#include <memory>
#include <queue>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include "parameters.h"
#include "front_end/segment_image.h"

namespace dynamic_vins{\


class CallBack{
public:
    using Ptr=std::unique_ptr<CallBack>;
    CallBack(){}

    inline cv::Mat GetImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg){
        cv_bridge::CvImageConstPtr ptr= cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
        return ptr->image.clone();
    }

    void Img0Callback(const sensor_msgs::ImageConstPtr &img_msg){
        img0_mutex.lock();
        if(img0_buf.size() < kQueueSize)
            img0_buf.push(img_msg);
        img0_mutex.unlock();
    }

    void Seg0Callback(const sensor_msgs::ImageConstPtr &img_msg){
        seg0_mutex.lock();
        if(seg0_buf.size() < kQueueSize)
            seg0_buf.push(img_msg);
        seg0_mutex.unlock();
    }

    void Img1Callback(const sensor_msgs::ImageConstPtr &img_msg){
        img1_mutex.lock();
        if(img1_buf.size() < kQueueSize)
            img1_buf.push(img_msg);
        img1_mutex.unlock();
    }

    void Seg1Callback(const sensor_msgs::ImageConstPtr &img_msg){
        seg1_mutex.lock();
        if(seg1_buf.size() < kQueueSize)
            seg1_buf.push(img_msg);
        seg1_mutex.unlock();
    }


    SegImage SyncProcess();


private:
    std::queue<sensor_msgs::ImageConstPtr> img0_buf;
    std::queue<sensor_msgs::ImageConstPtr> seg0_buf;
    std::queue<sensor_msgs::ImageConstPtr> img1_buf;
    std::queue<sensor_msgs::ImageConstPtr> seg1_buf;

    std::mutex m_buf;
    std::mutex img0_mutex,img1_mutex,seg0_mutex,seg1_mutex;
};


}

#endif //DYNAMIC_VINS_CALLBACK_H
