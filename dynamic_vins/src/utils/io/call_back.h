/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_CALLBACK_H
#define DYNAMIC_VINS_CALLBACK_H

#include <memory>
#include <queue>
#include <filesystem>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include "utils/parameters.h"
#include "front_end/semantic_image.h"
#include "utils/def.h"


namespace fs=std::filesystem;



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
        if(img0_buf.size() == kQueueSize){
            img0_buf.pop_front();
        }
        img0_buf.push_back(img_msg);
        img0_mutex.unlock();
    }

    void Seg0Callback(const sensor_msgs::ImageConstPtr &img_msg){
        seg0_mutex.lock();
        if(seg0_buf.size() == kQueueSize){
            seg0_buf.pop_front();
        }
        seg0_buf.push_back(img_msg);
        seg0_mutex.unlock();
    }

    void Img1Callback(const sensor_msgs::ImageConstPtr &img_msg){
        img1_mutex.lock();
        if(img1_buf.size() == kQueueSize){
            img1_buf.pop_front();
        }
        img1_buf.push_back(img_msg);
        img1_mutex.unlock();
    }

    void Seg1Callback(const sensor_msgs::ImageConstPtr &img_msg){
        seg1_mutex.lock();
        if(seg1_buf.size() == kQueueSize){
            seg1_buf.pop_front();
        }
        seg1_buf.push_back(img_msg);
        seg1_mutex.unlock();
    }


    SemanticImage SyncProcess();


private:
    std::list<sensor_msgs::ImageConstPtr> img0_buf;
    std::list<sensor_msgs::ImageConstPtr> seg0_buf;
    std::list<sensor_msgs::ImageConstPtr> img1_buf;
    std::list<sensor_msgs::ImageConstPtr> seg1_buf;

    std::mutex m_buf;
    std::mutex img0_mutex,img1_mutex,seg0_mutex,seg1_mutex;
};


class Dataloader{
public:
    using Ptr = std::shared_ptr<Dataloader>;
    Dataloader();

    //获取一帧图像
    SemanticImage LoadStereo(int delta_time);

private:
    vector<string> left_names;
    vector<string> right_names;

    int index{0};
    double time{0};

    TicToc tt;
};


}

#endif //DYNAMIC_VINS_CALLBACK_H
