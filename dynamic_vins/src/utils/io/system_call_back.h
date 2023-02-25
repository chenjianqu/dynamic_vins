/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_SYSTEM_CALL_BACK_H
#define DYNAMIC_VINS_SYSTEM_CALL_BACK_H

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>

#include "basic/def.h"
#include "estimator/estimator.h"
#include "basic/def.h"
#include "basic/semantic_image.h"
#include "utils/parameters.h"

namespace dynamic_vins{ \

class SystemCallBack{
public:
    using Ptr=std::unique_ptr<SystemCallBack>;
    SystemCallBack(std::shared_ptr<Estimator> estimator,ros::NodeHandle &nh_);

    void ImuCallback(const sensor_msgs::ImuConstPtr &imu_msg);

    void RestartCallback(const std_msgs::BoolConstPtr &restart_msg);

    void TerminalCallback(const std_msgs::BoolConstPtr &terminal_msg);

    void ImuSwitchCallback(const std_msgs::BoolConstPtr &switch_msg);

    void CamSwitchCallback(const std_msgs::BoolConstPtr &switch_msg);

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
    std::shared_ptr<Estimator> e;
    ros::NodeHandle &nh;

    ros::Subscriber sub_imu,sub_img0,sub_img1;
    ros::Subscriber sub_seg0,sub_seg1;
    ros::Subscriber sub_restart,sub_terminal,sub_imu_switch,sub_cam_switch;


    std::list<sensor_msgs::ImageConstPtr> img0_buf;
    std::list<sensor_msgs::ImageConstPtr> seg0_buf;
    std::list<sensor_msgs::ImageConstPtr> img1_buf;
    std::list<sensor_msgs::ImageConstPtr> seg1_buf;

    std::mutex m_buf;
    std::mutex img0_mutex,img1_mutex,seg0_mutex,seg1_mutex;
};

}

#endif //DYNAMIC_VINS_SYSTEM_CALL_BACK_H
