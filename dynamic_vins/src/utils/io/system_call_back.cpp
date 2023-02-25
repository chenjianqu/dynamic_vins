/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "system_call_back.h"
#include "utils/log_utils.h"
#include "io_parameters.h"

namespace dynamic_vins{\


SystemCallBack::SystemCallBack(std::shared_ptr<Estimator> estimator,ros::NodeHandle &nh_):e(estimator),nh(nh_){

    sub_img0 = nh.subscribe(io_para::kImage0Topic, 100, &SystemCallBack::Img0Callback, this);
    sub_img1 = nh.subscribe(io_para::kImage1Topic, 100, &SystemCallBack::Img1Callback, this);

    if(cfg::is_input_seg){
        sub_seg0 = nh.subscribe(io_para::kImage0SegTopic, 100, &SystemCallBack::Seg0Callback, this);
        sub_seg1 = nh.subscribe(io_para::kImage1SegTopic, 100, &SystemCallBack::Seg1Callback, this);
    }

    sub_imu = nh.subscribe(io_para::kImuTopic, 2000, &SystemCallBack::ImuCallback,this,
                          ros::TransportHints().tcpNoDelay());

    sub_restart = nh.subscribe("/vins_restart", 100, &SystemCallBack::RestartCallback,this);
    sub_terminal = nh.subscribe("/vins_terminal", 100, &SystemCallBack::TerminalCallback,this);
    sub_imu_switch = nh.subscribe("/vins_imu_switch", 100, &SystemCallBack::ImuSwitchCallback,this);
    sub_cam_switch = nh.subscribe("/vins_cam_switch", 100, &SystemCallBack::CamSwitchCallback,this);

}



void SystemCallBack::ImuCallback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    Vector3d acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    Vector3d gyr(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    e->InputIMU(t, acc, gyr);
}

void SystemCallBack::RestartCallback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true){
        Warnv("restart the e!");
        e->ClearState();
        e->SetParameter();
    }
}

void SystemCallBack::TerminalCallback(const std_msgs::BoolConstPtr &terminal_msg)
{
    if (terminal_msg->data == true){
        cerr<<"terminal the e!"<<endl;
        ros::shutdown();
        cfg::ok.store(false,std::memory_order_seq_cst);
    }
}

void SystemCallBack::ImuSwitchCallback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true){
        Warnv("use IMU!");
        e->ChangeSensorType(1, cfg::is_stereo);
    }
    else{
        Warnv("disable IMU!");
        e->ChangeSensorType(0, cfg::is_stereo);
    }
}

void SystemCallBack::CamSwitchCallback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true){
        Warnv("use stereo!");
        e->ChangeSensorType(cfg::use_imu, 1);
    }
    else{
        Warnv("use mono camera (left)!");
        e->ChangeSensorType(cfg::use_imu, 0);
    }
}




/**
 * 对接收到的图像消息进行同步
 * @return
 */
SemanticImage SystemCallBack::SyncProcess()
{
    SemanticImage img;
    while(cfg::ok.load(std::memory_order_seq_cst))
    {
        if( (cfg::is_input_seg  && (img0_buf.empty() || img1_buf.empty() || seg0_buf.empty() || seg1_buf.empty())) || //等待图片
        (!cfg::is_input_seg && (img0_buf.empty() || img1_buf.empty()))) {
            std::this_thread::sleep_for(2ms);
            continue;
        }
        Debugs("SyncProcess | msg size:{} {} {} {}",img0_buf.size(),img1_buf.size(),seg0_buf.size(),seg1_buf.size());
        ///下面以img0的时间戳为基准，找到与img0相近的图片
        img0_mutex.lock();
        img.color0= GetImageFromMsg(img0_buf.front());
        img.time0=img0_buf.front()->header.stamp.toSec();
        img.seq = img0_buf.front()->header.seq;//设置seq
        img0_buf.pop_front();
        img0_mutex.unlock();
        ///获取img1
        img1_mutex.lock();
        img.time1=img1_buf.front()->header.stamp.toSec();
        if(img.time0 + kDelay < img.time1){ //img0太早了
            img1_mutex.unlock();
            std::this_thread::sleep_for(2ms);
            continue;
        }
        else if(img.time1 + kDelay < img.time0){ //img1太早了
            while(img.time0 - img.time1 > kDelay){
                img1_buf.pop_front();
                img.time1=img1_buf.front()->header.stamp.toSec();
            }
        }
        img.color1= GetImageFromMsg(img1_buf.front());
        img1_buf.pop_front();
        img1_mutex.unlock();

        if(cfg::is_input_seg){
            ///获取 seg0
            seg0_mutex.lock();
            img.seg0_time=seg0_buf.front()->header.stamp.toSec();
            if(img.time0 + kDelay < img.seg0_time){ //img0太早了
                seg0_mutex.unlock();
                std::this_thread::sleep_for(2ms);
                continue;
            }
            else if(img.seg0_time + kDelay < img.time0){ //seg0太早了
                while(img.time0 - img.seg0_time > kDelay){
                    seg0_buf.pop_front();
                    img.seg0_time=seg0_buf.front()->header.stamp.toSec();
                }
            }
            img.seg0= GetImageFromMsg(seg0_buf.front());
            seg0_buf.pop_front();
            seg0_mutex.unlock();
            ///获取seg1
            seg1_mutex.lock();
            img.seg1_time=seg1_buf.front()->header.stamp.toSec();
            if(img.time0 + kDelay < img.seg1_time){ //img0太早了
                seg1_mutex.unlock();
                std::this_thread::sleep_for(2ms);
                continue;
            }
            else if(img.seg1_time + kDelay < img.time0){ //seg1太早了
                while(img.time0 - img.seg1_time > kDelay){
                    seg1_buf.pop_front();
                    img.seg1_time=seg1_buf.front()->header.stamp.toSec();
                }
            }
            img.seg1= GetImageFromMsg(seg1_buf.front());
            seg1_buf.pop_front();
            seg1_mutex.unlock();
        }
        break;
    }

    if(cfg::dataset==DatasetType::kViode){
        static int global_seq_id=0;
        img.seq = global_seq_id++;
    }

    return img;
}







}
