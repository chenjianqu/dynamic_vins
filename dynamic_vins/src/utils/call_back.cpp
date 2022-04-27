/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "call_back.h"
#include <chrono>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

namespace dynamic_vins{\

using namespace std::chrono_literals;



SegImage CallBack::SyncProcess()
{
    SegImage img;
    while(cfg::ok.load(std::memory_order_seq_cst))
    {
        if((cfg::is_input_seg && (img0_buf.empty() || img1_buf.empty() || seg0_buf.empty() || seg1_buf.empty())) || //等待图片
        (!cfg::is_input_seg && (img0_buf.empty() || img1_buf.empty()))) {
            std::this_thread::sleep_for(2ms);
            continue;
        }
        ///下面以img0的时间戳为基准，找到与img0相近的图片
        img0_mutex.lock();
        img.color0= GetImageFromMsg(img0_buf.front());
        img.time0=img0_buf.front()->header.stamp.toSec();
        img.seq = img0_buf.front()->header.seq;//设置seq
        img0_buf.pop();
        img0_mutex.unlock();

        img1_mutex.lock();
        img.time1=img1_buf.front()->header.stamp.toSec();
        if(img.time0 + kDelay < img.time1){ //img0太早了
            img1_mutex.unlock();
            std::this_thread::sleep_for(2ms);
            continue;
        }
        else if(img.time1 + kDelay < img.time0){ //img1太早了
            while(img.time0 - img.time1 > kDelay){
                img1_buf.pop();
                img.time1=img1_buf.front()->header.stamp.toSec();
            }
        }
        img.color1= GetImageFromMsg(img1_buf.front());
        img1_buf.pop();
        img1_mutex.unlock();

        if(cfg::is_input_seg){
            seg0_mutex.lock();
            img.seg0_time=seg0_buf.front()->header.stamp.toSec();
            if(img.time0 + kDelay < img.seg0_time){ //img0太早了
                seg0_mutex.unlock();
                std::this_thread::sleep_for(2ms);
                continue;
            }
            else if(img.seg0_time + kDelay < img.time0){ //seg0太早了
                while(img.time0 - img.seg0_time > kDelay){
                    seg0_buf.pop();
                    img.seg0_time=seg0_buf.front()->header.stamp.toSec();
                }
            }
            img.seg0= GetImageFromMsg(seg0_buf.front());
            seg0_buf.pop();
            seg0_mutex.unlock();

            seg1_mutex.lock();
            img.seg1_time=seg1_buf.front()->header.stamp.toSec();
            if(img.time0 + kDelay < img.seg1_time){ //img0太早了
                seg1_mutex.unlock();
                std::this_thread::sleep_for(2ms);
                continue;
            }
            else if(img.seg1_time + kDelay < img.time0){ //seg1太早了
                while(img.time0 - img.seg1_time > kDelay){
                    seg1_buf.pop();
                    img.seg1_time=seg1_buf.front()->header.stamp.toSec();
                }
            }
            img.seg1= GetImageFromMsg(seg1_buf.front());
            seg1_buf.pop();
            seg1_mutex.unlock();
        }
        break;
    }
    return img;
}



}

