/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "dataloader.h"
#include <chrono>
#include <ros/ros.h>

#include "io_parameters.h"
#include "utils/io_utils.h"

namespace dynamic_vins{\

using namespace std::chrono_literals;



SemanticImage CallBack::SyncProcess()
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
    return img;
}


Dataloader::Dataloader(){

    GetAllImageFiles(io_para::kImageDatasetLeft,left_names);
    GetAllImageFiles(io_para::kImageDatasetRight,right_names);

    cout<<"left:"<<left_names.size()<<endl;
    cout<<"right:"<<right_names.size()<<endl;
    if(left_names.size() != right_names.size()){
        cerr<< "left and right image number is not equal!"<<endl;
        return;
    }

    std::sort(left_names.begin(),left_names.end());
    std::sort(right_names.begin(),right_names.end());

    index=0;
    time=0.;
}


/**
 * 从磁盘上读取图片
 * @param delta_time 读取时间周期,ms
 * @return
 */
SemanticImage Dataloader::LoadStereo()
{
    SemanticImage img;

    if(index >= left_names.size()){
        cfg::ok=false;
        ros::shutdown();
        return img;
    }
    cout<<left_names[index]<<endl;

    img.color0  = cv::imread(left_names[index],-1);
    img.color1 = cv::imread(right_names[index],-1);

    std::filesystem::path name(left_names[index]);
    std::string name_stem =  name.stem().string();//获得文件名(不含后缀)

    img.time0 = time;
    img.time1 = time;
    img.seq = std::stoi(name_stem);



    time+=0.05; // 时间戳
    index++;

    return img;
}

void ImageViewer::Delay(int period){
    int delta_t =(int) tt.TocThenTic();
    int wait_time = period - delta_t;
    if(wait_time>0){
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
    }
}


void ImageViewer::ImageShow(cv::Mat &img,int period){
    int delta_t =(int) tt.TocThenTic();
    int wait_time = period - delta_t;
    wait_time = std::max(wait_time,1);

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

