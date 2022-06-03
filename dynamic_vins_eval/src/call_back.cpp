/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "call_back.h"
#include <chrono>
#include <thread>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>


using namespace std::chrono_literals;


Dataloader::Dataloader(const string &left_path){

    getAllImageFiles(left_path,left_names);

    cout<<"left:"<<left_names.size()<<endl;

    std::sort(left_names.begin(),left_names.end());

    index=0;
    time=0.;
    tt.Tic();
}

// 检查一个路径是否是目录
bool Dataloader::checkIsDir(const string &dir) {
    if (! std::filesystem::exists(dir)) {
        cout<<dir<<" not exists. Please check."<<endl;
        return false;
    }
    std::filesystem::directory_entry entry(dir);
    if (entry.is_directory())
        return true;
    return false;
}

// 搜索一个目录下所有的图像文件，以 jpg,jpeg,png 结尾的文件
void Dataloader::getAllImageFiles(const string& dir, vector<string> &files) {
    // 首先检查目录是否为空，以及是否是目录
    if (!checkIsDir(dir))
        return;

    // 递归遍历所有的文件
    std::filesystem::directory_iterator iters(dir);
    for(auto &iter: iters) {
        string file_path(dir);
        file_path += "/";
        file_path += iter.path().filename();

        // 查看是否是目录，如果是目录则循环递归
        if (checkIsDir(file_path)) {
            getAllImageFiles(file_path, files);
        }
        //不是目录则检查后缀是否是图像
        else {
            string extension = iter.path().extension(); // 获取文件的后缀名
            if (extension == ".jpg" || extension == ".png" || extension == ".jpeg") {
                files.push_back(file_path);
            }
        }
    }
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
        ros::shutdown();
        return img;
    }
    cout<<left_names[index]<<endl;

    img.color0  = cv::imread(left_names[index],-1);

    std::filesystem::path name(left_names[index]);
    std::string name_stem =  name.stem().string();//获得文件名(不含后缀)
    img.seq = std::stoi(name_stem);

    return img;
}


void Dataloader::ShowImage(const cv::Mat& img,int delta_time)
{
    int delta_t =(int) tt.TocThenTic();
    int wait_time = delta_time - delta_t;
    if(wait_time>0)
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));

    time+=0.05; // 时间戳
    index++;

    ///可视化
    bool pause=false;
    wait_time = std::max(wait_time,1);
    do{
        cv::imshow("Dataloader",img);
        int key = cv::waitKey(wait_time);
        if(key ==' '){
            pause = !pause;
        }
        else if(key== 27){ //ESC
            ros::shutdown();
            pause=false;
        }
        else if(key == 'r' || key == 'R'){
            index=0;
            time=0;
            pause=false;
        }
    } while (pause);

}



