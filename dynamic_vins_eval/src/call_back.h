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
#include <string>
#include <vector>
#include <tuple>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>


using namespace std;

namespace fs=std::filesystem;


class TicToc{
public:
    TicToc(){
        Tic();
    }
    void Tic(){
        start_ = std::chrono::system_clock::now();
    }
    double Toc(){
        end_ = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_ - start_;
        return elapsed_seconds.count() * 1000;
    }
    double TocThenTic(){
        auto t= Toc();
        Tic();
        return t;
    }
    void TocPrintTic(const char* str){
        std::cout << str << ":" << Toc() << " ms" <<std::endl;
        Tic();
    }
private:
    std::chrono::time_point<std::chrono::system_clock> start_, end_;
};


struct SemanticImage{
    SemanticImage()= default;

    cv::Mat color0,color1;
    double time0,time1;
    cv::Mat gray0,gray1;

    unsigned int seq;
    bool exist_inst{false};//当前帧是否检测到物体
};


class Dataloader{
public:
    using Ptr = std::shared_ptr<Dataloader>;
    Dataloader(const string &left_path);

    // 检查一个路径是否是目录
    static bool checkIsDir(const string &dir);

    // 搜索一个目录下所有的图像文件，以 jpg,jpeg,png 结尾的文件
    void getAllImageFiles(const string& dir, vector<string> &files);

    //获取一帧图像
    SemanticImage LoadStereo();

    void ShowImage(const cv::Mat& img,int delta_time);

private:
    vector<string> left_names;

    int index{0};
    double time{0};

    TicToc tt;
};



#endif //DYNAMIC_VINS_CALLBACK_H
