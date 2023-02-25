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
#include "utils/file_utils.h"
#include "utils/log_utils.h"

namespace dynamic_vins{\

using namespace std::chrono_literals;

Dataloader::Dataloader(const fs::path &left_images_dir_,const fs::path &right_images_dir_){
    is_stereo=true;
    left_images_dir = left_images_dir_;
    right_images_dir = right_images_dir_;

    GetAllImagePaths(left_images_dir, left_paths_vector);
    GetAllImagePaths(right_images_dir, right_paths_vector);

    Debugv("Dataloader() left_image_dir size:{}", left_paths_vector.size());
    Debugv("Dataloader() right_image_dir size:{}", right_paths_vector.size());

    if(left_paths_vector.size() != right_paths_vector.size()){
        cerr << fmt::format("left images number:{},right image number:{}, not equal!",
                            left_paths_vector.size(), right_paths_vector.size()) << endl;
        std::terminate();
    }

    std::sort(left_paths_vector.begin(), left_paths_vector.end());
    std::sort(right_paths_vector.begin(), right_paths_vector.end());

    //获取索引的最大值和最小值
    //vector<string> names;
    //GetAllImageNames(io_para::kImageDatasetLeft,names);
    //std::sort(names.begin(),names.end());
    //kStartIndex = stoi(fs::path(names[0]).stem().string());
    //kEndIndex = stoi(fs::path(*names.rbegin()).stem().string());
    //Debugv("Dataloader() kStartIndex:{}  kEndIndex:{}", kStartIndex,kEndIndex);

    //index=kStartIndex;
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

    if(index >= left_paths_vector.size()){
        return img;
    }
    cout << left_paths_vector[index] << endl;

    img.color0  = cv::imread(left_paths_vector[index], -1);
    img.color1 = cv::imread(right_paths_vector[index], -1);

    std::filesystem::path name(left_paths_vector[index]);
    std::string name_stem =  name.stem().string();//获得文件名(不含后缀)

    img.time0 = time;
    img.time1 = time;
    img.seq = std::stoi(name_stem);

    time+=0.05; // 时间戳
    index++;

    return img;
}



std::tuple<cv::Mat,cv::Mat> Dataloader::LoadStereoImages(){
    if(index >= left_paths_vector.size()){
        return {};
    }
    cout << left_paths_vector[index] << endl;

    cv::Mat color0  = cv::imread(left_paths_vector[index], -1);
    cv::Mat color1 = cv::imread(right_paths_vector[index], -1);

    index++;

    return {color0,color1};



}





}

