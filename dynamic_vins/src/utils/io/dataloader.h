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

#include "utils/parameters.h"
#include "basic/def.h"
#include "basic/semantic_image.h"


namespace fs=std::filesystem;

namespace dynamic_vins{\


/**
 * 图像读取类
 */
class Dataloader{
public:
    using Ptr = std::shared_ptr<Dataloader>;
    Dataloader(const fs::path &left_images_dir,const fs::path &right_images_dir);

    //获取一帧图像
    std::tuple<cv::Mat,cv::Mat> LoadStereoImages();

    SemanticImage LoadStereo();

private:
    bool is_stereo{false};

    vector<string> left_paths_vector;
    vector<string> right_paths_vector;

    int index{0};
    double time{0};

    fs::path left_images_dir,right_images_dir;
};



}

#endif //DYNAMIC_VINS_CALLBACK_H
