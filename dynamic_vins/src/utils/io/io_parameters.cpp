/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "io_parameters.h"

#include "utils/parameters.h"

namespace dynamic_vins{\

void IOParameter::SetParameters(const std::string &config_path)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(std::string("ERROR: Wrong path to settings:" + config_path));
    }

    if(cfg::is_use_imu){
        fs["imu_topic"] >> kImuTopic;
    }

    fs["visual_inst_duration"] >> kVisualInstDuration;

    std::string kBasicDir;
    fs["basic_dir"] >> kBasicDir;

    kOutputFolder = kBasicDir+"data/output/";

    kVinsResultPath = kOutputFolder + cfg::kDatasetSequence + "_ego-motion.txt";

    kObjectResultPath = kOutputFolder +cfg::kDatasetSequence+"_object.txt";

    //清空
    std::ofstream fout;
    fout.open(kVinsResultPath, std::ios::out);
    fout.close();
    fout.open(kObjectResultPath, std::ios::out);
    fout.close();

    fs["use_dataloader"]>>use_dataloader;
    if(use_dataloader){
        fs["image_dataset_left"]>>kImageDatasetLeft;
        fs["image_dataset_right"]>>kImageDatasetRight;
        fs["image_dataset_period"] >> kImageDatasetPeriod;
    }
    else{
        fs["image0_topic"] >> kImage0Topic;
        fs["image1_topic"] >> kImage1Topic;

        if(cfg::is_input_seg){
            fs["image0_segmentation_topic"] >> kImage0SegTopic;
            fs["image1_segmentation_topic"] >> kImage1SegTopic;
        }
    }

    fs.release();

}



}
