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


string GetPathKey(){
    string key;

    if(cfg::use_imu){
        key+="VIO";
    }
    else{
        key+="VO";
    }

    key+="_";

    if(cfg::slam==SLAM::kRaw){
        key+="raw";
    }
    else if(cfg::slam==SLAM::kNaive){
        key+="naive";
    }
    else if(cfg::slam==SLAM::kDynamic){
        key+="dynamic";
    }
    else{
        key+= "notdef";
    }
    key+="_";

    if(cfg::use_line){
        key+="LinePoint";
    }
    else{
        key+="PointOnly";
    }

    return key;
}


void IOParameter::SetParameters(const std::string &config_path,const std::string &seq_name)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(std::string("ERROR: Wrong path to settings:" + config_path));
    }

    if(cfg::use_imu){
        if(!fs["imu_topic"].isNone()){
            fs["imu_topic"] >> kImuTopic;
        }
        else{
            throw std::runtime_error(std::string("use_imu==true, but imu_topic was not set"));
        }
    }

    fs["visual_inst_duration"] >> kVisualInstDuration;

    std::string kBasicDir;
    fs["basic_dir"] >> kBasicDir;

    kOutputFolder = kBasicDir+"data/output/";

    static string mode_string = GetPathKey();

    kVinsResultPath = kOutputFolder + seq_name + "_" + mode_string + "_Odometry.txt";
    kObjectResultPath = kOutputFolder +seq_name+ "_" + mode_string + "_Object.txt";

    fs["use_dataloader"]>>use_dataloader;
    if(use_dataloader){

        if(fs["image_dataset_left"].isNone() || fs["image_dataset_right"].isNone()){
            std::cerr<<fmt::format("use_dataloader=true,but not set image_dataset_left or image_dataset_right")<<std::endl;
            std::terminate();
        }

        fs["image_dataset_left"] >> kImageDatasetLeft;
        kImageDatasetLeft = kImageDatasetLeft+seq_name+"/";

        fs["image_dataset_right"]>>kImageDatasetRight;
        kImageDatasetRight = kImageDatasetRight+seq_name+"/";

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

    if(!fs["pub_groundtruth_box"].isNone()){
        fs["pub_groundtruth_box"] >> is_pub_groundtruth_box;
        cout<<"pub_groundtruth_box=true"<<endl;
    }
    if(!fs["pub_predict_box"].isNone()){
        fs["pub_predict_box"] >> is_pub_predict_box;
        cout<<"pub_predict_box=true"<<endl;
    }
    if(!fs["pub_object_axis"].isNone()){
        fs["pub_object_axis"] >> is_pub_object_axis;
        cout<<"pub_object_axis=true"<<endl;
    }
    if(!fs["pub_object_trajectory"].isNone()){
        fs["pub_object_trajectory"] >> is_pub_object_trajectory;
        cout<<"pub_object_trajectory=true"<<endl;
    }

    if(!fs["show_input"].isNone()){
        fs["show_input"] >> is_show_input;
        cout<<"show_input=true"<<endl;
    }

    cv::FileNode inst_id_node=fs["print_inst_ids"];
    for(auto && it : inst_id_node){
        inst_ids_print.insert((int)it);
    }

    fs.release();

}



}
