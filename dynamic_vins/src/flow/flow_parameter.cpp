/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "flow_parameter.h"

#include <opencv2/opencv.hpp>
#include "utils/log_utils.h"


namespace dynamic_vins{\

void FlowParameter::SetParameters(const std::string &config_path)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error("ERROR: Wrong path to settings:"+config_path);
    }
    std::string kBasicDir;
    fs["basic_dir"] >> kBasicDir;

    fs["fnet_onnx_path"] >> kRaftFnetOnnxPath;
    kRaftFnetOnnxPath = kBasicDir + kRaftFnetOnnxPath;
    fs["fnet_tensorrt_path"] >> kRaftFnetTensorrtPath;
    kRaftFnetTensorrtPath = kBasicDir + kRaftFnetTensorrtPath;
    fs["cnet_onnx_path"] >> kRaftCnetOnnxPath;
    kRaftCnetOnnxPath = kBasicDir + kRaftCnetOnnxPath;
    fs["cnet_tensorrt_path"] >> kRaftCnetTensorrtPath;
    kRaftCnetTensorrtPath = kBasicDir + kRaftCnetTensorrtPath;
    fs["update_onnx_path"] >> kRaftUpdateOnnxPath;
    kRaftUpdateOnnxPath = kBasicDir + kRaftUpdateOnnxPath;
    fs["update_tensorrt_path"] >> kRaftUpdateTensorrtPath;
    kRaftUpdateTensorrtPath = kBasicDir + kRaftUpdateTensorrtPath;

    fs["use_offline_flow"]>>use_offline_flow;

    if(use_offline_flow){
        if(fs["flow_offline_path"].isNone()){
            std::cerr<<fmt::format("use_offline_flow=true,but not set flow_offline_path")<<std::endl;
            std::terminate();
        }
        else{
            fs["flow_offline_path"] >> kFlowOfflinePath;
        }
    }


    fs.release();
}



}

