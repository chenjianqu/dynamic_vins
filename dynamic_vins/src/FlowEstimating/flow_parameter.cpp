//
// Created by chen on 2022/4/25.
//

#include "flow_parameter.h"

#include <opencv2/opencv.hpp>


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

    fs.release();
}



}

