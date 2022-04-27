//
// Created by chen on 2022/4/26.
//

#include "front_end_parameters.h"

#include <opencv2/opencv.hpp>

namespace dynamic_vins{\


void FrontendParemater::SetParameters(const std::string &config_path)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(std::string("ERROR: Wrong path to settings:" + config_path));
    }

    kMaxCnt = fs["max_cnt"];
    kMaxDynamicCnt = fs["max_dynamic_cnt"];
    kMinDist = fs["min_dist"];
    kMinDynamicDist = fs["min_dynamic_dist"];
    kFThreshold = fs["F_threshold"];

    kInputHeight = fs["image_height"];
    kInputWidth = fs["image_width"];

    is_show_track = fs["show_track"];
    is_flow_back = fs["flow_back"];

    fs.release();

}






}


