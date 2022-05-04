/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "det2d_parameter.h"

#include <opencv2/opencv.hpp>



void dynamic_vins::Det2dParameter::SetParameters(const std::string &config_path) {

    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(std::string("ERROR: Wrong path to settings:"+config_path));
    }

    std::string kBasicDir;
    fs["basic_dir"] >> kBasicDir;

    fs["solo_onnx_path"] >> kDetectorOnnxPath;
    kDetectorOnnxPath = kBasicDir + kDetectorOnnxPath;
    fs["solo_serialize_path"] >> kDetectorSerializePath;
    kDetectorSerializePath = kBasicDir + kDetectorSerializePath;
    fs["SOLO_NMS_PRE"] >> kSoloNmsPre;
    fs["SOLO_MAX_PER_IMG"] >> kSoloMaxPerImg;
    fs["SOLO_NMS_KERNEL"] >> kSoloNmsKernel;
    fs["SOLO_NMS_SIGMA"] >> kSoloNmsSigma;
    fs["SOLO_SCORE_THR"] >> kSoloScoreThr;
    fs["SOLO_MASK_THR"] >> kSoloMaskThr;
    fs["SOLO_UPDATE_THR"] >> kSoloUpdateThr;

    if(!fs["warn_up_image"].isNone()){
        fs["warn_up_image"] >> kWarnUpImagePath;
    }

    fs.release();

}


