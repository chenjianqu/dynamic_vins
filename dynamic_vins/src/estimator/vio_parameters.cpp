/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "vio_parameters.h"

#include <opencv2/opencv.hpp>

#include "utils/dataset/kitti_utils.h"
#include "utils/def.h"
#include "utils/parameters.h"


namespace dynamic_vins{\






void VioParameters::SetParameters(const std::string &config_path)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(std::string("ERROR: Wrong path to settings:" + config_path));
    }

    kMaxSolverTime = fs["max_solver_time"];
    KNumIter = fs["max_num_iterations"];
    kMinParallax = fs["keyframe_parallax"];
    kMinParallax = kMinParallax / kFocalLength;

    if(cfg::use_imu){
        ACC_N = fs["acc_n"];
        ACC_W = fs["acc_w"];
        GYR_N = fs["gyr_n"];
        GYR_W = fs["gyr_w"];
        G.z() = fs["g_norm"];
    }

    fs["INIT_DEPTH"] >> kInitDepth;
    fs["BIAS_ACC_THRESHOLD"]>>BIAS_ACC_THRESHOLD;
    fs["BIAS_GYR_THRESHOLD"]>>BIAS_GYR_THRESHOLD;

    TD = fs["td"];

    if(cfg::use_line){
        if(fs["line_min_obs"].isNone()){
            kLineMinObs=5;
        }
        else{
            kLineMinObs = fs["line_min_obs"];
        }
    }

    if(cfg::slam == SLAM::kDynamic){
        if(fs["instance_init_min_num"].isNone()){
            throw std::runtime_error("VioParameters::SetParameters() fs[\"instance_static_err_threshold\"].isNone()");
        }
        kInstanceStaticErrThreshold = fs["instance_static_err_threshold"];
        if(fs["instance_init_min_num"].isNone()){
            throw std::runtime_error("VioParameters::SetParameters() fs[\"instance_init_min_num\"].isNone()");
        }
        kInstanceInitMinNum = fs["instance_init_min_num"];
    }

    if(!fs["print_detail"].isNone()){
        fs["print_detail"]>>is_print_detail;
    }


    fs.release();



}










}
