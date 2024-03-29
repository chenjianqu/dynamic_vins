/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_PARAMETER_H
#define DYNAMIC_VINS_PARAMETER_H

#include <ros/ros.h>
#include <vector>
#include <fstream>
#include <map>
#include <iostream>
#include <exception>
#include <tuple>

#include <spdlog/spdlog.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>

#include "estimator/utility.h"
#include "camera_model.h"
#include "log_utils.h"


namespace dynamic_vins{\



using PointT=pcl::PointXYZRGB;
using PointCloud=pcl::PointCloud<PointT>;


constexpr double kFocalLength = 460.0;
constexpr int kWinSize = 10;
constexpr int kNumFeat = 1000;

constexpr int kQueueSize=200;
constexpr double kDelay=0.005;
constexpr int kImageQueueSize=100;


enum class SLAM{
    kRaw,
    kNaive,
    kDynamic,
};

enum class DatasetType{
    kViode,
    kKitti,
    kEuRoc,
    kCustom
};

class Config {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr=std::shared_ptr<Config>;

    explicit Config(const std::string &file_name,const std::string &seq_name);

    inline static std::string kExCalibResultPath;

    inline static int kCamNum;
    inline static bool is_stereo;
    inline static int use_imu;
    inline static std::map<int, Eigen::Vector3d> pts_gt;
    inline static std::string FISHEYE_MASK;

    inline static int kInputHeight,kInputWidth,kInputChannel=3;

    inline static SLAM slam;
    inline static DatasetType dataset;
    inline static std::string dataset_name;
    inline static bool is_input_seg; //输入是否有语义分割结果
    inline static bool is_only_frontend;
    inline static bool is_only_imgprocess;

    inline static bool use_line;

    inline static bool is_undistort_input{false};//是否对整个图像进行去畸变

    inline static int is_estimate_ex;
    inline static int is_estimate_td;

    inline static std::string kBasicDir;
    inline static std::string kDatasetSequence;

    inline static bool use_dense_flow{false};
    inline static bool use_background_flow{false};

    inline static bool use_plane_constraint{false};
    inline static bool use_det3d{false};

    inline static bool dst_mode{false};

    inline static bool is_vertical_draw{false};

    inline static std::atomic_bool ok{true};

};

using cfg = Config;


}

#endif

