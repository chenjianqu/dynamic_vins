/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
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

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <spdlog/spdlog.h>

#include "utility/utility.h"
#include "utility/camera_model.h"
#include "utility/log_utils.h"


namespace dynamic_vins{\

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::pair;
using std::vector;
using std::tuple;

using namespace std::chrono_literals;
namespace fs=std::filesystem;


using PointT=pcl::PointXYZRGB;
using PointCloud=pcl::PointCloud<PointT>;

template <typename EigenType>
using EigenContainer = std::vector< EigenType ,Eigen::aligned_allocator<EigenType>>;

using Vec2d = Eigen::Vector2d;
using Vec3d = Eigen::Vector3d;
using Vec7d = Eigen::Matrix<double, 7, 1>;
using Mat2d = Eigen::Matrix2d;
using Mat3d = Eigen::Matrix3d;
using Mat4d = Eigen::Matrix4d;
using Mat23d = Eigen::Matrix<double, 2, 3>;
using Mat24d = Eigen::Matrix<double, 2, 4>;
using Mat34d = Eigen::Matrix<double, 3, 4>;
using Mat35d = Eigen::Matrix<double, 3, 5>;
using Mat36d = Eigen::Matrix<double, 3, 6>;
using Mat37d = Eigen::Matrix<double, 3, 7>;
using Quatd = Eigen::Quaterniond;


using VecVector3d = EigenContainer<Eigen::Vector3d>;
using VecMatrix3d = EigenContainer<Eigen::Matrix3d>;

constexpr double kFocalLength = 460.0;
constexpr int kWinSize = 10;
constexpr int kNumFeat = 1000;

constexpr int kQueueSize=200;
constexpr double kDelay=0.005;

enum class SlamType{
    kRaw,
    kNaive,
    kDynamic
};

enum class DatasetType{
    kViode,
    kKitti
};

class Config {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr=std::shared_ptr<Config>;

    explicit Config(const std::string &file_name);

    inline static int ROLLING_SHUTTER;
    inline static std::string kExCalibResultPath;
    inline static std::string kVinsResultPath;
    inline static std::string kOutputFolder;
    inline static std::string kImuTopic;

    inline static int kCamNum;
    inline static bool is_stereo;
    inline static int is_use_imu;
    inline static std::map<int, Eigen::Vector3d> pts_gt;
    inline static std::string kImage0Topic, kImage1Topic,kImage0SegTopic,kImage1SegTopic;
    inline static std::string FISHEYE_MASK;

    inline static int kInputHeight,kInputWidth,kInputChannel=3;

    inline static SlamType slam;
    inline static DatasetType dataset;
    inline static bool is_input_seg; //输入是否有语义分割结果
    inline static bool is_only_frontend;

    inline static int is_estimate_ex;
    inline static int is_estimate_td;

    inline static int kVisualInstDuration;

    inline static std::string kBasicDir;

    inline static bool use_dense_flow;

    inline static std::atomic_bool ok{true};
};

using cfg = Config;


}

#endif

