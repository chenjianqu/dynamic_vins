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


constexpr int kInferImageListSize=30;

constexpr double kFocalLength = 460.0;
constexpr int kWindowSize = 10;
constexpr int kNumFeat = 1000;

constexpr int kInstFeatSize=500;
constexpr int kSpeedSize=6;
constexpr int kBoxSize=3;

constexpr int kQueueSize=200;
constexpr double kDelay=0.005;

//图像归一化参数，注意是以RGB的顺序排序
inline float kSoloImgMean[3]={123.675, 116.28, 103.53};
inline float kSoloImgStd[3]={58.395, 57.12, 57.375};

constexpr int kBatchSize=1;
constexpr int kSoloTensorChannel=128;//张量的输出通道数应该是128

inline std::vector<float> kSoloNumGrids={40, 36, 24, 16, 12};//各个层级划分的网格数
inline std::vector<float> kSoloStrides={8, 8, 16, 32, 32};//各个层级的预测结果的stride


inline std::shared_ptr<spdlog::logger> vio_logger;
inline std::shared_ptr<spdlog::logger> tk_logger;
inline std::shared_ptr<spdlog::logger> sg_logger;


inline std::vector<std::vector<int>> kTensorQueueShapes{
        {1, 128, 12, 12},
        {1, 128, 16, 16},
        {1, 128, 24, 24},
        {1, 128, 36, 36},
        {1, 128, 40, 40},
        {1, 80, 12, 12},
        {1, 80, 16, 16},
        {1, 80, 24, 24},
        {1, 80, 36, 36},
        {1, 80, 40, 40},
        {1, 128, 96, 288}
};



enum SizeParameterization{
    kSizePose = 7,
    kSizeSpeedBias = 9,
    kSizeFeature = 1
};


enum StateOrder{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};

enum class SlamType{
    kRaw,
    kNaive,
    kDynamic
};

enum class DatasetType{
    kViode,
    kKitti
};


enum class SolverFlag{
    kInitial,
    kNonLinear
};

enum class MarginFlag{
    kMarginOld = 0,
    kMarginSecondNew = 1
};


class Config {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using Ptr=std::shared_ptr<Config>;

    explicit Config(const std::string &file_name);

    inline static double kInitDepth;
    inline static double kMinParallax;
    inline static double ACC_N, ACC_W;
    inline static double GYR_N, GYR_W;

    inline static std::vector<Eigen::Matrix3d> RIC;
    inline static std::vector<Eigen::Vector3d> TIC;

    inline static Eigen::Vector3d G{0.0, 0.0, 9.8};

    inline static double BIAS_ACC_THRESHOLD,BIAS_GYR_THRESHOLD;
    inline static double kMaxSolverTime;
    inline static int KNumIter;
    inline static int is_estimate_ex;
    inline static int is_estimate_td;
    inline static int ROLLING_SHUTTER;
    inline static std::string kExCalibResultPath;
    inline static std::string kVinsResultPath;
    inline static std::string kOutputFolder;
    inline static std::string kImuTopic;
    inline static int kRow, kCol;
    inline static double TD;
    inline static int kCamNum;
    inline static bool is_stereo;
    inline static int is_use_imu;
    inline static std::map<int, Eigen::Vector3d> pts_gt;
    inline static std::string kImage0Topic, kImage1Topic,kImage0SegTopic,kImage1SegTopic;
    inline static std::string FISHEYE_MASK;
    inline static std::vector<std::string> kCamPath;
    inline static int kMaxCnt; //每帧图像上的最多检测的特征数量
    inline static int kMaxDynamicCnt;
    inline static int kMinDist; //检测特征点时的最小距离
    inline static int kMinDynamicDist; //检测特征点时的最小距离
    inline static double kFThreshold;
    inline static int is_show_track;//是否显示光流跟踪的结果
    inline static int is_flow_back; //是否反向计算光流，判断之前光流跟踪的特征点的质量

    inline static std::unordered_map<unsigned int,int> ViodeKeyToIndex;
    inline static std::set<int> ViodeDynamicIndex;

    inline static std::string kDetectorOnnxPath;
    inline static std::string kDetectorSerializePath;

    inline static int kInputHeight,kInputWidth,kInputChannel=3;

    inline static SlamType slam;
    inline static DatasetType dataset;
    inline static bool is_input_seg;

    inline static std::vector<std::string> CocoLabelVector;

    inline static std::string kEstimatorLogPath;
    inline static std::string kEstimatorLogLevel;
    inline static std::string kEstimatorLogFlush;
    inline static std::string kFeatureTrackerLogPath;
    inline static std::string kFeatureTrackerLogLevel;
    inline static std::string kFeatureTrackerLogFlush;
    inline static std::string kSegmentorLogPath;
    inline static std::string kSegmentorLogLevel;
    inline static std::string kSegmentorLogFlush;

    inline static int kVisualInstDuration;

    inline static std::string kExtractorModelPath;

    inline static std::string kRaftFnetOnnxPath;
    inline static std::string kRaftFnetTensorrtPath;
    inline static std::string kRaftCnetOnnxPath;
    inline static std::string kRaftCnetTensorrtPath;
    inline static std::string kRaftUpdateOnnxPath;
    inline static std::string kRaftUpdateTensorrtPath;

    inline static int kSoloNmsPre;
    inline static int kSoloMaxPerImg;
    inline static std::string kSoloNmsKernel;
    inline static float kSoloNmsSigma;
    inline static float kSoloScoreThr;
    inline static float kSoloMaskThr;
    inline static float kSoloUpdateThr;

    inline static int kTrackingMaxAge;
    inline static int kTrackingNInit;

    inline static std::string kBasicDir;

    inline static std::atomic_bool ok{true};
};

using cfg = Config;


}

#endif

