/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_VIO_PARAMETERS_H
#define DYNAMIC_VINS_VIO_PARAMETERS_H

#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>

namespace dynamic_vins{\


constexpr double kDynamicDepthMin=0.1;//动态特征点深度的最小值
constexpr double kDynamicDepthMax=100;//动态特征点深度的最大值


enum SizeParameterization{
    kSizePose = 7,
    kSizeSpeedBias = 9,
    kSizeFeature = 1,
    kInstFeatSize=500,
    kSpeedSize=6,
    kBoxSize=3
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


enum class SolverFlag{
    kInitial,
    kNonLinear
};

enum class MarginFlag{
    kMarginOld = 0,
    kMarginSecondNew = 1
};




class VioParameters{
public:
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

    inline static int kInstanceInitMinNum;//为了初始化实例,在某一帧的最少三角化特征数量
    inline static double kInstanceStaticErrThreshold;//判断物体是否运动的重投影误差阈值

    inline static double TD;


    static void SetParameters(const std::string &config_path);
};

using para = VioParameters;


}

#endif //DYNAMIC_VINS_VIO_PARAMETERS_H
