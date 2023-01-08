/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_MOT_PARAMETER_H
#define DYNAMIC_VINS_MOT_PARAMETER_H

#include <string>

namespace dynamic_vins{\

class MotParameter{
public:
    static constexpr int64_t kBudget = 100;//外观特征的存储帧数
    static constexpr int64_t kFeatDim = 512;//外观特征的维度

    static constexpr int STATE_DIM = 7;//卡尔曼滤波的状态维度
    static constexpr int MEASURE_DIM = 4;//卡尔曼滤波的测量维度

    static inline float kReIdImgMean[3]={0.485f, 0.456f, 0.406f};//ReId网络的预处理归一化均值
    static inline float kReIdImgStd[3]={0.229f, 0.224f, 0.225f};

    inline static std::string kExtractorModelPath;
    inline static int kTrackingMaxAge;
    inline static int kTrackingNInit;


    static void SetParameters(const std::string &config_path);
};

using mot_para = MotParameter;





}



#endif //DYNAMIC_VINS_MOT_PARAMETER_H
