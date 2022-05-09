/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_DET2D_PARAMETER_H
#define DYNAMIC_VINS_DET2D_PARAMETER_H

#include <vector>
#include <string>

namespace dynamic_vins{\


class Det2dParameter{
public:

    //图像归一化参数，注意是以RGB的顺序排序
    static inline float kSoloImgMean[3]={123.675, 116.28, 103.53};
    static inline float kSoloImgStd[3]={58.395, 57.12, 57.375};

    static constexpr int kBatchSize=1;
    static constexpr int kSoloTensorChannel=128;//张量的输出通道数应该是128

    static inline std::vector<float> kSoloNumGrids={40, 36, 24, 16, 12};//各个层级划分的网格数
    static inline std::vector<float> kSoloStrides={8, 8, 16, 32, 32};//各个层级的预测结果的stride


    static inline std::vector<std::vector<int>> kTensorQueueShapes{
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

    inline static int kSoloNmsPre;
    inline static int kSoloMaxPerImg;
    inline static std::string kSoloNmsKernel;
    inline static float kSoloNmsSigma;
    inline static float kSoloScoreThr;
    inline static float kSoloMaskThr;
    inline static float kSoloUpdateThr;

    inline static std::string kDetectorOnnxPath;
    inline static std::string kDetectorSerializePath;

    inline static std::string kWarnUpImagePath;

    inline static int model_input_width;
    inline static int model_input_height;
    inline static int model_input_channel;

    static void SetParameters(const std::string &config_path);

};

using det2d_para=Det2dParameter;

}

#endif //DYNAMIC_VINS_DET2D_PARAMETER_H
