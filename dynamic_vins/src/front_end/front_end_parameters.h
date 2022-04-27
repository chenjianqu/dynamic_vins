//
// Created by chen on 2022/4/26.
//

#ifndef DYNAMIC_VINS_FRONT_END_PARAMETERS_H
#define DYNAMIC_VINS_FRONT_END_PARAMETERS_H

#include <string>

namespace dynamic_vins{\

class FrontendParemater{
public:
    inline static int kMaxCnt; //每帧图像上的最多检测的特征数量
    inline static int kMaxDynamicCnt;
    inline static int kMinDist; //检测特征点时的最小距离
    inline static int kMinDynamicDist; //检测特征点时的最小距离
    inline static double kFThreshold;
    inline static int is_show_track;//是否显示光流跟踪的结果
    inline static int is_flow_back; //是否反向计算光流，判断之前光流跟踪的特征点的质量

    inline static int kInputHeight,kInputWidth,kInputChannel=3;

    static void SetParameters(const std::string &config_path);

};

using fe_para = FrontendParemater;


}

#endif //DYNAMIC_VINS_FRONT_END_PARAMETERS_H