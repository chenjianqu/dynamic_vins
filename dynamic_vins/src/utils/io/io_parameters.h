/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_IO_PARAMETERS_H
#define DYNAMIC_VINS_IO_PARAMETERS_H


#include <string>

namespace dynamic_vins{\

class IOParameter{
public:
    inline static std::string kImage0Topic, kImage1Topic,kImage0SegTopic,kImage1SegTopic;

    inline static int kVisualInstDuration;

    inline static std::string kVinsResultPath;
    inline static std::string kObjectResultPath;

    inline static std::string kOutputFolder;
    inline static std::string kImuTopic;

    inline static std::string kImageDatasetLeft;
    inline static std::string kImageDatasetRight;
    inline static int kImageDatasetPeriod;
    inline static bool use_dataloader;

    inline static bool is_pub_groundtruth_box{false};
    inline static bool is_pub_predict_box{false};
    inline static bool is_pub_object_axis{false};
    inline static bool is_pub_object_trajectory{false};

    inline static bool is_show_input{false};



    static void SetParameters(const std::string &config_path);
};

using io_para = IOParameter;





}


#endif //DYNAMIC_VINS_IO_PARAMETERS_H
