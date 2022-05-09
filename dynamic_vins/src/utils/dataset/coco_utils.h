/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_COCO_UTILS_H
#define DYNAMIC_VINS_COCO_UTILS_H

#include <vector>
#include <string>

namespace dynamic_vins{\


class CocoUtils{
public:
    inline static std::vector<std::string> CocoLabel;

    static void SetParameters(const std::string &config_path);
};
using coco=CocoUtils;





}

#endif //DYNAMIC_VINS_COCO_UTILS_H
