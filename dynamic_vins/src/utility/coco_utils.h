//
// Created by chen on 2022/4/25.
//

#ifndef DYNAMIC_VINS_COCO_UTILS_H
#define DYNAMIC_VINS_COCO_UTILS_H

#include <vector>
#include <string>

namespace dynamic_vins{\


class CocoUtils{
public:
    inline static std::vector<std::string> CocoLabelVector;

    static void SetParameters(const std::string &config_path);
};
using coco=CocoUtils;





}

#endif //DYNAMIC_VINS_COCO_UTILS_H
