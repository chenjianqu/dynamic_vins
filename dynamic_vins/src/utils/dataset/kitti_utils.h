//
// Created by chen on 2022/4/25.
//

#ifndef DYNAMIC_VINS_KITTI_UTILS_H
#define DYNAMIC_VINS_KITTI_UTILS_H

#include "utils/parameters.h"

namespace dynamic_vins::kitti{\


std::map<std::string,Eigen::MatrixXd> ReadCalibFile(const std::string &path);



}

#endif //DYNAMIC_VINS_KITTI_UTILS_H
