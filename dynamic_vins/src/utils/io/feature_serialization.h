/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_FEATURE_SERIALIZATION_H
#define DYNAMIC_VINS_FEATURE_SERIALIZATION_H

#include "basic/def.h"
#include "line_detector/frame_lines.h"

namespace dynamic_vins{\

/**
 * 特征序列化和反序列化，用于DEBUG
 */

void SerializePointFeature(const string& path,
                           const std::map<unsigned int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &points);

std::map<unsigned int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>>
DeserializePointFeature(const string& path);

void SerializeLineFeature(const string& path,const std::map<unsigned int, std::vector<std::pair<int,Line>>> &lines);

std::map<unsigned int, std::vector<std::pair<int,Line>>>
DeserializeLineFeature(const string& path);


}


#endif //DYNAMIC_VINS_FEATURE_SERIALIZATION_H
