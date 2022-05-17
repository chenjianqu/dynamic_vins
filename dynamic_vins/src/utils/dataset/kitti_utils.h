/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_KITTI_UTILS_H
#define DYNAMIC_VINS_KITTI_UTILS_H

#include "utils/parameters.h"

namespace dynamic_vins::kitti{\


    std::map<std::string,Eigen::MatrixXd> ReadCalibFile(const std::string &path);

    void SaveInstanceTrajectory(unsigned int frame_id,unsigned int track_id,std::string &type,
                                int truncated,int occluded,double alpha,Vec4d &box,
                                Vec3d &dims,Vec3d &location,double rotation_y,double score);

    void ClearTrajectoryFile();


}

#endif //DYNAMIC_VINS_KITTI_UTILS_H
