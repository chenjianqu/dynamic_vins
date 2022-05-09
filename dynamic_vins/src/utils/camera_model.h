/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_CAMERA_MODEL_H
#define DYNAMIC_VINS_CAMERA_MODEL_H

#include <memory>
#include <string>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>

#include "utils/def.h"

namespace dynamic_vins{\


class PinHoleCamera{
public:
    using Ptr=std::shared_ptr<PinHoleCamera>;
    PinHoleCamera(){}


    bool ReadFromYamlFile(const std::string& filename);

    void LiftProjective(const Vec2d& p, Vec3d& P) const;

    void ProjectPoint(const Vec3d& p3d, Vec2d& p2d) const{
        p2d.x() = p3d.x()/p3d.z() * fx + cx;
        p2d.y() = p3d.y()/p3d.z() * fy + cy;
    }

    float fx,fy,cx,cy;
    float baseline;
    int image_width,image_height;
    float k1,k2,p1,p2;//畸变矫正
    float inv_k11,inv_k22,inv_k13,inv_k23;//用于反投影
    std::string camera_name;
};


inline std::shared_ptr<PinHoleCamera> cam0;
inline std::shared_ptr<PinHoleCamera> cam1;


void InitCamera(const std::string& config_path);

}

#endif //DYNAMIC_VINS_CAMERA_MODEL_H
