/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_IO_UTILS_H
#define DYNAMIC_VINS_IO_UTILS_H

#include <filesystem>

#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Point32.h>
#include <tf/transform_broadcaster.h>


#include "utils/def.h"



namespace dynamic_vins{\

void ClearDirectory(const string &path);

vector<fs::path> GetDirectoryFileNames(const string &path);

bool CheckIsDir(const string &dir);

void GetAllImageFiles(const string& dir, vector<string> &files) ;

inline geometry_msgs::Point EigenToGeometryPoint(const Vec3d &vec_point){
    geometry_msgs::Point point;
    point.x = vec_point.x();
    point.y = vec_point.y();
    point.z = vec_point.z();
    return point;
}

inline geometry_msgs::Point32 EigenToGeometryPoint32(const Vec3d &vec_point){
    geometry_msgs::Point32 point;
    point.x =(float) vec_point.x();
    point.y =(float) vec_point.y();
    point.z =(float) vec_point.z();
    return point;
}

inline geometry_msgs::Vector3 EigenToGeometryVector3(const Vec3d &vec_point){
    geometry_msgs::Vector3 point;
    point.x =(float) vec_point.x();
    point.y =(float) vec_point.y();
    point.z =(float) vec_point.z();
    return point;
}


inline geometry_msgs::Quaternion EigenToGeometryQuaternion(const Eigen::Quaterniond &q_eigen){
    geometry_msgs::Quaternion q;
    q.x = q_eigen.x();
    q.y = q_eigen.y();
    q.z = q_eigen.z();
    q.w = q_eigen.w();
    return q;
}


inline tf::Vector3 EigenToTfVector(const Vec3d &v_eigen){
    return {v_eigen.x(),v_eigen.y(),v_eigen.z()};
}




inline tf::Quaternion EigenToTfQuaternion(const Eigen::Quaterniond &q_eigen){
    tf::Quaternion q;
    q.setW(q_eigen.w());
    q.setX(q_eigen.x());
    q.setY(q_eigen.y());
    q.setZ(q_eigen.z());
    return q;
}


inline std_msgs::ColorRGBA ScalarBgrToColorRGBA(const cv::Scalar &color){
    std_msgs::ColorRGBA color_rgba;
    color_rgba.r = color[2];
    color_rgba.g = color[1];
    color_rgba.b = color[0];
    color_rgba.a = color[3];
    if(color_rgba.a==0){
        color_rgba.a=1.;
    }
    return color_rgba;
}

cv::Scalar GenerateNormBgrColor(const string &color_str);


}


#endif //DYNAMIC_VINS_IO_UTILS_H
