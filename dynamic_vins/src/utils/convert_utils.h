/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/


#ifndef DYNAMIC_VINS_CONVERT_UTILS_H
#define DYNAMIC_VINS_CONVERT_UTILS_H

#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Point32.h>
#include <tf/transform_broadcaster.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "def.h"

namespace dynamic_vins{\

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


inline pcl::PointCloud<pcl::PointXYZ>::Ptr EigenToPclXYZ(const vector<Vec3d> &points){
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>);
    for(auto &pt : points){
        pc->points.emplace_back(pt.x(),pt.y(),pt.z());
    }
    return pc;
}

inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr EigenToPclXYZRGB(const vector<Vec3d> &points,const cv::Scalar &bgr_color){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    pc->points.reserve(pc->size());
    for(auto &pt : points){
        pcl::PointXYZRGB p(bgr_color[2],bgr_color[1],bgr_color[0]);
        p.x = pt.x();
        p.y = pt.y();
        p.z = pt.z();
        pc->points.emplace_back(p);
    }
    pc->height=1;
    pc->width=pc->points.size();
    return pc;
}


template<typename T>
inline vector<Vec3d> PclToEigen(typename pcl::PointCloud<T>::Ptr &pcl_points){
    vector<Vec3d> point_eigen(pcl_points->size());
    for(int i=0;i<pcl_points->size();++i){
        point_eigen[i] << pcl_points->points[i].x,pcl_points->points[i].y,pcl_points->points[i].z;
    }
    return point_eigen;
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

cv::Scalar BgrColor(const string &color_str,bool is_norm=true);


template<typename T>
string CvMatToStr(const cv::Mat &m){
    if(m.empty()){
        return {};
    }
    else if(m.channels()>1){
        return "CvMatToStr() input Mat has more than one channel";
    }
    else{
        string ans;
        for(int i=0;i<m.rows;++i){
            for(int j=0;j<m.cols;++j){
                ans += std::to_string(m.at<T>(i,j)) + " ";
            }
            ans+="\n";
        }
        return ans;
    }
}



}

#endif //DYNAMIC_VINS_CONVERT_UTILS_H
