/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_VIO_UTIL_H
#define DYNAMIC_VINS_VIO_UTIL_H

#include <optional>

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <sophus/so3.hpp>

#include "utils/def.h"
#include "utils/box3d.h"
#include "landmark.h"
#include "line_landmark.h"

namespace dynamic_vins{\


template<typename T>
inline Mat3d hat(T && v){
    return Sophus::SO3d::hat(std::forward<T>(v)); //完美转发
}



void TriangulatePoint(Mat34d &Pose0, Mat34d &Pose1, Vec2d &point0, Vec2d &point1, Vec3d &point_3d);

Vec3d TriangulatePoint(const Mat34d &Pose0, const Mat34d &Pose1, const Vec2d &point0, const Vec2d &point1);


void TriangulateDynamicPoint(Mat34d &Pose0, Mat34d &Pose1,
                             Vec2d &point0, Vec2d &point1, Vec3d &v, Vec3d &a,
                             double delta_t, Vec3d &point_3d);

void TriangulateDynamicPoint(const Mat34d &Pose0, const Mat34d &Pose1,
                             const Vec2d &point0, const  Vec2d &point1,
                             const Mat3d &R_woj, const Vec3d &P_woj,
                             const Mat3d &R_woi, const Vec3d &P_woi,
                             Vec3d &point_3d);


std::optional<Vec3d> FitBox3DFromPoints(vector<Vec3d> &points,const Vec3d& dims);

std::optional<Vec3d> FitBox3DFromCameraFrame(vector<Vec3d> &points,const Vec3d& dims);

std::optional<Vec3d> FitBox3DSimple(vector<Vec3d> &points,const Vec3d& dims);

std::optional<Vec3d> FitBox3DWithRANSAC(vector<Vec3d> &points,const Vec3d& dims);


void OutliersRejection(std::set<int> &removeIndex,std::list<StaticLandmark>& point_landmarks);


double ReprojectionError(Mat3d &Ri, Vec3d &Pi, Mat3d &rici, Vec3d &tici,
                         Mat3d &Rj, Vec3d &Pj, Mat3d &ricj, Vec3d &ticj,
                         double depth, Vec3d &uvi, Vec3d &uvj);


void TriangulateOneLine(LineLandmark &line);

void TriangulateOneLineStereo(LineLandmark &line);


bool SolvePoseByPnP(Mat3d &R_initial, Vec3d &P_initial,
                           vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);

double CompensatedParallax2(const StaticLandmark &landmark, int frame_count);


}

#endif //DYNAMIC_VINS_VIO_UTIL_H
