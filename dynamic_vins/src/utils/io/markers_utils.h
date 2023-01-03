/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_MARKERS_UTILS_H
#define DYNAMIC_VINS_MARKERS_UTILS_H

#include <visualization_msgs/MarkerArray.h>
#include "basic/def.h"
#include "basic/state.h"
#include "utils/parameters.h"
#include "io_parameters.h"


namespace dynamic_vins{\

using namespace visualization_msgs;


extern int32_t kMarkerTypeNumber;


Marker LineStripMarker(geometry_msgs::Point p[8], unsigned int id,
                       const cv::Scalar &color, double scale= 0.1,
                       Marker::_action_type action= Marker::ADD,
                       const string &ns="cube_strip",int offset= 0);

Marker LineStripMarker(EigenContainer<Eigen::Vector3d> &p, unsigned int id,
                       const cv::Scalar &color, double scale= 0.1,
                       Marker::_action_type action= Marker::ADD,
                       const string &ns="cube_strip",int offset= 0);

Marker TextMarker(const PointT &point, unsigned int id, const std::string &text,
                  const cv::Scalar &color, double scale,
                  Marker::_action_type action= Marker::ADD,
                  const string &ns="text",int offset= 1);

Marker TextMarker(const Eigen::Vector3d &point, unsigned int id, const std::string &text,
                  const cv::Scalar &color, double scale= 1.,
                  Marker::_action_type action= Marker::ADD,
                  const string &ns="text",int offset= 1);

Marker ArrowMarker(const Eigen::Vector3d &start_pt, const Eigen::Vector3d &end_pt, unsigned int id,
                   const cv::Scalar &color, double scale= 0.1,
                   Marker::_action_type action= Marker::ADD,
                   const string &ns="arrow",int offset= 2);

Marker CubeMarker(Mat38d &corners, unsigned int id, const cv::Scalar &color,
                  double scale= 0.1, Marker::_action_type action= Marker::ADD,
                  const string &ns="cube",int offset= 3,float duration=io_para::kVisualInstDuration);


Marker LineMarker(const Eigen::Vector3d &p1,const Eigen::Vector3d &p2,unsigned int id, const cv::Scalar &color,
                  double scale=0.1, Marker::_action_type action=Marker::ADD, const string &ns="line",int offset=4);

std::tuple<Marker,Marker,Marker> AxisMarker(Mat34d &axis, unsigned int id,
                                            Marker::_action_type action= Marker::ADD,
                                            const string &ns="axis",int offset= 5);


Marker BuildTrajectoryMarker(unsigned int id, std::list<State> &history, State* sliding_window, const cv::Scalar &color,
                             Marker::_action_type action=Marker::ADD, const string &ns="trajectory" ,int offset=6);




}

#endif //DYNAMIC_VINS_MARKERS_UTILS_H
