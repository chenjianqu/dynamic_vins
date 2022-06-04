/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_BUILD_MARKERS_H
#define DYNAMIC_VINS_BUILD_MARKERS_H

#include <visualization_msgs/MarkerArray.h>

#include "utils/def.h"
#include "utils/parameters.h"


namespace dynamic_vins{\

using namespace visualization_msgs;


extern int32_t kMarkerTypeNumber;


Marker BuildLineStripMarker(PointT &maxPt,PointT &minPt,unsigned int id,const cv::Scalar &color,
    Marker::_action_type action=Marker::ADD,int offset=0);

Marker BuildLineStripMarker(geometry_msgs::Point p[8],unsigned int id,const cv::Scalar &color,
                            Marker::_action_type action=Marker::ADD,int offset=0);

Marker BuildLineStripMarker(EigenContainer<Eigen::Vector3d> &p,unsigned int id,const cv::Scalar &color,
                            Marker::_action_type action=Marker::ADD,int offset=0);

Marker BuildTextMarker(const PointT &point,unsigned int id,const std::string &text,const cv::Scalar &color,
                       double scale,Marker::_action_type action=Marker::ADD,int offset=1);

Marker BuildTextMarker(const Eigen::Vector3d &point,unsigned int id,const std::string &text,const cv::Scalar &color,
                       double scale=1.,Marker::_action_type action=Marker::ADD,int offset=1);

Marker BuildArrowMarker(const Eigen::Vector3d &start_pt,const Eigen::Vector3d &end_pt,unsigned int id,
                        const cv::Scalar &color,Marker::_action_type action=Marker::ADD,int offset=2);

Marker BuildCubeMarker(Mat38d &corners,unsigned int id,const cv::Scalar &color,
                       Marker::_action_type action=Marker::ADD,int offset=3);


std::tuple<Marker,Marker,Marker> BuildAxisMarker(Mat34d &axis,unsigned int id,Marker::_action_type action=Marker::ADD,int offset=5);


}


#endif //DYNAMIC_VINS_BUILD_MARKERS_H
