/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/
/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>


namespace dynamic_vins{\

class CameraPoseVisualization {
public:
    using Ptr=std::shared_ptr<CameraPoseVisualization>;

    std::string m_marker_ns;

    CameraPoseVisualization(float r, float g, float b, float a);
	
    void setImageBoundaryColor(float r, float g, float b, float a=1.0);
    void setOpticalCenterConnectorColor(float r, float g, float b, float a=1.0);
    void setScale(double s);
    void setLineWidth(double width);

    void add_pose(const Eigen::Vector3d& p, const Eigen::Quaterniond& q);
    void reset();

    void publish_by(ros::Publisher& pub, const std_msgs::Header& header);
    void add_edge(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1);
    void add_loopedge(const Eigen::Vector3d& p0, const Eigen::Vector3d& p1);

private:
    std::vector<visualization_msgs::Marker> m_markers;
    std_msgs::ColorRGBA m_image_boundary_color;
    std_msgs::ColorRGBA m_optical_center_connector_color;
    double m_scale;
    double m_line_width;

    inline static const Eigen::Vector3d imlt = Eigen::Vector3d(-1.0, -0.5, 1.0);
    inline static const Eigen::Vector3d imrt = Eigen::Vector3d( 1.0, -0.5, 1.0);
    inline static const Eigen::Vector3d imlb = Eigen::Vector3d(-1.0,  0.5, 1.0);
    inline static const Eigen::Vector3d imrb = Eigen::Vector3d( 1.0,  0.5, 1.0);
    inline static const Eigen::Vector3d lt0 = Eigen::Vector3d(-0.7, -0.5, 1.0);
    inline static const Eigen::Vector3d lt1 = Eigen::Vector3d(-0.7, -0.2, 1.0);
    inline static const Eigen::Vector3d lt2 = Eigen::Vector3d(-1.0, -0.2, 1.0);
    inline static const Eigen::Vector3d oc = Eigen::Vector3d(0.0, 0.0, 0.0);
};
}