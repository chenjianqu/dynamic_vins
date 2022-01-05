/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <fstream>

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/PointCloud2.h>

#include <eigen3/Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>

#include "camera_pose_visualization.h"
#include "estimator/estimator.h"
#include "parameters.h"

namespace dynamic_vins{\


extern ros::Publisher pub_odometry;
extern ros::Publisher pub_path, pub_pose;
extern ros::Publisher pub_cloud, pub_map;
extern ros::Publisher pub_key_poses;
extern ros::Publisher pub_ref_pose, pub_cur_pose;
extern ros::Publisher pub_key;
extern nav_msgs::Path path;
extern ros::Publisher pub_pose_graph;
extern int IMAGE_ROW, IMAGE_COL;

void registerPub(ros::NodeHandle &n);

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t);

void PubTrackImage(const cv::Mat &imgTrack, const double t);

void printStatistics(const Estimator &estimator, double t);

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header);

void pubInitialGuess(const Estimator &estimator, const std_msgs::Header &header);

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header);

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header);

void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header);

void pubTF(const Estimator &estimator, const std_msgs::Header &header);

void pubKeyframe(const Estimator &estimator);

void pubRelocalization(const Estimator &estimator);

void pubCar(const Estimator & estimator, const std_msgs::Header &header);

void pubInstancePointCloud( Estimator &estimator, const std_msgs::Header &header);
void printInstanceData(const Estimator &estimator);

void printInstancePose(Instance &inst);
void printInstanceDepth(Instance &inst);


visualization_msgs::Marker BuildTextMarker(const PointT &point,int id,const std::string &text,const cv::Scalar &color,const double scale);
visualization_msgs::Marker BuildTextMarker(const Eigen::Vector3d &point,int id,const std::string &text,const cv::Scalar &color,const double scale=0.7);

visualization_msgs::Marker BuildLineStripMarker(PointT &maxPt,PointT &minPt,int id,const cv::Scalar &color);
visualization_msgs::Marker BuildLineStripMarker(geometry_msgs::Point p[8],int id,const cv::Scalar &color);
visualization_msgs::Marker BuildLineStripMarker(EigenContainer<Eigen::Vector3d> &p,int id,const cv::Scalar &color);


visualization_msgs::Marker BuildArrowMarker(const Eigen::Vector3d &start_pt,const Eigen::Vector3d &end_pt,int id,const cv::Scalar &color);

}