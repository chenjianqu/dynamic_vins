/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <fstream>

#include <eigen3/Eigen/Dense>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>

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

#include "camera_pose_visualization.h"
#include "estimator/estimator.h"
#include "utils/parameters.h"
#include "basic/box3d.h"
#include "io_parameters.h"

namespace dynamic_vins{\

using namespace visualization_msgs;

class Publisher{
public:

    static void RegisterPub(ros::NodeHandle &n);

    static void PubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t);

    static void PrintStatistics(double t);

    static void PubOdometry(const std_msgs::Header &header);

    static void PubKeyPoses(const std_msgs::Header &header);

    static void PubCameraPose(const std_msgs::Header &header);

    static void PubPointCloud(const std_msgs::Header &header);

    static void PubLines(const std_msgs::Header &header);

    static void PubTF(const std_msgs::Header &header);

    static void PubKeyframe();

    static void PubInstancePointCloud(const std_msgs::Header &header);

    static void PubInstances(const std_msgs::Header &header);

    static void PubPredictBox3D(const std_msgs::Header &header);

    static void PubGroundTruthBox3D(const std_msgs::Header &header);

    static void PubTransform(const Mat3d &R,const Vec3d &P,tf::TransformBroadcaster &br,ros::Time time,
                             const string &frame_id,const string &child_frame_id);

    static void PubSceneVec(const std_msgs::Header &header);

    inline static std::shared_ptr<Estimator> e;
};





}

