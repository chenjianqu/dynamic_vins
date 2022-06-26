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
#include "utils/box3d.h"
#include "io_parameters.h"

namespace dynamic_vins{\

using namespace visualization_msgs;

class Publisher{
public:

    static void RegisterPub(ros::NodeHandle &n);

    static  void PubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t);

    static   void PrintStatistics(double t);

    static   void PubOdometry(const std_msgs::Header &header);

    static   void pubInitialGuess(const std_msgs::Header &header);

    static   void PubKeyPoses(const std_msgs::Header &header);

    static   void PubCameraPose(const std_msgs::Header &header);

    static   void PubPointCloud(const std_msgs::Header &header);

    static   void PubTF(const std_msgs::Header &header);

    static     void PubKeyframe();

    static void PubInstances(const std_msgs::Header &header);

    static void PubPredictBox3D(std::vector<Box3D> &boxes);

    static void PubTransform(const Mat3d &R,const Vec3d &P,tf::TransformBroadcaster &br,ros::Time time,
                      const string &frame_id,const string &child_frame_id);

    static Marker BuildTrajectoryMarker(unsigned int id, std::list<State> &history, State* sliding_window, const cv::Scalar &color,
                                        Marker::_action_type action=Marker::ADD, const string &ns="trajectory" ,int offset=4);


    inline static std::shared_ptr<Estimator> e;
};


class ImagePublisher{
public:
    using Ptr=std::shared_ptr<ImagePublisher>;

    explicit ImagePublisher(ros::NodeHandle &n);

    static void Pub(cv::Mat &img,const string &topic);


    inline static ros::NodeHandle *nh;
    inline static std::unordered_map<string,ros::Publisher> pub_map;

};


class PointCloudPublisher{
public:
    using Ptr=std::shared_ptr<PointCloudPublisher>;

    explicit PointCloudPublisher(ros::NodeHandle &n);

    static void Pub(PointCloud &cloud,const string &topic);
    static void Pub(sensor_msgs::PointCloud &cloud,const string &topic);

    inline static ros::NodeHandle *nh;
    inline static std::unordered_map<string,ros::Publisher> pub_map;

};



}

