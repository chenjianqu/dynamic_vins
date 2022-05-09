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
#include "parameters.h"
#include "box3d.h"

namespace dynamic_vins{\

class Publisher{
public:
    static void registerPub(ros::NodeHandle &n);

    static  void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t);

    static   void PubTrackImage(const cv::Mat &imgTrack, double t);

    static   void printStatistics(const Estimator &estimator, double t);

    static   void pubOdometry(const Estimator &estimator, const std_msgs::Header &header);

    static   void pubInitialGuess(const Estimator &estimator, const std_msgs::Header &header);

    static   void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header);

    static   void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header);

    static   void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header);

    static   void pubTF(const Estimator &estimator, const std_msgs::Header &header);

    static     void pubKeyframe(const Estimator &estimator);

    static    void pubRelocalization(const Estimator &estimator);

    static   void pubCar(const Estimator & estimator, const std_msgs::Header &header);

    static void pubInstancePointCloud( Estimator &estimator, const std_msgs::Header &header);

    static   void printInstanceData(const Estimator &estimator);

    static  void printInstancePose(Instance &inst);

    static void printInstanceDepth(Instance &inst);

    static void PubPredictBox3D(const Estimator & estimator,std::vector<Box3D> &boxes);

    static  visualization_msgs::Marker BuildTextMarker(const PointT &point,int id,const std::string &text,const cv::Scalar &color,double scale);

    static  visualization_msgs::Marker BuildTextMarker(const Eigen::Vector3d &point,int id,const std::string &text,const cv::Scalar &color,double scale=1.);

    static visualization_msgs::Marker BuildLineStripMarker(PointT &maxPt,PointT &minPt,int id,const cv::Scalar &color);

    static  visualization_msgs::Marker BuildLineStripMarker(geometry_msgs::Point p[8],int id,const cv::Scalar &color);

    static   visualization_msgs::Marker BuildLineStripMarker(EigenContainer<Eigen::Vector3d> &p,int id,const cv::Scalar &color);

    static visualization_msgs::Marker BuildCubeMarker(Eigen::Matrix<double,8,3> &corners,int id);

    static  visualization_msgs::Marker BuildArrowMarker(const Eigen::Vector3d &start_pt,const Eigen::Vector3d &end_pt,int id,const cv::Scalar &color);

};




}

