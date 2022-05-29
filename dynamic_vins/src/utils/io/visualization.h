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

    static  void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t);

    static   void PubTrackImage(const cv::Mat &imgTrack, double t);

    static   void printStatistics(double t);

    static   void pubOdometry(const std_msgs::Header &header);

    static   void pubInitialGuess(const std_msgs::Header &header);

    static   void pubKeyPoses(const std_msgs::Header &header);

    static   void pubCameraPose( const std_msgs::Header &header);

    static   void pubPointCloud( const std_msgs::Header &header);

    static   void pubTF( const std_msgs::Header &header);

    static     void pubKeyframe();

    static    void pubRelocalization();

    static   void pubCar(const std_msgs::Header &header);

    static void pubInstancePointCloud( const std_msgs::Header &header);

    static   void printInstanceData();

    static  void printInstancePose(Instance &inst);

    static void printInstanceDepth(Instance &inst);

    static void PubPredictBox3D(std::vector<Box3D> &boxes);

    static  Marker BuildTextMarker(const PointT &point,unsigned int id,const std::string &text,const cv::Scalar &color,
                                   double scale,Marker::_action_type action=Marker::ADD);

    static  Marker BuildTextMarker(const Eigen::Vector3d &point,unsigned int id,const std::string &text,const cv::Scalar &color,
                                   double scale=1.,Marker::_action_type action=Marker::ADD);

    static Marker BuildLineStripMarker(PointT &maxPt,PointT &minPt,unsigned int id,const cv::Scalar &color,
                                       Marker::_action_type action=Marker::ADD);

    static  Marker BuildLineStripMarker(geometry_msgs::Point p[8],unsigned int id,const cv::Scalar &color,
                                                            Marker::_action_type action=Marker::ADD);

    static   Marker BuildLineStripMarker(EigenContainer<Eigen::Vector3d> &p,unsigned int id,const cv::Scalar &color,
                                         Marker::_action_type action=Marker::ADD);

    static Marker BuildCubeMarker(Eigen::Matrix<double,8,3> &corners,unsigned int id,Marker::_action_type action=Marker::ADD);

    static  Marker BuildArrowMarker(const Eigen::Vector3d &start_pt,const Eigen::Vector3d &end_pt,unsigned int id,
                                    const cv::Scalar &color,Marker::_action_type action=Marker::ADD);

    static Marker BuildTrajectoryMarker(unsigned int id,std::list<State> &history,State* sliding_window,const cv::Scalar &color,
                                        bool clear=false,Marker::_action_type action=Marker::ADD);

    static void PubTransform(const Mat3d &R,const Vec3d &P,tf::TransformBroadcaster &br,ros::Time time,
                      const string &frame_id,const string &child_frame_id);

    inline static std::shared_ptr<Estimator> e;
};


 void SaveInstanceTrajectory(unsigned int frame_id,unsigned int track_id,std::string &type,
                                   int truncated,int occluded,double alpha,Vec4d &box,
                                   Vec3d &dims,Vec3d &location,double rotation_y,double score);


}

