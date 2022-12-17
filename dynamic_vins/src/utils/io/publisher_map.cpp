/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "publisher_map.h"
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>

#include "estimator/body.h"


namespace dynamic_vins{\

using visualization_msgs::MarkerArray;
using visualization_msgs::Marker;


PublisherMap::PublisherMap(ros::NodeHandle &n){
    nh = &n;
}


void PublisherMap::PubImage(cv::Mat &img,const string &topic){

    if(pub_map.find(topic)==pub_map.end()){
        ros::Publisher pub = nh->advertise<sensor_msgs::Image>(topic,10);
        pub_map.insert({topic,pub});
    }

    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(body.frame_time);
    sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", img).toImageMsg();

    pub_map[topic].publish(imgTrackMsg);
}


void PublisherMap::PubPointCloud(sensor_msgs::PointCloud &cloud,const string &topic){
    if(pub_map.find(topic)==pub_map.end()){
        ros::Publisher pub = nh->advertise<sensor_msgs::PointCloud>(topic,10);
        pub_map.insert({topic,pub});
    }

    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(body.frame_time);

    cloud.header = header;

    pub_map[topic].publish(cloud);

}


void PublisherMap::PubPointCloud(PointCloud &cloud,const string &topic){
    if(pub_map.find(topic)==pub_map.end()){
        ros::Publisher pub = nh->advertise<sensor_msgs::PointCloud2>(topic,10);
        pub_map.insert({topic,pub});
    }

    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(body.frame_time);

    sensor_msgs::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(cloud,point_cloud_msg);
    point_cloud_msg.header = header;

    pub_map[topic].publish(point_cloud_msg);
}


void PublisherMap::PubMarkers(MarkerArray &markers,const string &topic){

    if(pub_map.find(topic)==pub_map.end()){
        ros::Publisher pub = nh->advertise<MarkerArray>(topic,1000);
        pub_map.insert({topic,pub});
    }

    pub_map[topic].publish(markers);
}




}


