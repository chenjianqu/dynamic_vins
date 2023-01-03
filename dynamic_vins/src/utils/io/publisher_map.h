/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_PUBLISHER_MAP_H
#define DYNAMIC_VINS_PUBLISHER_MAP_H

#include <pcl/point_cloud.h>
#include <pcl/common/common.h>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/MarkerArray.h>

#include "basic/def.h"

namespace dynamic_vins{\

/**
 * ROS的消息发布类
 */
class PublisherMap{
public:
    using Ptr=std::shared_ptr<PublisherMap>;

    explicit PublisherMap(ros::NodeHandle &n);

    static void PubImage(cv::Mat &img,const string &topic);

    static void PubPointCloud(pcl::PointCloud<pcl::PointXYZRGB> &cloud,const string &topic);

    static void PubPointCloud(sensor_msgs::PointCloud &cloud,const string &topic);

    static void PubMarkers(visualization_msgs::MarkerArray &markers,const string &topic);

    /**
     * 发布一条消息
     * @tparam T
     * @param msg
     * @param topic
     */
    template<typename T>
    static void Pub(T msg,const string &topic){
        if(pub_map.find(topic)==pub_map.end()){
            ros::Publisher pub = nh->advertise<T>(topic,1000);
            pub_map.insert({topic,pub});
        }

        pub_map[topic].publish(msg);
    }


    /**
     * 获取一个发布器
     * @tparam T
     * @param topic
     * @return
     */
    template<typename T>
    static ros::Publisher GetPublisher(const string &topic){
        if(pub_map.find(topic)==pub_map.end()){
            ros::Publisher pub = nh->advertise<T>(topic,1000);
            pub_map.insert({topic,pub});
        }
        return pub_map[topic];
    }


private:
    inline static ros::NodeHandle *nh;

    inline static std::unordered_map<string,ros::Publisher> pub_map;

};




}

#endif //DYNAMIC_VINS_PUBLISHER_MAP_H
