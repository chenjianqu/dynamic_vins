#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <mutex>
#include <list>
#include <thread>
#include <regex>

#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "oxts_parser.h"


using namespace std;

visualization_msgs::Marker
BuildLineStripMarker(const Eigen::Vector3d& point0,const Eigen::Vector3d& point1)
{
    visualization_msgs::Marker msg;
    msg.header.frame_id="map";
    msg.header.stamp=ros::Time::now();
    msg.ns="box_strip";
    msg.action=visualization_msgs::Marker::ADD;
    msg.pose.orientation.w=1.0;

    //暂时使用类别代替这个ID
    msg.id=0;//当存在多个marker时用于标志出来
    //cout<<msg.id<<endl;
    msg.lifetime=ros::Duration(4);//持续时间3s，若为ros::Duration()表示一直持续

    msg.type=visualization_msgs::Marker::LINE_STRIP;//marker的类型
    msg.scale.x=0.01;//线宽
    msg.color.r=1.0;msg.color.g=1.0;msg.color.b=1.0;
    msg.color.a=1.0;//不透明度

    //设置立方体的八个顶点
    geometry_msgs::Point minPt,maxPt;
    minPt.x=0;minPt.y=0;minPt.z=0;
    maxPt.x=1;maxPt.y=1;maxPt.z=1;
    geometry_msgs::Point p[8];
    p[0].x=minPt.x;p[0].y=minPt.y;p[0].z=minPt.z;
    p[1].x=maxPt.x;p[1].y=minPt.y;p[1].z=minPt.z;
    p[2].x=maxPt.x;p[2].y=minPt.y;p[2].z=maxPt.z;
    p[3].x=minPt.x;p[3].y=minPt.y;p[3].z=maxPt.z;
    p[4].x=minPt.x;p[4].y=maxPt.y;p[4].z=maxPt.z;
    p[5].x=maxPt.x;p[5].y=maxPt.y;p[5].z=maxPt.z;
    p[6].x=maxPt.x;p[6].y=maxPt.y;p[6].z=minPt.z;
    p[7].x=minPt.x;p[7].y=maxPt.y;p[7].z=minPt.z;

    //这个类型仅将相邻点进行连线
    for(auto &pt : p)
        msg.points.push_back(pt);
    //为了保证矩形框的其它边存在：
    msg.points.push_back(p[0]);

    return msg;
}


void PubOxts(ros::NodeHandle &nh,const string &data_path)
{
    vector<Eigen::Matrix4d> pose ;
    ParseOxts(pose,data_path);

    int kPubDeltaTime=1000; //发布时间间隔,默认100ms

    ros::Publisher obj_pub=nh.advertise<visualization_msgs::MarkerArray>(
            "marker_test_topic",10);

    int index=0;
    double time=0.;

    while(ros::ok()){

        visualization_msgs::MarkerArray markers;

        visualization_msgs::Marker msg;
        msg.header.frame_id="map";
        msg.header.stamp=ros::Time::now();
        msg.ns="box_strip";
        msg.action=visualization_msgs::Marker::ADD;
        msg.pose.orientation.w=1.0;

        //暂时使用类别代替这个ID
        msg.id=0;//当存在多个marker时用于标志出来
        msg.lifetime=ros::Duration(10);//持续时间3s，若为ros::Duration()表示一直持续

        msg.type=visualization_msgs::Marker::LINE_STRIP;//marker的类型
        msg.scale.x=0.5;//线宽
        msg.color.r=1.0;msg.color.g=0.0;msg.color.b=1.0;
        msg.color.a=1.0;//不透明度

        for(int i=1;i<pose.size();++i){
            geometry_msgs::Point p;
            p.x = pose[i](0,3);
            p.y = pose[i](1,3);
            p.z = pose[i](2,3);
            msg.points.push_back(p);
        };

        markers.markers.push_back(msg);

        obj_pub.publish(markers);
        ros::spinOnce();


        std::this_thread::sleep_for(std::chrono::milliseconds(kPubDeltaTime));

        index++;
        cout<<index<<endl;
    }
}


int main(int argc, char** argv)
{
    setlocale(LC_ALL, "");//防止中文乱码
    ros::init(argc, argv, "oxts_parser");
    ros::start();

    ros::NodeHandle nh;

    string data_path="/home/chen/CLionProjects/CV_Tools/cv_ws/src/kitti_pub/data/oxts/0007.txt";
    if(argc==2){
        data_path = argv[1];
    }

    PubOxts(nh,data_path);

    return 0;
}