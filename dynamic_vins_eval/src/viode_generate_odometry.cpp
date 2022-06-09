#include<string>
#include<set>
#include<vector>
#include <exception>
#include <iostream>
#include <fstream>

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>


using namespace std;

ofstream outfile;

void OdometryCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
    cout<<msg->header<<endl;
    cout<<msg->pose.pose<<endl;
    cout<<msg->twist.twist<<endl;
    geometry_msgs::Pose p=msg->pose.pose;
    geometry_msgs::Twist v=msg->twist.twist;


        /*
        outfile.precision(0);
        outfile<<msg->header.stamp.toSec() * 1e9<<",";
        outfile.precision(5);
        outfile<<p.position.x<<","<<
            p.position.y<<","<<
            p.position.z<<","<<
            p.orientation.w<<","<<
            p.orientation.x<<","<<
            p.orientation.y<<","<<
            p.orientation.z<<","<<
            v.linear.x<<","<<
            v.linear.y<<","<<
            v.linear.z<<","<<
            v.angular.x<<","<<
            v.angular.y<<","<<
            v.angular.z<<","<<
            0.0<<","<<
            0.0<<","<<
            0.0<<endl;
            */
        outfile  <<msg->header.stamp << " "
        <<p.position.x<<" "
        <<p.position.y<<" "
        <<p.position.z<<" "
        <<p.orientation.x<<" "
        <<p.orientation.y<<" "
        <<p.orientation.z<<" "
        <<p.orientation.w<<endl;
}


int main(int argc, char** argv)
{
    if(argc != 3){
        cerr<<"parameters number wrong!,usage: rosrun dynamic_vins_eval viode_generate_odometry TOPIC filename.txt"<<endl;
        return 1;
    }

    string file_name=argv[2];
    string topic_name=argv[1];

    outfile.open(file_name.c_str(),ios::trunc);
    if(!outfile.is_open()){
        cerr<<"can not open:"<<file_name<<endl;
        return 1;
    }


	setlocale(LC_ALL, "");//防止中文乱码
	ros::init(argc, argv, "viode_generate_odometry");
	ros::start();
	ros::NodeHandle nh;

    outfile.setf(ios::fixed, ios::floatfield);

    ROS_INFO("启动Save节点");

    ros::Subscriber textSub = nh.subscribe(topic_name, 5, OdometryCallback);

	ros::Rate loop_rate(10);
	ros::spin();

	outfile.close();

	ROS_INFO("主节点结束");

	return 0;
}






