//
// Created by chen on 2021/9/20.
//

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

using namespace std;



int main(int argc, char **argv)
{
    ros::init(argc, argv, "write_rosbag_node");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);



    return 0;
}