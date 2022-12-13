/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_datatypes.h>

#include "io_utils.h"

using namespace std;




int main(int argc,char **argv)
{
    if(argc!=2){
        cerr<<"usage:rosrun custom_dataset sub_write_images [seq_name]";
        return -1;
    }

    ros::init(argc, argv, "sub_write_image_node");

    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);


    return 0;
}