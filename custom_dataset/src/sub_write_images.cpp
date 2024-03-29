/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#include "io_utils.h"

using namespace std;

const string img_stereo_topic="/zed_cam/cam0";

string kBaseDIR;

const int kWidth = 1280;
const int kHeight = 720;


void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr= cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
    ///序号
    int seq = img_msg->header.seq;
    string pad_number = PadNumber(seq,6);
    cv::imwrite(kBaseDIR + "/" + pad_number+".png",ptr->image);
    cout<<kBaseDIR + "/" + pad_number+".png"<<endl;
}

int main(int argc,char **argv)
{
    if(argc!=2){
        cerr<<"usage:rosrun custom_dataset sub_write_images [seq_name]";
        return -1;
    }

    ros::init(argc, argv, "sub_write_image_node");

    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    kBaseDIR = argv[1];

    ///创建文件夹
    ClearDirectory(kBaseDIR);

    ros::Subscriber sub_img0 = n.subscribe(img_stereo_topic, 1000, img0_callback);

    cout<<"start wait image"<<endl;

    ros::spin();

    return 0;
}