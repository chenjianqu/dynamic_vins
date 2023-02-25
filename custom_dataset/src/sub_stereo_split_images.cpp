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

const string img_stereo_topic="/zed/zed_node/stereo/image_rect_color";

const string kBaseDIR="/home/chen/datasets/MyData/ZED";

const int kWidth = 960;
const int kHeight = 540;

string seq_name;

string dataset_path,left_image_path,right_image_path;
string time_file_path;

int counter=0;

cv::Rect rect0,rect1;


void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr= cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);

    ///时间
    std::ostringstream oss;
    oss << std::fixed <<std::setprecision(6) << img_msg->header.stamp.toSec();
    string time_str = oss.str();
    ///序号
    string pad_number = PadNumber(counter,6);
    ///写入时间和序号
    WriteTextFile(time_file_path,pad_number+" "+time_str);

    ///保存图像
    cv::Mat img0 = ptr->image(rect0);
    cv::Mat img1 = ptr->image(rect1);

    cv::imwrite(left_image_path+pad_number+".png",img0);
    cv::imwrite(right_image_path+pad_number+".png",img1);

    cout<<counter<<endl;

    counter++;
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

    seq_name = argv[1];

    dataset_path = kBaseDIR + "/" + seq_name + "/";
    left_image_path = dataset_path+"cam0/";
    right_image_path = dataset_path+"cam1/";

    rect0 = cv::Rect(0,0,kWidth,kHeight);
    rect1 = cv::Rect(kWidth,0,kWidth,kHeight);

    ///创建文件夹
    ClearDirectory(dataset_path);
    ClearDirectory(left_image_path);
    ClearDirectory(right_image_path);

    time_file_path = dataset_path+"time.txt";

    ros::Subscriber sub_img0 = n.subscribe(img_stereo_topic, 1000, img0_callback);

    cout<<"start wait image"<<endl;

    ros::spin();

    return 0;
}