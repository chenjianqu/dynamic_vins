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
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include "io_utils.h"

using namespace std;
using namespace cv;



const string img_stereo_topic="/zed/zed_node/stereo/image_rect_color";

const string kBaseDIR="/home/chen/datasets/MyData/ZED";

const int kWidth = 1280;
const int kHeight = 720;

string seq_name;

string dataset_path,left_image_path,right_image_path;
string time_file_path;

int counter=0;

cv::Rect rect0,rect1;


int WriteUVC(int argc, char** argv)
{

    cv:: VideoCapture cap = cv::VideoCapture(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, kWidth*2);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, kHeight);
    cv::Mat frame;
    cap.read(frame);
    cout<<frame.size<<endl;

    while(cap.isOpened()){
        auto start = std::chrono::steady_clock::now();

        cap.read(frame);

        cv::Mat img0=frame(rect0);
        cv::Mat img1=frame(rect1);

        if(frame.empty()){
            std::cout << "Read frame failed!" << std::endl;
            break;
        }

        if(argc==3){
            ///时间
            std::ostringstream oss;
            oss << std::fixed <<std::setprecision(6) << ros::Time::now().toSec();
            string time_str = oss.str();
            ///序号
            string pad_number = PadNumber(counter,6);
            ///写入时间和序号
            WriteTextFile(time_file_path,pad_number+" "+time_str);

            ///保存图像
            cv::imwrite(left_image_path+pad_number+".png",img0);
            cv::imwrite(right_image_path+pad_number+".png",img1);
        }
        else{
            cv::imshow("Test", frame);
            if(cv::waitKey(1)== 27) break;
        }

        cout<<counter<<" "<<std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count()<<"ms"<<endl;

        counter++;
    }
}


int main(int argc, char** argv) {

    if(argc!=2){
        cerr<<"usage:rosrun custom_dataset sub_write_images [seq_name]";
        return -1;
    }

    ros::init(argc, argv, "read_stereo_node");

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

    return WriteUVC(argc,argv);
}
