/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <memory>
#include <thread>
#include <mutex>
#include <chrono>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "utils/log_utils.h"
#include "line_detector/line.h"
#include "line_detector/line_detector.h"
#include "line_detector/line_geometry.h"

using namespace std;


namespace dynamic_vins{\


LineDetector::Ptr line_detector;

FrameLines::Ptr curr_lines,prev_lines;


/**
 * 直线检测和跟踪
 * @param gray0
 * @param gray1
 */
void TrackLine(cv::Mat gray0, cv::Mat mask){
    Debugt("TrackLine | start");

    ///线特征的提取和跟踪
    curr_lines = line_detector->Detect(gray0,mask);
    Debugt("TrackLine | detect new lines size:{}",curr_lines->keylsd.size());

    line_detector->TrackLeftLine(prev_lines, curr_lines);
    Debugt("TrackLine | track lines size:{}",curr_lines->keylsd.size());

    curr_lines->SetLines();
    //curr_lines->UndistortedLineEndPoints(cam_t.cam0);

    Debugt("TrackLine | finished");
}


int Run(int argc, char **argv)
{
    if(argc!=2){
        cerr<<"please input right parameters"<<endl;
        return -1;
    }

    string file_name = argv[1];
    cout<<fmt::format("cfg_file:{}",argv[1])<<endl;

    ///初始化logger
    MyLogger::InitLogger(file_name);

    line_detector = std::make_shared<LineDetector>(file_name);


    //string img_path_0="/home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/0002/000008.png";
    //string img_path_1="/home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/0002/000009.png";

    //string img_path_0="/home/chen/datasets/VIODE/cam0/day_03_high/1597198401.366509_cam0.png";
    //string img_path_1="/home/chen/datasets/VIODE/cam0/day_03_high/1597198401.566331_cam0.png";

    string img_path_0="/home/chen/datasets/VIODE/cam0/day_03_high/1597198429.601021_cam0.png";
    string img_path_1="/home/chen/datasets/VIODE/cam0/day_03_high/1597198429.800692_cam0.png";

    cv::Mat img_color0 = cv::imread(img_path_0);
    cv::Mat img_color1 = cv::imread(img_path_1);

    cv::Mat img_gray0;
    cv::cvtColor(img_color0,img_gray0,cv::COLOR_BGR2GRAY);
    cv::Mat img_gray1;
    cv::cvtColor(img_color1,img_gray1,cv::COLOR_BGR2GRAY);

    ///检测第一张图像
    TrackLine(img_gray0,cv::Mat());

    cv::Mat img_vis=img_color0.clone();
    line_detector->VisualizeLine(img_vis, curr_lines);


    ///可视化
    cv::imshow("img_show",img_vis);
    cv::waitKey(0);

    prev_lines = curr_lines;

    ///检测第二张图像
    TrackLine(img_gray1,cv::Mat());


    img_vis=img_color1.clone();
    line_detector->VisualizeLine(img_vis, curr_lines);
    if(prev_lines)
        line_detector->VisualizeLineMonoMatch(img_vis, prev_lines, curr_lines);

    cout<<prev_lines->keylsd.size()<<endl;
    cout<<curr_lines->keylsd.size()<<endl;


    ///可视化
    cv::imshow("img_show",img_vis);
    cv::waitKey(0);

    cv::imwrite("/home/chen/img_line.png",img_vis);

    return 0;
}

}



int main(int argc, char **argv)
{
    return dynamic_vins::Run(argc,argv);
}






