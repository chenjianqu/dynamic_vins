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


using namespace std;

/**
 * 通过设置不同的距离来检测角点
 */
void SimpleDetection(){
    string img_path_0="/home/chen/datasets/VIODE/cam0/day_03_high/1597198415.157930_cam0.png";

    cv::Mat img_color0 = cv::imread(img_path_0);

    cv::Mat img_gray0,img_gray1;
    cv::cvtColor(img_color0,img_gray0,cv::COLOR_BGR2GRAY);

    int feat_size=200;
    int min_dist=20;
    vector<cv::Point_<float>> n_pts;
    cv::goodFeaturesToTrack(img_gray0, n_pts, feat_size, 0.01, min_dist, cv::Mat());

    cout<<n_pts.size()<<endl;

    int cross_len=6;
    cv::Mat img_show0 = img_color0.clone();
    for(auto &p:n_pts){
        cv::circle(img_show0, p, 3, cv::Scalar(255, 255, 0),-1);
        cv::circle(img_show0, p, 2, cv::Scalar(255, 200, 0),-1);
        cv::circle(img_show0, p, 1, cv::Scalar(255, 0, 0),-1);
    }

    cv::imwrite("/home/chen/img_det.png",img_show0);

    ///可视化
    //cv::imshow("img0",img_show0);

    //cv::waitKey(0);
}


/**
 * 在已存在的角点的情况下，检测新的角点
 */
void RepeatDetection(){
    //string img_path_0="/home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/0003/000038.png";
    string img_path_0="/home/chen/datasets/VIODE/cam0/day_03_high/1597198425.603095_cam0.png";

    cv::Mat img_color0 = cv::imread(img_path_0);

    cv::Mat img_gray0,img_gray1;
    cv::cvtColor(img_color0,img_gray0,cv::COLOR_BGR2GRAY);

    int feat_size=100;
    int min_dist=20;
    vector<cv::Point_<float>> n_pts;
    cv::goodFeaturesToTrack(img_gray0, n_pts, feat_size, 0.01, min_dist, cv::Mat());

    cv::Mat img_show0 = img_color0.clone();
    for(auto &p:n_pts){
        cv::circle(img_show0, p, 3, cv::Scalar(255, 0, 0),-1);
    }

    cv::Mat mask(cv::Size(img_gray0.cols,img_gray0.rows),CV_8UC1,cv::Scalar(255));
    for(auto &p:n_pts){
        cv::circle(mask, p, 20, cv::Scalar(0),-1);
    }

    vector<cv::Point_<float>> n_pts2;
    cv::goodFeaturesToTrack(img_gray0, n_pts2, feat_size, 0.01, min_dist, mask);

    cv::Mat mask_bgr;
    cv::cvtColor(mask,mask_bgr,CV_GRAY2BGR);
    cv::scaleAdd(mask_bgr,0.5,img_show0,img_show0);

    for(auto &p:n_pts2){
        cv::circle(img_show0, p, 3, cv::Scalar(0, 0, 255),-1);
    }


    ///可视化
    cv::imshow("img0",img_show0);
    cv::imshow("mask",mask);
    cv::waitKey(0);

    cv::imwrite("/home/chen/img0.png",img_show0);
    cv::imwrite("/home/chen/mask.png",mask);

}



int main(int argc, char **argv)
{
    //SimpleDetection();

    RepeatDetection();

    return 0;
}



