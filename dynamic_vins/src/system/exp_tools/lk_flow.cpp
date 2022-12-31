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


inline bool InBorder(const cv::Point2f &pt, int row, int col)
{
    constexpr int kBorderSize = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return kBorderSize <= img_x && img_x < col - kBorderSize && kBorderSize <= img_y && img_y < row - kBorderSize;
}

inline float PointDistance(const cv::Point2f& pt1, const cv::Point2f& pt2)
{
    float dx = pt1.x - pt2.x;
    float dy = pt1.y - pt2.y;
    return std::sqrt(dx * dx + dy * dy);
}

template<typename T>
void ReduceVector(std::vector<T> &v, std::vector<uchar> status){
    int j = 0;
    for (int i = 0; i < (int)v.size(); i++){
        if (status[i])
            v[j++] = v[i];
    }
    v.resize(j);
}


cv::Mat DrawTrack(const cv::Mat &imLeft,
                               vector<cv::Point2f> &curPts,
                               vector<cv::Point2f> &lastPts){
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    cv::Mat imTrack = imLeft.clone();

    for(auto &p:curPts){
        cv::circle(imTrack, p, 4, cv::Scalar(255, 255, 0),-1);
        cv::circle(imTrack, p, 2, cv::Scalar(255, 200, 0),-1);
        cv::circle(imTrack, p, 1, cv::Scalar(255, 0, 0),-1);
    }

    for (size_t i = 0; i < curPts.size(); i++){
        cv::arrowedLine(imTrack, lastPts[i], curPts[i], cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
    }

    return imTrack;

}



/**
 * LK光流估计
 * @param img1
 * @param img2
 * @param pts1
 * @param pts2
 * @return
 */
std::tuple<std::vector<uchar>,std::vector<uchar>>
FeatureTrackByLK(const cv::Mat &img1, const cv::Mat &img2, vector<cv::Point2f> &pts1, vector<cv::Point2f> &pts2,bool is_flow_back)
{
    std::vector<uchar> status;
    std::vector<float> err;
    if(img1.empty() || img2.empty() || pts1.empty()){
        throw std::runtime_error("FeatureTrackByLK() input wrong, received at least one of parameter are empty");
    }
    //前向光流计算
    cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2,status, err,
                             cv::Size(21, 21), 3);

    //反向光流计算 判断之前光流跟踪的特征点的质量
    vector<uchar> reverse_status;
    if(is_flow_back){
        std::vector<cv::Point2f> reverse_pts = pts1;
        cv::calcOpticalFlowPyrLK(img2, img1, pts2, reverse_pts,
                                 reverse_status, err, cv::Size(21, 21), 1,
                                 cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        for(size_t i = 0; i < reverse_status.size(); i++){
            if(reverse_status[i] && PointDistance(pts1[i], reverse_pts[i]) <= 0.5)
                reverse_status[i] = 1;
            else
                reverse_status[i] = 0;
        }
    }

    ///将落在图像外面的特征点的状态删除
    for (size_t i = 0; i < pts2.size(); ++i){
        if (status[i] && !InBorder(pts2[i], img2.rows, img2.cols))
            status[i] = 0;
    }

    return {status,reverse_status};
}




int main(int argc, char **argv)
{

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

    int feat_size=200;
    int min_dist=20;
    vector<cv::Point_<float>> n_pts;
    cv::goodFeaturesToTrack(img_gray0, n_pts, feat_size, 0.01, min_dist, cv::Mat());

    vector<cv::Point2f> pts2;
    auto [status,reverse_status] = FeatureTrackByLK(img_gray0, img_gray1, n_pts, pts2,true);

    ReduceVector(n_pts, status);
    ReduceVector(pts2, status);
    ReduceVector(reverse_status, status);

    //cv::Mat img_show = DrawTrack(img_color1,n_pts,pts2);

    //int rows = imLeft.rows;
    int cols = img_color1.cols;
    cv::Mat img_show = img_color1.clone();

    for(auto &p:pts2){
        cv::circle(img_show, p, 4, cv::Scalar(255, 255, 0),-1);
        cv::circle(img_show, p, 2, cv::Scalar(255, 200, 0),-1);
        cv::circle(img_show, p, 1, cv::Scalar(255, 0, 0),-1);
    }

    for (size_t i = 0; i < pts2.size(); i++){
        if(reverse_status[i])
            cv::arrowedLine(img_show, n_pts[i], pts2[i], cv::Scalar(0, 255, 0), 2, 8, 0, 0.2);
        else
            cv::arrowedLine(img_show, n_pts[i], pts2[i], cv::Scalar(0, 0, 255), 2, 8, 0, 0.2);
    }


    cv::imwrite("/home/chen/img_flow_back.png",img_show);

    ///可视化
    cv::imshow("img_show",img_show);
    cv::waitKey(0);

    return 0;
}






