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

#include "utils/orb/ORBextractor.h"


using namespace std;
using namespace cv;


inline bool InBorder(const cv::Point2f &pt, int row, int col)
{
    constexpr int kBorderSize = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return kBorderSize <= img_x && img_x < col - kBorderSize && kBorderSize <= img_y && img_y < row - kBorderSize;
}

inline float distance(const cv::Point2f& pt1, const cv::Point2f& pt2)
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



int OrbDest(const cv::Mat &img_1,const cv::Mat &img_2) {

    //-- 初始化
    std::vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    //-- 第一步:检测 Oriented FAST 角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    cout<<"keypoints_1.size:"<<keypoints_1.size()<<endl;
    cout<<"keypoints_2.size:"<<keypoints_2.size()<<endl;

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

    Mat outimg1;
    drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    imshow("ORB features", outimg1);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> matches;
    t1 = chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

    //-- 第四步:匹配点对筛选
    // 计算最小距离和最大距离
    auto min_max = minmax_element(matches.begin(), matches.end(),
                                  [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    std::vector<DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    //-- 第五步:绘制匹配结果
    Mat img_match;
    Mat img_goodmatch;
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
    imshow("all matches", img_match);
    imshow("good matches", img_goodmatch);
    waitKey(0);

    return 0;
}


tuple<cv::Mat,cv::Mat> InstanceImagePadding(cv::Mat &img1,cv::Mat &img2){
    int rows = std::max(img1.rows,img2.rows);
    int cols = std::max(img1.cols,img2.cols);
    cv::Mat img1_padded,img2_padded;
    cv::copyMakeBorder(img1,img1_padded,0,rows-img1.rows,0,cols-img1.cols,cv::BORDER_CONSTANT,cv::Scalar(0));
    cv::copyMakeBorder(img2,img2_padded,0,rows-img2.rows,0,cols-img2.cols,cv::BORDER_CONSTANT,cv::Scalar(0));
    return {img1_padded,img2_padded};
}



int main(int argc, char **argv)
{
    //string img_path_0="/home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/0003/000038.png";

    string img_path_0="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/36_5_gray.png";
    string img_path_1="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/37_5_gray.png";

    string mask_path_0="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/36_5_mask.png";
    string mask_path_1="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/37_5_mask.png";

    cv::Mat img_gray0 = cv::imread(img_path_0,0);
    cv::Mat img_gray1 = cv::imread(img_path_1,0);

    cv::Mat mask0 = cv::imread(mask_path_0,0);
    cv::Mat mask1 = cv::imread(mask_path_1,0);


    /*ORBextractor* mpORBextractorLeft = new ORBextractor(
            100,//抽取100个特征
            1.2,
            1,//尺度金字塔的数量
            10,
            5);*/

    std::vector<cv::KeyPoint> n_pts;
    cv::Mat descriptors;

/*    (*mpORBextractorLeft)(img_gray0,				//待提取特征点的图像
            cv::Mat(),		//掩摸图像, 实际没有用到
            n_pts,			//输出变量，用于保存提取后的特征点
            descriptors);	//输出变量，用于保存特征点的描述子*/



    //cout<<n_pts.size()<<endl;

    ///对两张图像进行padding，使得两张图像大小一致
    auto [img0_padded,img1_padded] = InstanceImagePadding(img_gray0,img_gray1);

    OrbDest(img0_padded,img1_padded);

    /*    vector<cv::Point2f> pts2;
        auto [status,reverse_status] = FeatureTrackByLK(img0_padded, img1_padded, n_pts, pts2,false);

        //cv::imshow("img0_padded",img0_padded);
        //cv::waitKey(0);

        ///将落在图像外面的特征点的状态删除
        for (size_t i = 0; i < pts2.size(); ++i){
            if (status[i] && !InBorder(pts2[i], img_gray1.rows, img_gray1.cols))
                status[i] = 0;

            if(status[i] && mask1.at<uchar>(pts2[i])<0.5){
                status[i] = 0;
            }
        }

        ReduceVector(n_pts, status);
        ReduceVector(pts2, status);
        ReduceVector(reverse_status, status);

            cout<<"pts2.size() "<<pts2.size()<<endl;

        */


    //cv::Mat img_show = DrawTrack(img_color1,n_pts,pts2);

    cv::Mat img_show;
    cv::cvtColor(img_gray1,img_show,CV_GRAY2BGR);

    /*    for(int i=0;i<pts2.size();++i){
        cv::Point p=pts2[i];
        //cv::circle(img_show, p, 4, cv::Scalar(255, 255, 0),-1);
        //cv::circle(img_show, p, 2, cv::Scalar(255, 200, 0),-1);
        cv::circle(img_show, p, 2, cv::Scalar(255, 0, 0),-1);

        cv::Point p_last=n_pts[i];
        cv::circle(img_show, p_last, 2, cv::Scalar(0, 0, 255),-1);

    }*/

    cv::Mat img_show_padded;
    cv::vconcat(img0_padded, img1_padded, img_show_padded);
    cv::cvtColor(img_show_padded,img_show_padded,CV_GRAY2BGR);
    int rows_offset = img0_padded.rows;

    for(int i=0;i<n_pts.size();++i){
        cv::KeyPoint p_last=n_pts[i];
        cv::circle(img_show_padded, p_last.pt, 2, cv::Scalar(0, 0, 255),-1);
    }



/*    for(int i=0;i<n_pts.size();++i){
        cv::Point p_last=n_pts[i];
        cv::circle(img_show_padded, p_last, 2, cv::Scalar(0, 0, 255),-1);

        cv::Point p=pts2[i];
        cv::circle(img_show_padded, cv::Point(p.x,p.y+rows_offset), 2, cv::Scalar(255, 0, 0),-1);

        //cv::arrowedLine(img_show_padded, p_last, cv::Point(p.x,p.y+rows_offset),
        //                cv::Scalar(0, 255, 0), 1, 8, 0, 0.1);
    }*/
    cv::imwrite("/home/chen/inst_orb.png",img_show_padded);



    return 0;
}






