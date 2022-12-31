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
#include <random>

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
                             cv::Size(21, 21), 1);

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

inline void ErodeMask(cv::Mat &in, cv::Mat &out,int kernel_size=5){
    auto erode_kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                                  cv::Size(kernel_size,kernel_size),cv::Point(-1,-1));
    cv::erode(in,out,erode_kernel);
}

tuple<cv::Mat,cv::Mat> InstanceImagePadding(cv::Mat &img1,cv::Mat &img2){
    int rows = std::max(img1.rows,img2.rows);
    int cols = std::max(img1.cols,img2.cols);
    cv::Mat img1_padded,img2_padded;
    cv::copyMakeBorder(img1,img1_padded,0,rows-img1.rows,0,cols-img1.cols,cv::BORDER_CONSTANT,cv::Scalar(0));
    cv::copyMakeBorder(img2,img2_padded,0,rows-img2.rows,0,cols-img2.cols,cv::BORDER_CONSTANT,cv::Scalar(0));
    return {img1_padded,img2_padded};
}


std::vector<cv::Point2f> DetectRegularCorners(int detect_num, const cv::Mat &inst_mask,int step ,cv::Rect rect=cv::Rect()){
    int cnt=0;
    int row_start=0,row_end=inst_mask.rows,col_start=0,col_end=inst_mask.cols;
    if(!rect.empty()){
        row_start = rect.tl().y;
        col_start = rect.tl().x;
        row_end = rect.br().y;
        col_end = rect.br().x;
    }
    int half_step = std::ceil(step/2.);
    row_start += half_step;
    row_end -= half_step;
    col_start += half_step;
    col_end -= half_step;

    vector<cv::Point2f> vec;
    for(int i=row_start;i<row_end;i+=step){
        for(int j=col_start;j<col_end;j+=step){
            if(inst_mask.at<uchar>(i,j) > 0.5){
                vec.emplace_back(j,i);
                cnt++;
            }
        }
    }

    std::shuffle(vec.begin(),vec.end(),
                 std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

    vector<cv::Point2f> vec_out(vec.begin(),vec.begin()+std::min(detect_num,(int)vec.size()));
    return vec_out;
}


/**
 * 角点检测
 */
void DetectPointsTest(){
    string img_path_0="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/36_5_gray_roi.png";
    string img_path_1="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/37_5_gray_roi.png";

    string mask_path_0="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/36_5_mask_roi.png";
    string mask_path_1="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/37_5_mask_roi.png";

    cv::Mat img_gray0 = cv::imread(img_path_0,0);
    cv::Mat img_gray1 = cv::imread(img_path_1,0);

    cv::Mat mask0 = cv::imread(mask_path_0,0);
    cv::Mat mask1 = cv::imread(mask_path_1,0);

    ErodeMask(mask0,mask0,5);
    ErodeMask(mask1,mask1,5);

    int feat_size=100;
    int min_dist=8;
    vector<cv::Point_<float>> n_pts;
    cv::goodFeaturesToTrack(img_gray0, n_pts, feat_size, 0.01, min_dist, mask0);
    cout<<"n_pts.size() "<<n_pts.size()<<endl;

    ///对两张图像进行padding，使得两张图像大小一致
    auto [img0_padded,img1_padded] = InstanceImagePadding(img_gray0,img_gray1);

    cv::Mat img_show;
    cv::cvtColor(img_gray0,img_show,CV_GRAY2BGR);

    for(int i=0;i<n_pts.size();++i){
        cv::circle(mask0,n_pts[i],2,cv::Scalar(0),-1);

        cv::circle(img_show,n_pts[i],1,cv::Scalar(255,0,0),-1);
    }


    ///检测额外点
    //int detect_num = std::max(50,100-(int)n_pts.size());
    int detect_num = 50;
    auto extra_points = DetectRegularCorners(detect_num,mask0,4);

    for(int i=0;i<extra_points.size();++i){
        //cv::circle(mask0,extra_points[i],2,cv::Scalar(0),-1);

        cv::circle(img_show,extra_points[i],1,cv::Scalar(0,255,0),-1);
    }

    cout<<"extra_points.size() "<<extra_points.size()<<endl;


    cv::imwrite("/home/chen/det_points.png",img_show);
    cv::imwrite("/home/chen/det_points_mask.png",mask0);


}



void RoiVisualization(){
    string img_path_0="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/37_5_gray.png";
    string mask_path_0="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/37_5_mask.png";

    cv::Mat img_gray0 = cv::imread(img_path_0,0);
    cv::Mat mask0 = cv::imread(mask_path_0,0);

    cv::Mat img_color0;
    cv::cvtColor(img_gray0,img_color0,CV_GRAY2BGR);

    cv::Mat mask_bgr(mask0.rows,mask0.cols,CV_8UC3,cv::Scalar(0,0,0));

    int row_max=0,row_min=mask0.rows;
    int col_max=0,col_min=mask0.cols;

    int rows = mask0.rows;
    int cols = mask0.cols;
    for(int i=0;i<rows;++i){
        for(int j=0;j<cols;++j){
            if(mask0.at<uchar>(i,j)>0.5){
                mask_bgr.at<cv::Vec3b>(i,j)[0]=255;
                mask_bgr.at<cv::Vec3b>(i,j)[1]=255;

                if(i>row_max){
                    row_max=i;
                }
                else if(i<row_min){
                    row_min=i;
                }
                if(j>col_max){
                    col_max=j;
                }
                else if(j<col_min){
                    col_min=j;
                }
            }
        }
    }

    cv::Rect rect(cv::Point(col_min,row_min),cv::Point(col_max,row_max));

    cv::Mat img_show;
    cv::scaleAdd(mask_bgr,1,img_color0,img_show);

    cv::rectangle(img_show,rect,cv::Scalar(0,255,0),2);

    cv::imwrite("/home/chen/inst_mask_show.png",img_show);

}



/**
 * 裁切光流跟踪
 */
void RoiTest(){

    string img_path_0="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/9_2_gray.png";
    string img_path_1="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/10_2_gray.png";

    string mask_path_0="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/9_2_mask.png";
    string mask_path_1="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/10_2_mask.png";

    cv::Mat img_gray0 = cv::imread(img_path_0,0);
    cv::Mat img_gray1 = cv::imread(img_path_1,0);

    cv::Mat mask0 = cv::imread(mask_path_0,0);
    cv::Mat mask1 = cv::imread(mask_path_1,0);

    cout<<"img_gray0.size:"<<img_gray0.size<<endl;
    cout<<"img_gray1.size:"<<img_gray1.size<<endl;

    ErodeMask(mask0,mask0,5);
    ErodeMask(mask1,mask1,5);

    int feat_size=200;
    int min_dist=8;
    vector<cv::Point_<float>> n_pts;
    cv::goodFeaturesToTrack(img_gray0, n_pts, feat_size, 0.01, min_dist, mask0);
    cout<<"n_pts.size() "<<n_pts.size()<<endl;

    cv::Mat img_show_0;
    cv::cvtColor(img_gray0,img_show_0,CV_GRAY2BGR);
    for(int i=0;i<n_pts.size();++i){
        cv::Point p_last=n_pts[i];
        cv::circle(img_show_0, p_last, 2, cv::Scalar(0, 0, 255),-1);
    }
    cv::imwrite("/home/chen/img_show_0.png",img_show_0);

    ///对两张图像进行padding，使得两张图像大小一致
    auto [img0_padded,img1_padded] = InstanceImagePadding(img_gray0,img_gray1);

    cv::imwrite("/home/chen/img0_padded.png",img0_padded);
    cv::imwrite("/home/chen/img1_padded.png",img1_padded);


    vector<cv::Point2f> pts2;
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
    for(int i=0;i<pts2.size();++i){
        cv::Point p_last=n_pts[i];
        cv::circle(img_show_padded, p_last, 2, cv::Scalar(0, 0, 255),-1);

        cv::Point p=pts2[i];
        cv::circle(img_show_padded, cv::Point(p.x,p.y+rows_offset), 2, cv::Scalar(255, 0, 0),-1);

        cv::arrowedLine(img_show_padded, p_last, cv::Point(p.x,p.y+rows_offset),cv::Scalar(0, 255, 0), 1, 8, 0, 0.1);
    }
    cv::imwrite("/home/chen/inst_lk.png",img_show_padded);

    /*    for (size_t i = 0; i < pts2.size(); i++){
            if(reverse_status[i])
                cv::arrowedLine(img_show, n_pts[i], pts2[i], cv::Scalar(0, 255, 0), 2, 8, 0, 0.2);
            else
                cv::arrowedLine(img_show, n_pts[i], pts2[i], cv::Scalar(0, 0, 255), 2, 8, 0, 0.2);
        }*/

    cv::imwrite("/home/chen/img_flow_back.png",img_show);

    ///可视化
    //cv::imshow("img_show",img_show);
    //cv::waitKey(0);
}


/**
 * 完整图片光流跟踪
 */
void FullTest(){

    string img_path_0="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/36_5_gray.png";
    string img_path_1="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/37_5_gray.png";

    string mask_path_0="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/36_5_mask.png";
    string mask_path_1="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/37_5_mask.png";

    cv::Mat img_gray0 = cv::imread(img_path_0,0);
    cv::Mat img_gray1 = cv::imread(img_path_1,0);

    cv::Mat mask0 = cv::imread(mask_path_0,0);
    cv::Mat mask1 = cv::imread(mask_path_1,0);

    ErodeMask(mask0,mask0,5);
    ErodeMask(mask1,mask1,5);

    int feat_size=200;
    int min_dist=8;
    vector<cv::Point_<float>> n_pts_0;
    cv::goodFeaturesToTrack(img_gray0, n_pts_0, feat_size, 0.01, min_dist, mask0);
    cout<<"n_pts0.size() "<<n_pts_0.size()<<endl;

    vector<cv::Point_<float>> n_pts_1;
    //cv::goodFeaturesToTrack(img_gray1, n_pts_1, feat_size, 0.01, min_dist, mask1);
    cout<<"n_pts1.size() "<<n_pts_1.size()<<endl;

    cv::Mat img_show_padded;
    cv::vconcat(img_gray0, img_gray1, img_show_padded);
    cv::cvtColor(img_show_padded,img_show_padded,CV_GRAY2BGR);

    int rows_offset = img_gray0.rows;

    for(int i=0;i<n_pts_0.size();++i){
        cv::circle(img_show_padded, n_pts_0[i], 3, cv::Scalar(0, 0, 255),-1);
    }

    auto [status,reverse_status] = FeatureTrackByLK(img_gray0, img_gray1, n_pts_0, n_pts_1,true);

    //cv::imshow("img0_padded",img0_padded);
    //cv::waitKey(0);

    ///将落在图像外面的特征点的状态删除
    for (size_t i = 0; i < n_pts_1.size(); ++i){
        if (status[i] && !InBorder(n_pts_1[i], img_gray1.rows, img_gray1.cols))
            status[i] = 0;

        if(status[i] && mask1.at<uchar>(n_pts_1[i])<0.5){
            status[i] = 0;
        }
    }

    ReduceVector(n_pts_0, status);
    ReduceVector(n_pts_1, status);
    ReduceVector(reverse_status, status);

    cout<<"n_pts_1.size() "<<n_pts_1.size()<<endl;


    for(int i=0;i<n_pts_1.size();++i){
        cv::Point p_last=n_pts_0[i];
        cv::circle(img_show_padded, p_last, 3, cv::Scalar(0, 0, 255),-1);

        cv::Point p=n_pts_1[i];
        cv::circle(img_show_padded, cv::Point(p.x,p.y+rows_offset), 3, cv::Scalar(255, 0, 0),-1);

        cv::arrowedLine(img_show_padded, p_last, cv::Point(p.x,p.y+rows_offset),cv::Scalar(0, 255, 0), 1, 8, 0,
                        0.02);
    }
    cv::imwrite("/home/chen/inst_lk.png",img_show_padded);


}



int main(int argc, char **argv)
{

    //FullTest();

    //DetectPointsTest();

    RoiTest();

    //RoiVisualization();

    return 0;
}






