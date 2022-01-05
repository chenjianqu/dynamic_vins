/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_FEATURE_UTILS_H
#define DYNAMIC_VINS_FEATURE_UTILS_H

#include <string>
#include <vector>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <camodocal/camera_models/CameraFactory.h>
#include <camodocal/camera_models/CataCamera.h>
#include <camodocal/camera_models/PinholeCamera.h>

#include "parameters.h"
#include "utils.h"

namespace dynamic_vins{\

inline float distance(const cv::Point2f& pt1, const cv::Point2f& pt2)
{
    float dx = pt1.x - pt2.x;
    float dy = pt1.y - pt2.y;
    return std::sqrt(dx * dx + dy * dy);
}

inline bool InBorder(const cv::Point2f &pt, int row, int col)
{
    constexpr int kBorderSize = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return kBorderSize <= img_x && img_x < col - kBorderSize && kBorderSize <= img_y && img_y < row - kBorderSize;
}


/**
 * 将gpu mat转换为point2f
 * @param d_mat
 * @param vec
 */
inline void GpuMat2Points(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec){
    std::vector<cv::Point2f> points(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_32FC2, (void*)&points[0]);
    d_mat.download(mat);
    vec = points;
}

inline void GpuMat2Status(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec){
    std::vector<uchar> points(d_mat.cols);
    cv::Mat mat(1, d_mat.cols, CV_8UC1, (void*)&points[0]);
    d_mat.download(mat);
    vec=points;
}

/**
 * 将point2f转换为gpu mat
 * @param vec
 * @param d_mat
 */
inline void Points2GpuMat(const std::vector<cv::Point2f>& vec, cv::cuda::GpuMat& d_mat){
    cv::Mat mat(1, vec.size(), CV_32FC2, (void*)&vec[0]);
    d_mat=cv::cuda::GpuMat(mat);
}

inline void Status2GpuMat(const std::vector<uchar>& vec, cv::cuda::GpuMat& d_mat){
    cv::Mat mat(1, vec.size(), CV_8UC1, (void*)&vec[0]);
    d_mat=cv::cuda::GpuMat(mat);
}

void SuperpositionMask(cv::Mat &mask1, const cv::Mat &mask2);

/**
 * 对mask进行10x10的腐蚀运算
 * @param in
 * @param out
 */
inline void Erode10Gpu(cv::cuda::GpuMat &in,cv::cuda::GpuMat &out){
    static auto erode_kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(10,10),cv::Point(-1,-1));
    static auto erode_filter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE,CV_8UC1,erode_kernel);
    erode_filter->apply(in,out);
}

inline void SetMask(const cv::Mat &init_mask, cv::Mat &mask_out, std::vector<cv::Point2f> &points){
    mask_out = init_mask.clone();
    for(const auto& pt : points){
        cv::circle(mask_out, pt, cfg::kMinDist, 0, -1);
    }
}

inline void SetStatusByMask(vector<uchar> &status,vector<cv::Point2f> &points,cv::Mat &mask){
    for(size_t i=0;i<status.size();++i) if(status[i] && mask.at<uchar>(points[i]) == 0) status[i]=0;
}


std::vector<cv::Point2f> UndistortedPts(std::vector<cv::Point2f> &pts, camodocal::CameraPtr cam);

std::vector<uchar> FeatureTrackByLK(const cv::Mat &img1, const cv::Mat &img2,
                                    std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2);

std::vector<uchar> FeatureTrackByLKGpu(const cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>& lkOpticalFlow,
                                       const cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>& lkOpticalFlowBack,
                                       const cv::cuda::GpuMat &img_prev,const cv::cuda::GpuMat &img_next,
                                       std::vector<cv::Point2f> &pts_prev,std::vector<cv::Point2f> &pts_next);

std::vector<uchar> FeatureTrackByDenseFlow(cv::Mat &flow,
                                           std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2);

/**
 * 使用GPU检测Shi Tomasi角点
 * @param detect_num
 * @param img
 * @param mask
 * @return
 */
static std::vector<cv::Point2f> DetectShiTomasiCornersGpu(int detect_num, const cv::cuda::GpuMat &img, const cv::cuda::GpuMat &mask)
{
    auto detector = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, detect_num, 0.01, cfg::kMinDist);
    cv::cuda::GpuMat d_new_pts;
    detector->detect(img,d_new_pts,mask);
    std::vector<cv::Point2f> points;
    GpuMat2Points(d_new_pts, points);
    return points;
}

/**
 * 检测Shi Tomasi角点
 * @param detect_num
 * @param img
 * @param mask
 * @return
 */
inline std::vector<cv::Point2f> DetectShiTomasiCorners(int detect_num, const cv::Mat &img, const cv::Mat &mask)
{
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(img, points, detect_num, 0.01, cfg::kMinDist, mask);
    return points;
}

/**
 * 输出mask中规则点，规则点是指像素(10,15)、(10+delta,15+delta)...这样的点
 * @param detect_num 检测的数量
 * @param inst_mask mask,二值图像，0处表示背景，1处表示该实例
 * @param rect 在矩形框内检测点
 * @param curr_pts 已有的点
 * @return
 */
std::vector<cv::Point2f> DetectRegularCorners(int detect_num, const cv::Mat &inst_mask,
                                              std::vector<cv::Point2f> &curr_pts, cv::Rect rect=cv::Rect());


}

#endif //DYNAMIC_VINS_FEATURE_UTILS_H
