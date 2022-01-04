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

void GpuMat2Points(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec);
void GpuMat2Status(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec);

void Points2GpuMat(const std::vector<cv::Point2f>& vec, cv::cuda::GpuMat& d_mat);
void Status2GpuMat(const std::vector<uchar>& vec, cv::cuda::GpuMat& d_mat);



std::vector<uchar> FlowTrack(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::Point2f> &pts1, std::vector<cv::Point2f> &pts2);

std::vector<uchar> FlowTrackGpu(const cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>& lkOpticalFlow,
                                const cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>& lkOpticalFlowBack,
                                const cv::cuda::GpuMat &img_prev,
                                const cv::cuda::GpuMat &img_next, std::vector<cv::Point2f> &pts_prev, std::vector<cv::Point2f> &pts_next);

std::vector<cv::Point2f> DetectNewFeaturesGPU(int detect_num, const cv::cuda::GpuMat &img, const cv::cuda::GpuMat &mask);

inline std::vector<cv::Point2f> DetectNewFeaturesCPU(int detect_num, const cv::Mat &img, const cv::Mat &mask)
{
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(img, points, detect_num, 0.01, Config::kMinDist, mask);
    return points;
}

void SuperpositionMask(cv::Mat &mask1, const cv::Mat &mask2);


void SetMask(const cv::Mat &init_mask, cv::Mat &mask_out, std::vector<cv::Point2f> &points);
void SetMaskGpu(const cv::cuda::GpuMat &init_mask, cv::cuda::GpuMat &mask_out, std::vector<cv::Point2f> &points);


std::vector<cv::Point2f> UndistortedPts(std::vector<cv::Point2f> &pts, camodocal::CameraPtr cam);


#endif //DYNAMIC_VINS_FEATURE_UTILS_H
