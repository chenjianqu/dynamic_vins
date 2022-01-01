//
// Created by chen on 2021/11/19.
//

#ifndef DYNAMIC_VINS_SEGMENTIMAGE_H
#define DYNAMIC_VINS_SEGMENTIMAGE_H

#include <string>
#include <vector>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <camodocal/camera_models/CameraFactory.h>
#include <camodocal/camera_models/CataCamera.h>
#include <camodocal/camera_models/PinholeCamera.h>

#include "../parameters.h"
#include "../utils.h"


inline float distance(const cv::Point2f& pt1, const cv::Point2f& pt2)
{
    float dx = pt1.x - pt2.x;
    float dy = pt1.y - pt2.y;
    return std::sqrt(dx * dx + dy * dy);
}

inline bool inBorder(const cv::Point2f &pt,int row,int col)
{
    constexpr int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

void gpuMat2Points(const cv::cuda::GpuMat& d_mat, std::vector<cv::Point2f>& vec);
void gpuMat2Status(const cv::cuda::GpuMat& d_mat, std::vector<uchar>& vec);

void points2GpuMat(const std::vector<cv::Point2f>& vec, cv::cuda::GpuMat& d_mat);
void status2GpuMat(const std::vector<uchar>& vec, cv::cuda::GpuMat& d_mat);



std::vector<uchar> flowTrack(const cv::Mat &img1,const cv::Mat &img2,std::vector<cv::Point2f> &pts1,std::vector<cv::Point2f> &pts2);

std::vector<uchar> flowTrackGpu(const cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>& lkOpticalFlow,
                                const cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>& lkOpticalFlowBack,
                                const cv::cuda::GpuMat &img_prev,
                                const cv::cuda::GpuMat &img_next,std::vector<cv::Point2f> &pts_prev,std::vector<cv::Point2f> &pts_next);

std::vector<cv::Point2f> detectNewFeaturesGPU(int detect_num,const cv::cuda::GpuMat &img,const cv::cuda::GpuMat &mask);
inline std::vector<cv::Point2f> detectNewFeaturesCPU(int detect_num,const cv::Mat &img,const cv::Mat &mask)
{
    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(img, points, detect_num, 0.01, Config::MIN_DIST, mask);
    return points;
}

void superpositionMask(cv::Mat &mask1, const cv::Mat &mask2);


void setMask(const cv::Mat &init_mask,cv::Mat &mask_out,std::vector<cv::Point2f> &points);
void setMaskGpu(const cv::cuda::GpuMat &init_mask,cv::cuda::GpuMat &mask_out,std::vector<cv::Point2f> &points);


std::vector<cv::Point2f> undistortedPts(std::vector<cv::Point2f> &pts, camodocal::CameraPtr cam);


#endif //DYNAMIC_VINS_SEGMENTIMAGE_H
