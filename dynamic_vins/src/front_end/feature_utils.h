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

#include "utils/def.h"
#include "semantic_image.h"
#include "utils/camera_model.h"

using std::vector;

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


template<typename T>
void ReduceVector(std::vector<T> &v, std::vector<uchar> status){
    int j = 0;
    for (int i = 0; i < (int)v.size(); i++){
        if (status[i])
            v[j++] = v[i];
    }
    v.resize(j);
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
inline void ErodeMaskGpu(cv::cuda::GpuMat &in, cv::cuda::GpuMat &out,int kernel_size=10){
    auto erode_kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                                  cv::Size(kernel_size,kernel_size),cv::Point(-1,-1));
    auto erode_filter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE,CV_8UC1,erode_kernel);
    erode_filter->apply(in,out);
}


inline void SetStatusByMask(vector<uchar> &status,vector<cv::Point2f> &points,cv::Mat &mask){
    for(size_t i=0;i<status.size();++i) if(status[i] && mask.at<uchar>(points[i]) == 0) status[i]=0;
}


std::vector<cv::Point2f> UndistortedPts(std::vector<cv::Point2f> &pts, PinHoleCamera::Ptr &cam);

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
std::vector<cv::Point2f> DetectShiTomasiCornersGpu(int detect_num, const cv::cuda::GpuMat &img, const cv::cuda::GpuMat &mask);



/**
 * 输出mask中规则点，规则点是指像素(10,15)、(10+delta,15+delta)...这样的点
 * @param detect_num 检测的数量
 * @param inst_mask mask,二值图像，0处表示背景，1处表示该实例
 * @param rect 在矩形框内检测点
 * @param curr_pts 已有的点
 * @return
 */
std::vector<cv::Point2f> DetectRegularCorners(int detect_num, const cv::Mat &inst_mask, cv::Rect rect=cv::Rect());


/**
 * 计算两个物体Mask的IoU
 * @param mask1 二元Mask1
 * @param instInfo1  物体信息1
 * @param mask1_area Mask1的大小
 * @param mask2 二元Mask2
 * @param instInfo2  物体信息2
 * @param mask2_area Mask2的大小
 * @return IoU
 */
inline float GetMaskIoU(const torch::Tensor &mask1, const InstInfo &instInfo1, const float mask1_area,
                 const torch::Tensor &mask2, const InstInfo &instInfo2, const float mask2_area){
    auto intersection_mask=(mask1 * mask2);
    float intersection_area = intersection_mask.sum(torch::IntArrayRef({0,1})).item().toFloat();
    return intersection_area/(mask1_area + mask2_area - intersection_area);
}

/**
 * 将id和特征组合在一起
 * @param ids
 * @param curr_un_pts
 * @param out_pairs
 */
inline void SetIdPointPair(vector<unsigned int> &ids,
                           vector<cv::Point2f> &curr_un_pts,
                           std::map<unsigned int, cv::Point2f> &out_pairs){
    out_pairs.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
        out_pairs.insert({ids[i], curr_un_pts[i]});
}

void PtsVelocity(double dt, vector<unsigned int> &ids, vector<cv::Point2f> &curr_un_pts,
                        std::map<unsigned int, cv::Point2f> &prev_id_pts,vector<cv::Point2f> &output_velocity);



void SortPoints(vector<cv::Point2f> &cur_pts, vector<int> &track_cnt, vector<int> &ids);



void DrawText(cv::Mat &img, const std::string &str, const cv::Scalar &color, const cv::Point& pos, float scale= 1.f, int thickness= 1, bool reverse = false);

void DrawBbox(cv::Mat &img, const cv::Rect2f& bbox, const std::string &label = "", const cv::Scalar &color = {0, 0, 0});

cv::Scalar color_map(int64_t n);


float BoxIoU(const cv::Point2f &box1_minPt, const cv::Point2f &box1_maxPt,
             const cv::Point2f &box2_minPt, const cv::Point2f &box2_maxPt);

float BoxIoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt);




inline std::string DimsToStr(cv::Size list){
    return "[" + std::to_string(list.height) + ", " + std::to_string(list.width) + "]";
}


inline cv::Point2f operator*(const cv::Point2f &lp,const cv::Point2f &rp){
    return {lp.x * rp.x,lp.y * rp.y};
}


}

#endif //DYNAMIC_VINS_FEATURE_UTILS_H
