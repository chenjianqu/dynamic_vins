/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_UTILS_H
#define DYNAMIC_VINS_UTILS_H

#include <string>
#include <vector>
#include <chrono>

#include <spdlog/logger.h>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "parameters.h"




class TicToc{
public:
    TicToc(){
        tic();
    }

    void tic(){
        start = std::chrono::system_clock::now();
    }

    double toc(){
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000;
    }

    double toc_then_tic(){
        auto t=toc();
        tic();
        return t;
    }

    void toc_print_tic(const char* str){
        cout<<str<<":"<<toc()<<" ms"<<endl;
        tic();
    }

private:
    std::chrono::time_point<std::chrono::system_clock> start, end;
};



struct InstInfo{
    std::string name;
    int label_id;
    int id;
    int track_id;
    cv::Point2f min_pt,max_pt;
    cv::Rect2f rect;
    float prob;

    cv::Point2f mask_center;

    cv::Mat mask_cv;
    cv::cuda::GpuMat mask_gpu;
    torch::Tensor mask_tensor;
};



struct SegImage{
    cv::Mat color0,seg0,color1,seg1;
    cv::cuda::GpuMat color0_gpu,color1_gpu;
    double time0,seg0_time,time1,seg1_time;
    cv::Mat gray0,gray1;
    cv::cuda::GpuMat gray0_gpu,gray1_gpu;

    torch::Tensor mask_tensor;
    std::vector<InstInfo> insts_info;

    cv::Mat merge_mask,inv_merge_mask;
    cv::cuda::GpuMat merge_mask_gpu,inv_merge_mask_gpu;

    void SetMask();
    void SetMaskGpu();
    void SetMaskGpuSimple();

    void SetGrayImage();
    void SetGrayImageGpu();
    void SetColorImage();
    void SetColorImageGpu();
};



struct ImageInfo{
    int origin_h,origin_w;
    ///图像的裁切信息
    int rect_x, rect_y, rect_w, rect_h;
};




template <typename T>
static std::string DimsToStr(torch::ArrayRef<T> list){
    int i = 0;
    std::string text= "[";
    for(auto e : list) {
        if (i++ > 0) text+= ", ";
        text += std::to_string(e);
    }
    text += "]";
    return text;
}


inline cv::Point2f operator*(const cv::Point2f &lp,const cv::Point2f &rp)
{
    return {lp.x * rp.x,lp.y * rp.y};
}

template<typename MatrixType>
inline std::string EigenToStr(const MatrixType &m){
    std::string text;
    for(int i=0;i<m.rows();++i){
        for(int j=0;j<m.cols();++j){
            text+=fmt::format("{:.2f} ",m(i,j));
        }
        if(m.rows()>1)
            text+="\n";
    }
    return text;
}

template<typename T>
inline std::string VecToStr(const Eigen::Matrix<T,3,1> &vec){
    return EigenToStr(vec.transpose());
}

template<typename T>
inline std::string QuaternionToStr(const Eigen::Quaternion<T> &q){
    return fmt::format("x:{:.2f} y:{:.2f} z:{:.2f} w:{:.2f}",q.x(),q.y(),q.z(),q.w());
}


void DrawText(cv::Mat &img, const std::string &str, const cv::Scalar &color, const cv::Point& pos, float scale= 1.f, int thickness= 1, bool reverse = false);

void DrawBbox(cv::Mat &img, const cv::Rect2f& bbox, const std::string &label = "", const cv::Scalar &color = {0, 0, 0});



float CalBoxIoU(const cv::Point2f &box1_minPt, const cv::Point2f &box1_maxPt,
                const cv::Point2f &box2_minPt, const cv::Point2f &box2_maxPt);

float CalBoxIoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt);

cv::Scalar color_map(int64_t n);


template <typename Arg1, typename... Args>
inline void DebugV(const char* fmt, const Arg1 &arg1, const Args&... args){ vio_logger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void DebugV(const T& msg){vio_logger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void InfoV(const char* fmt, const Arg1 &arg1, const Args&... args){vio_logger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void InfoV(const T& msg){vio_logger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void WarnV(const char* fmt, const Arg1 &arg1, const Args&... args){vio_logger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void WarnV(const T& msg){vio_logger->log(spdlog::level::warn, msg);}


template <typename Arg1, typename... Args>
inline void DebugS(const char* fmt, const Arg1 &arg1, const Args&... args){ sg_logger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void DebugS(const T& msg){sg_logger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void InfoS(const char* fmt, const Arg1 &arg1, const Args&... args){sg_logger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void InfoS(const T& msg){sg_logger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void WarnS(const char* fmt, const Arg1 &arg1, const Args&... args){sg_logger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void WarnS(const T& msg){sg_logger->log(spdlog::level::warn, msg);}


template <typename Arg1, typename... Args>
inline void DebugT(const char* fmt, const Arg1 &arg1, const Args&... args){ tk_logger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void DebugT(const T& msg){tk_logger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void InfoT(const char* fmt, const Arg1 &arg1, const Args&... args){tk_logger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void InfoT(const T& msg){tk_logger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void WarnT(const char* fmt, const Arg1 &arg1, const Args&... args){tk_logger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void WarnT(const T& msg){tk_logger->log(spdlog::level::warn, msg);}


#endif //DYNAMIC_VINS_UTILS_H
