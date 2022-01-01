//
// Created by chen on 2021/12/1.
//

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

    void setMask();
    void setMaskGpu();
    void setMaskGpuSimple();

    void setGrayImage();
    void setGrayImageGpu();
    void setColorImage();
    void setColorImageGpu();
};



struct ImageInfo{
    int originH,originW;
    ///图像的裁切信息
    int rect_x, rect_y, rect_w, rect_h;
};




template <typename T>
static std::string dims2str(torch::ArrayRef<T> list){
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
inline std::string eigen2str(const MatrixType &m){
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
inline std::string vec2str(const Eigen::Matrix<T,3,1> &vec){
    return eigen2str(vec.transpose());
}

template<typename T>
inline std::string quaternion2str(const Eigen::Quaternion<T> &q){
    return fmt::format("x:{:.2f} y:{:.2f} z:{:.2f} w:{:.2f}",q.x(),q.y(),q.z(),q.w());
}


void draw_text(cv::Mat &img, const std::string &str,const cv::Scalar &color, const cv::Point& pos,  float scale=1.f, int thickness=1,bool reverse = false);

void draw_bbox(cv::Mat &img, const cv::Rect2f& bbox,const std::string &label = "", const cv::Scalar &color = {0, 0, 0});



float getBoxIoU(const cv::Point2f &box1_minPt, const cv::Point2f &box1_maxPt,
                   const cv::Point2f &box2_minPt, const cv::Point2f &box2_maxPt);

float getBoxIoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt);

cv::Scalar color_map(int64_t n);


template <typename Arg1, typename... Args>
inline void debug_v(const char* fmt, const Arg1 &arg1, const Args&... args){ vioLogger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void debug_v(const T& msg){vioLogger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void info_v(const char* fmt, const Arg1 &arg1, const Args&... args){vioLogger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void info_v(const T& msg){vioLogger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void warn_v(const char* fmt, const Arg1 &arg1, const Args&... args){vioLogger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void warn_v(const T& msg){vioLogger->log(spdlog::level::warn, msg);}


template <typename Arg1, typename... Args>
inline void debug_s(const char* fmt, const Arg1 &arg1, const Args&... args){ sgLogger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void debug_s(const T& msg){sgLogger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void info_s(const char* fmt, const Arg1 &arg1, const Args&... args){sgLogger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void info_s(const T& msg){sgLogger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void warn_s(const char* fmt, const Arg1 &arg1, const Args&... args){sgLogger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void warn_s(const T& msg){sgLogger->log(spdlog::level::warn, msg);}


template <typename Arg1, typename... Args>
inline void debug_t(const char* fmt, const Arg1 &arg1, const Args&... args){ tkLogger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void debug_t(const T& msg){tkLogger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void info_t(const char* fmt, const Arg1 &arg1, const Args&... args){tkLogger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void info_t(const T& msg){tkLogger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void warn_t(const char* fmt, const Arg1 &arg1, const Args&... args){tkLogger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void warn_t(const T& msg){tkLogger->log(spdlog::level::warn, msg);}


#endif //DYNAMIC_VINS_UTILS_H
