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
#include <NvInfer.h>

#include "parameters.h"

namespace dynamic_vins{\


class TicToc{
public:
    TicToc(){
        Tic();
    }
    void Tic(){
        start_ = std::chrono::system_clock::now();
    }
    double Toc(){
        end_ = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_ - start_;
        return elapsed_seconds.count() * 1000;
    }
    double TocThenTic(){
        auto t= Toc();
        Tic();
        return t;
    }
    void TocPrintTic(const char* str){
        cout << str << ":" << Toc() << " ms" << endl;
        Tic();
    }
private:
    std::chrono::time_point<std::chrono::system_clock> start_, end_;
};

struct Track {
    int id;
    cv::Rect2f box;
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

static std::string DimsToStr(nvinfer1::Dims list){
    std::string text= "[";
    for(int i=0;i<list.nbDims;++i){
        if (i > 0) text+= ", ";
        text += std::to_string(list.d[i]);
    }
    text += "]";
    return text;
}


inline cv::Point2f operator*(const cv::Point2f &lp,const cv::Point2f &rp){
    return {lp.x * rp.x,lp.y * rp.y};
}

template<typename MatrixType>
inline std::string EigenToStr(const MatrixType &m){
    std::string text;
    for(int i=0;i<m.rows();++i){
        for(int j=0;j<m.cols();++j)
            text+=fmt::format("{:.2f} ",m(i,j));
        if(m.rows()>1) text+="\n";
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
inline void Debugv(const char* fmt, const Arg1 &arg1, const Args&... args){ vio_logger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void Debugv(const T& msg){vio_logger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void Infov(const char* fmt, const Arg1 &arg1, const Args&... args){vio_logger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void Infov(const T& msg){vio_logger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void Warnv(const char* fmt, const Arg1 &arg1, const Args&... args){vio_logger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void Warnv(const T& msg){vio_logger->log(spdlog::level::warn, msg);}
template <typename Arg1, typename... Args>
inline void Errorv(const char* fmt, const Arg1 &arg1, const Args&... args){vio_logger->log(spdlog::level::err, fmt, arg1, args...);}
template<typename T>
inline void Errorv(const T& msg){vio_logger->log(spdlog::level::err, msg);}
template <typename Arg1, typename... Args>
inline void Criticalv(const char* fmt, const Arg1 &arg1, const Args&... args){vio_logger->log(spdlog::level::critical, fmt, arg1, args...);}
template<typename T>
inline void Criticalv(const T& msg){vio_logger->log(spdlog::level::critical, msg);}

template <typename Arg1, typename... Args>
inline void Debugs(const char* fmt, const Arg1 &arg1, const Args&... args){ sg_logger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void Debugs(const T& msg){sg_logger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void Infos(const char* fmt, const Arg1 &arg1, const Args&... args){sg_logger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void Infos(const T& msg){sg_logger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void Warns(const char* fmt, const Arg1 &arg1, const Args&... args){sg_logger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void Warns(const T& msg){sg_logger->log(spdlog::level::warn, msg);}
template <typename Arg1, typename... Args>
inline void Criticals(const char* fmt, const Arg1 &arg1, const Args&... args){sg_logger->log(spdlog::level::critical, fmt, arg1, args...);}
template<typename T>
inline void Criticals(const T& msg){sg_logger->log(spdlog::level::critical, msg);}

template <typename Arg1, typename... Args>
inline void Debugt(const char* fmt, const Arg1 &arg1, const Args&... args){ tk_logger->log(spdlog::level::debug, fmt, arg1, args...);}
template<typename T>
inline void Debugt(const T& msg){tk_logger->log(spdlog::level::debug, msg); }
template <typename Arg1, typename... Args>
inline void Infot(const char* fmt, const Arg1 &arg1, const Args&... args){tk_logger->log(spdlog::level::info, fmt, arg1, args...);}
template<typename T>
inline void Infot(const T& msg){tk_logger->log(spdlog::level::info, msg);}
template <typename Arg1, typename... Args>
inline void Warnt(const char* fmt, const Arg1 &arg1, const Args&... args){tk_logger->log(spdlog::level::warn, fmt, arg1, args...);}
template<typename T>
inline void Warnt(const T& msg){tk_logger->log(spdlog::level::warn, msg);}
template <typename Arg1, typename... Args>
inline void Criticalt(const char* fmt, const Arg1 &arg1, const Args&... args){tk_logger->log(spdlog::level::critical, fmt, arg1, args...);}
template<typename T>
inline void Criticalt(const T& msg){tk_logger->log(spdlog::level::critical, msg);}


}

#endif //DYNAMIC_VINS_UTILS_H
