/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_VIODE_UTILS_H
#define DYNAMIC_VINS_VIODE_UTILS_H

#include "utils/parameters.h"
#include "basic/def.h"
#include "basic/semantic_image.h"


namespace dynamic_vins{ \

class VIODE{
public:
    static unsigned int PixelToKey(uchar r, uchar g, uchar b){
        //key的计算公式r*1000000+g*1000*b
        return r*1000000+g*1000*b;
    }

     static unsigned int PixelToKey(const cv::Point2f &pt, const cv::Mat &segImg){
        return PixelToKey(segImg.at<cv::Vec3b>(pt)[2], segImg.at<cv::Vec3b>(pt)[1], segImg.at<cv::Vec3b>(pt)[0]);
    }

     static unsigned int PixelToKey(uchar* row_ptr){
        return PixelToKey(row_ptr[2], row_ptr[1], row_ptr[0]);
    }


    ////key的计算公式r*1000000+g*1000+b
     static cv::Scalar KeyToPixel(unsigned int key){
        return {static_cast<double>(key%1000),static_cast<double>(static_cast<int>(key/1000)%1000),
                static_cast<double>(key/1000000)};//set b g r
    }


    //判断该点是否是动态物体点
     static bool IsDynamic(unsigned int key){
        if(ViodeDynamicIndex.count(ViodeKeyToIndex[key]) != 0)
            return true;
        else
            return false;
    }

     static bool IsDynamic(const cv::Point2f &pt, const cv::Mat &seg_img){
        auto key = PixelToKey(pt, seg_img);
        return IsDynamic(key);
    }

     static bool IsDynamic(uchar* row_ptr){
        return IsDynamic(PixelToKey(row_ptr));
    }

     static bool OnDynamicObject(const cv::Point2f &pt, const cv::Mat &label_img, unsigned int key){
        return PixelToKey(pt, label_img) == key;
    }

    static void SetViodeMaskSimple(SemanticImage &img);


    static void SetViodeMaskAndRoi(SemanticImage &img);

    static std::unordered_map<unsigned int,int> ReadViodeRgbIds(const string &rgb_to_label_file);


    static void SetParameters(const std::string &config_path);

    inline static std::unordered_map<unsigned int,int> ViodeKeyToIndex;
    inline static std::set<int> ViodeDynamicIndex;

private:
    struct InstanceSimple{
        InstanceSimple()=default;
        InstanceSimple(int row_start_, int row_end_, int col_start_, int col_end_):
        row_start(row_start_),row_end(row_end_),col_start(col_start_),col_end(col_end_){
            mask=cv::Mat(row_end-row_start,col_end-col_start,CV_8UC1,cv::Scalar(0));
        }
        cv::Mat mask;
        size_t num_pixel{0};
        int row_start{},row_end{},col_start{},col_end{};
        int row_min,row_max,col_min,col_max;
    };

    static cv::Mat BuildViodeMask(SemanticImage &img,std::unordered_map<unsigned int,InstanceSimple> &insts);

};



}


#endif //DYNAMIC_VINS_VIODE_UTILS_H
