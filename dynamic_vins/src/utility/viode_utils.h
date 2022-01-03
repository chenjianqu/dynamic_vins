/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_VIODE_UTILS_H
#define DYNAMIC_VINS_VIODE_UTILS_H

#include "../parameters.h"
#include "../utils.h"

class ViodeUtils {

};


namespace VIODE{


    inline unsigned int PixelToKey(uchar r, uchar g, uchar b){
        //key的计算公式r*1000000+g*1000*b
        return r*1000000+g*1000*b;
    }

    inline unsigned int PixelToKey(const cv::Point2f &pt, const cv::Mat &segImg){
        return PixelToKey(segImg.at<cv::Vec3b>(pt)[2], segImg.at<cv::Vec3b>(pt)[1], segImg.at<cv::Vec3b>(pt)[0]);
    }

    inline unsigned int PixelToKey(uchar* row_ptr){
        return PixelToKey(row_ptr[2], row_ptr[1], row_ptr[0]);
    }


    ////key的计算公式r*1000000+g*1000+b
    inline cv::Scalar KeyToPixel(unsigned int key){
        return {static_cast<double>(key%1000),static_cast<double>(static_cast<int>(key/1000)%1000),
                static_cast<double>(key/1000000)};//set b g r
    }


    //判断该点是否是动态物体点
    inline bool IsDynamic(unsigned int key){
        if(Config::ViodeDynamicIndex.count(Config::ViodeKeyToIndex[key]) != 0)
            return true;
        else
            return false;
    }

    inline bool IsDynamic(const cv::Point2f &pt, const cv::Mat &seg_img){
        auto key = PixelToKey(pt, seg_img);
        return IsDynamic(key);
    }

    inline bool IsDynamic(uchar* row_ptr){
        return IsDynamic(PixelToKey(row_ptr));
    }

    inline bool OnDynamicObject(const cv::Point2f &pt, const cv::Mat &label_img, unsigned int key){
        return PixelToKey(pt, label_img) == key;
    }

    void SetViodeMaskSimple(SegImage &img);

    void SetViodeMask(SegImage &img);
}



#endif //DYNAMIC_VINS_VIODE_UTILS_H
