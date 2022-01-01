//
// Created by chen on 2021/12/3.
//

#ifndef DYNAMIC_VINS_VIODEUTILS_H
#define DYNAMIC_VINS_VIODEUTILS_H

#include "../parameters.h"
#include "../utils.h"

class ViodeUtils {

};


namespace VIODE{



    inline unsigned int pixel2key(uchar r, uchar g, uchar b){
        //key的计算公式r*1000000+g*1000*b
        return r*1000000+g*1000*b;
    }

    inline unsigned int pixel2key(const cv::Point2f &pt,const cv::Mat &segImg){
        return pixel2key(segImg.at<cv::Vec3b>(pt)[2], segImg.at<cv::Vec3b>(pt)[1], segImg.at<cv::Vec3b>(pt)[0]);
    }

    inline unsigned int pixel2key(uchar* row_ptr){
        return pixel2key(row_ptr[2], row_ptr[1], row_ptr[0]);
    }


    ////key的计算公式r*1000000+g*1000+b
    inline cv::Scalar key2pixel(unsigned int key){
        return {static_cast<double>(key%1000),static_cast<double>(static_cast<int>(key/1000)%1000),
                static_cast<double>(key/1000000)};//set b g r
    }


    //判断该点是否是动态物体点
    inline bool isDynamic(unsigned int key)
    {
        if(Config::VIODE_DynamicIndex.count(Config::VIODE_Key2Index[key]) != 0)
            return true;
        else
            return false;
    }


    inline bool isDynamic(const cv::Point2f &pt,const cv::Mat &segImg){
        auto key = pixel2key(pt,segImg);
        return isDynamic(key);
    }

    inline bool isDynamic(uchar* row_ptr){
        return isDynamic(pixel2key(row_ptr));
    }



    inline bool onDynamicObject(const cv::Point2f &pt,const cv::Mat &labelImg,unsigned int key)
    {
        return pixel2key(pt,labelImg) == key;
    }




    void setViodeMaskSimple(SegImage &img);


    void setViodeMask(SegImage &img);
}



#endif //DYNAMIC_VINS_VIODEUTILS_H
