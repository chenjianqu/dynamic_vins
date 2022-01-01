//
// Created by chen on 2021/11/7.
//

#ifndef DYNAMIC_VINS_PIPELINE_H
#define DYNAMIC_VINS_PIPELINE_H

#include <iostream>

#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torchvision/vision.h>

#include "../parameters.h"
#include "../utils.h"
#include "../featureTracker/SegmentImage.h"

class Pipeline {
public:
    using Ptr=std::shared_ptr<Pipeline>;

    Pipeline(){

    }

    template<typename ImageType>
    std::tuple<float,float> getXYWHS(const ImageType &img);

    cv::Mat readImage(const std::string& fileName);
    cv::Mat processPad(cv::Mat &img);
    cv::Mat processPadCuda(cv::Mat &img);
    cv::Mat processPadCuda(cv::cuda::GpuMat &img);
    void processKitti(cv::Mat &input,cv::Mat &output0,cv::Mat &output1);
    cv::Mat processCut(cv::Mat &img);

    static void setBufferWithNorm(const cv::Mat &img,float *buffer);

    void* setInputTensor(cv::Mat &img);
    void* setInputTensorCuda(cv::Mat &img);

    cv::Mat processMask(cv::Mat &mask,std::vector<InstInfo> &insts);

    ImageInfo imageInfo;
    torch::Tensor input_tensor;

private:
};


#endif //DYNAMIC_VINS_PIPELINE_H
