/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef DYNAMIC_VINS_PIPELINE_H
#define DYNAMIC_VINS_PIPELINE_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>

#include "basic/def.h"
#include "basic/box2d.h"

namespace dynamic_vins{\


struct ImageInfo{
    int origin_h,origin_w;
    ///图像的裁切信息
    int rect_x, rect_y, rect_w, rect_h;
};


class Pipeline {
public:
    using Ptr=std::shared_ptr<Pipeline>;

    Pipeline()=default;

    std::tuple<float,float> GetXYWHS(int img_h,int img_w);

    cv::Mat ReadImage(const std::string& fileName);
    cv::Mat ProcessPad(cv::Mat &img);
    cv::Mat ProcessPadCuda(cv::Mat &img);
    cv::Mat ProcessPadCuda(cv::cuda::GpuMat &img);
    void ProcessKitti(cv::Mat &input, cv::Mat &output0, cv::Mat &output1);
    cv::Mat ProcessCut(cv::Mat &img);

    void* SetInputTensor(cv::Mat &img);
    void* ProcessInput(cv::Mat &img){
        auto t = ImageToTensor(img);
        return ProcessInput(t);
    }

    cv::Mat ProcessMask(cv::Mat &mask, std::vector<Box2D> &insts);

    static void SetBufferWithNorm(const cv::Mat &img, float *buffer);

    static torch::Tensor ImageToTensor(cv::cuda::GpuMat &img);


    void* ProcessInput(torch::Tensor &img);


    ImageInfo image_info;
    torch::Tensor input_tensor;

    static torch::Tensor ImageToTensor(cv::Mat &img);

private:
};

}


#endif //DYNAMIC_VINS_PIPELINE_H
