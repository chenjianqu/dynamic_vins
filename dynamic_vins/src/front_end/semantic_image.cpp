/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/


#include "semantic_image.h"
#include "utils/log_utils.h"

namespace dynamic_vins{\


/**
 * 根据实例分割结果,设置每个实例的mask和背景mask
 */
void SemanticImage::SetMask(){
    exist_inst = !boxes2d.empty();
    if(!exist_inst){
        Warns("Can not detect any object in picture");
        return;
    }
    cv::Size mask_size((int)mask_tensor.sizes()[2],(int)mask_tensor.sizes()[1]);

    mask_tensor = mask_tensor.to(torch::kInt8).abs().clamp(0,1);

    ///计算合并的mask
    auto merge_tensor = (mask_tensor.sum(0).clamp(0,1)*255).to(torch::kUInt8);

    merge_mask_gpu = cv::cuda::GpuMat(mask_size, CV_8UC1, merge_tensor.data_ptr()).clone();///一定要clone，不然tensor内存的数据会被改变
    merge_mask_gpu.download(merge_mask);

    cv::cuda::bitwise_not(merge_mask_gpu,inv_merge_mask_gpu);
    inv_merge_mask_gpu.download(inv_merge_mask);

    Debugs("SemanticImage::SetMask() set inv_merge_mask_gpu");

    /*std::stringstream ss;
    ss<<merge_tensor.scalar_type();
    Debugs("SetMaskGpu merge_tensor:type:{}", ss.str());
    Debugs("SetMaskGpu merge_mask_gpu:({},{}) type:{}", merge_mask_gpu.rows, merge_mask_gpu.cols, merge_mask_gpu.type());
    Debugs("SetMaskGpu inv_merge_mask_gpu:({},{}) type:{}", inv_merge_mask_gpu.rows, inv_merge_mask_gpu.cols,inv_merge_mask_gpu.type());*/

    for(int i=0; i < (int)boxes2d.size(); ++i){
        auto inst_mask_tensor = mask_tensor[i];
        //boxes2d[i]->mask_tensor = inst_mask_tensor;
        boxes2d[i]->mask_gpu = cv::cuda::GpuMat(mask_size, CV_8UC1,
                                               (inst_mask_tensor * 255).to(torch::kUInt8).data_ptr()).clone();
        boxes2d[i]->mask_gpu.download(boxes2d[i]->mask_cv);
        ///cal center
        /*auto inds=inst_mask_tensor.nonzero();
        auto center_inds = inds.sum(0) / inds.sizes()[0];
        boxes2d[i]->mask_center=cv::Point2f(center_inds.index({1}).item().toFloat(),
                                           center_inds.index({0}).item().toFloat());*/
    }
    Debugs("SemanticImage::SetMask() finished");

}


/**
 * 根据实例分割结果,计算背景区域的mask
 */
void SemanticImage::SetBackgroundMask(){
    exist_inst = !boxes2d.empty();
    if(!exist_inst){
        Warns("Can not detect any object in picture");
        return;
    }
    cv::Size mask_size((int)mask_tensor.sizes()[2],(int)mask_tensor.sizes()[1]);

    mask_tensor = mask_tensor.to(torch::kInt8).abs().clamp(0,1);

    ///计算合并的mask
    auto merge_tensor = (mask_tensor.sum(0).clamp(0,1)*255).to(torch::kUInt8);

    merge_mask_gpu = cv::cuda::GpuMat(mask_size, CV_8UC1, merge_tensor.data_ptr()).clone();///一定要clone，不然tensor内存的数据会被改变
    merge_mask_gpu.download(merge_mask);

    cv::cuda::bitwise_not(merge_mask_gpu,inv_merge_mask_gpu);
    inv_merge_mask_gpu.download(inv_merge_mask);
}


void SemanticImage::SetGrayImage(){
    cv::cvtColor(color0, gray0, CV_BGR2GRAY);
    if(!color1.empty())
        cv::cvtColor(color1, gray1, CV_BGR2GRAY);
}


void SemanticImage::SetGrayImageGpu(){
    if(color0_gpu.empty()){
        color0_gpu.upload(color0);
    }
    cv::cuda::cvtColor(color0_gpu,gray0_gpu,CV_BGR2GRAY);
    if(!color1.empty()){
        if(color1_gpu.empty()){
            color1_gpu.upload(color1);
        }
        cv::cuda::cvtColor(color1_gpu,gray1_gpu,CV_BGR2GRAY);
    }
    gray0_gpu.download(gray0);
    gray1_gpu.download(gray1);

}


void SemanticImage::SetColorImage(){
    cv::cvtColor(gray0, color0, CV_GRAY2BGR);
    if(!gray1.empty())
        cv::cvtColor(gray1, color1, CV_GRAY2BGR);
}


void SemanticImage::SetColorImageGpu(){
    if(gray0_gpu.empty()){
        gray0_gpu.upload(gray0);
    }
    cv::cuda::cvtColor(gray0_gpu, color0_gpu, CV_GRAY2BGR);
    if(!gray1.empty()){
        if(gray1_gpu.empty()){
            gray1_gpu.upload(gray1);
        }
        cv::cuda::cvtColor(gray1_gpu, color1_gpu, CV_GRAY2BGR);
    }
}












}


