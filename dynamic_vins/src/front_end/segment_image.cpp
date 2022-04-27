/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/


#include "segment_image.h"
#include "utils/log_utils.h"

namespace dynamic_vins{\


void SegImage::SetMask(){
    cv::Size mask_size((int)mask_tensor.sizes()[2],(int)mask_tensor.sizes()[1]);

    ///计算合并的mask
    auto merger_tensor = (mask_tensor.sum(0).to(torch::kInt8) * 255);
    merge_mask = cv::Mat(mask_size, CV_8UC1, merger_tensor.to(torch::kCPU).data_ptr()).clone();
    mask_tensor = mask_tensor.to(torch::kInt8);

    for(int i=0; i < (int)insts_info.size(); ++i)
    {
        auto inst_mask_tensor = mask_tensor[i];
        insts_info[i].mask_cv = std::move(cv::Mat(mask_size, CV_8UC1, (inst_mask_tensor * 255).to(torch::kCPU).data_ptr()).clone());
        ///cal center
        auto inds=inst_mask_tensor.nonzero();
        auto center_inds = inds.sum(0) / inds.sizes()[0];
        insts_info[i].mask_center=cv::Point2f(center_inds.index({1}).item().toFloat(),center_inds.index({0}).item().toFloat());
    }
}


void SegImage::SetMaskGpu(){
    exist_inst = !insts_info.empty();
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

    std::stringstream ss;
    ss<<merge_tensor.scalar_type();
    Debugs("SetMaskGpu merge_tensor:type:{}", ss.str());
    Debugs("SetMaskGpu merge_mask_gpu:({},{}) type:{}", merge_mask_gpu.rows, merge_mask_gpu.cols, merge_mask_gpu.type());
    Debugs("SetMaskGpu inv_merge_mask_gpu:({},{}) type:{}", inv_merge_mask_gpu.rows, inv_merge_mask_gpu.cols,
           inv_merge_mask_gpu.type());

    for(int i=0; i < (int)insts_info.size(); ++i){
        auto inst_mask_tensor = mask_tensor[i];
        insts_info[i].mask_tensor = inst_mask_tensor;
        insts_info[i].mask_gpu = cv::cuda::GpuMat(mask_size, CV_8UC1, (inst_mask_tensor * 255).to(torch::kUInt8).data_ptr()).clone();
        insts_info[i].mask_gpu.download(insts_info[i].mask_cv);
        ///cal center
        auto inds=inst_mask_tensor.nonzero();
        auto center_inds = inds.sum(0) / inds.sizes()[0];
        insts_info[i].mask_center=cv::Point2f(center_inds.index({1}).item().toFloat(),center_inds.index({0}).item().toFloat());
    }
}



void SegImage::SetMaskGpuSimple(){
    exist_inst = !insts_info.empty();
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


void SegImage::SetGrayImage(){
    cv::cvtColor(color0, gray0, CV_BGR2GRAY);
    if(!color1.empty())
        cv::cvtColor(color1, gray1, CV_BGR2GRAY);
}


void SegImage::SetGrayImageGpu(){
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


void SegImage::SetColorImage(){
    cv::cvtColor(gray0, color0, CV_GRAY2BGR);
    if(!gray1.empty())
        cv::cvtColor(gray1, color1, CV_GRAY2BGR);
}


void SegImage::SetColorImageGpu(){
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




float CalBoxIoU(const cv::Point2f &box1_minPt, const cv::Point2f &box1_maxPt,
                const cv::Point2f &box2_minPt, const cv::Point2f &box2_maxPt){
    cv::Point2f center1 = (box1_minPt+box1_maxPt)/2.f;
    cv::Point2f center2 = (box2_minPt+box2_maxPt)/2.f;
    float w1 = box1_maxPt.x - (float)box1_minPt.x;
    float h1 = box1_maxPt.y - (float)box1_minPt.y;
    float w2 = box2_maxPt.x - (float)box2_minPt.x;
    float h2 = box2_maxPt.y - (float)box2_minPt.y;

    if(std::abs(center1.x - center2.x) >= (w1/2+w2/2) && std::abs(center1.y - center2.y) >= (h1/2+h2/2)){
        return 0;
    }

    float inter_w = w1 + w2 - (std::max(center1.x + w1, center2.x + w2) - std::min(center1.x, center2.x));
    float inter_h = h1 + h2 - (std::max(center1.y + h1, center2.y + h2) - std::min(center1.y, center2.y));
    return (inter_h*inter_w) / (w1*h1 + w2*h2 - inter_h*inter_w);
}


/**
 * 计算两个box之间的IOU
 * @param bb_test
 * @param bb_gt
 * @return
 */
float CalBoxIoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt) {
    auto in = (bb_test & bb_gt).area();
    auto un = bb_test.area() + bb_gt.area() - in;
    if (un <  DBL_EPSILON)
        return 0;
    return in / un;
}







}

