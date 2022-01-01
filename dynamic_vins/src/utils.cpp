//
// Created by chen on 2021/12/1.
//

#include "utils.h"

void SegImage::setMask(){
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


void SegImage::setMaskGpu(){
    if(insts_info.empty()){
        sgLogger->warn("Can not detect any object in picture");
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
    sgLogger->debug("setMaskGpu merge_tensor:type:{}",ss.str());
    sgLogger->debug("setMaskGpu merge_mask_gpu:({},{}) type:{}",merge_mask_gpu.rows,merge_mask_gpu.cols,merge_mask_gpu.type());
    sgLogger->debug("setMaskGpu inv_merge_mask_gpu:({},{}) type:{}",inv_merge_mask_gpu.rows,inv_merge_mask_gpu.cols,inv_merge_mask_gpu.type());

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



void SegImage::setMaskGpuSimple(){
    if(insts_info.empty()){
        sgLogger->warn("Can not detect any object in picture");
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




void SegImage::setGrayImage(){
    cv::cvtColor(color0, gray0, CV_BGR2GRAY);
    if(!color1.empty())
        cv::cvtColor(color1, gray1, CV_BGR2GRAY);
}

void SegImage::setGrayImageGpu(){
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
}

void SegImage::setColorImage(){
    cv::cvtColor(gray0, color0, CV_GRAY2BGR);
    if(!gray1.empty())
        cv::cvtColor(gray1, color1, CV_GRAY2BGR);
}


void SegImage::setColorImageGpu(){
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

void draw_text(cv::Mat &img, const std::string &str, const cv::Scalar &color, const cv::Point& pos,  float scale, int thickness,bool reverse) {
    auto t_size = cv::getTextSize(str, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, nullptr);
    cv::Point bottom_left, upper_right;
    if (reverse) {
        upper_right = pos;
        bottom_left = cv::Point(upper_right.x - t_size.width, upper_right.y + t_size.height);
    } else {
        bottom_left = pos;
        upper_right = cv::Point(bottom_left.x + t_size.width, bottom_left.y - t_size.height);
    }

    cv::rectangle(img, bottom_left, upper_right, color, -1);
    cv::putText(img, str, bottom_left, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255),thickness);
}

void draw_bbox(cv::Mat &img, const cv::Rect2f& bbox, const std::string &label, const cv::Scalar &color) {
    cv::rectangle(img, bbox, color);
    if (!label.empty()) {
        draw_text(img, label, color, bbox.tl());
    }
}



float getBoxIoU(const cv::Point2f &box1_minPt, const cv::Point2f &box1_maxPt,
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
float getBoxIoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt) {
    auto in = (bb_test & bb_gt).area();
    auto un = bb_test.area() + bb_gt.area() - in;
    if (un <  DBL_EPSILON)
        return 0;
    return in / un;
}


cv::Scalar color_map(int64_t n) {
    auto bit_get = [](int64_t x, int64_t i) {
        return x & (1 << i);
    };

    int64_t r = 0, g = 0, b = 0;
    int64_t i = n;
    for (int64_t j = 7; j >= 0; --j) {
        r |= bit_get(i, 0) << j;
        g |= bit_get(i, 1) << j;
        b |= bit_get(i, 2) << j;
        i >>= 3;
    }
    return cv::Scalar(b, g, r);
}
