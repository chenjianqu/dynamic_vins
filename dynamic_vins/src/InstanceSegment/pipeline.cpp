/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "pipeline.h"

#include <iostream>
#include <opencv2/cudaimgproc.hpp>

#include "featureTracker/segment_image.h"
#include "utils.h"

namespace dynamic_vins{\


using InterpolateFuncOptions=torch::nn::functional::InterpolateFuncOptions;
namespace idx=torch::indexing;

std::tuple<float,float> Pipeline::GetXYWHS(int img_h,int img_w)
{
    image_info.origin_h = img_h;
    image_info.origin_w = img_w;

    int w, h, x, y;
    float r_w = Config::kInputWidth / (img_w * 1.0f);
    float r_h = Config::kInputHeight / (img_h * 1.0f);
    if (r_h > r_w) {
        w = Config::kInputWidth;
        h = r_w * img_h;
        if(h%2==1)h++;//这里确保h为偶数，便于后面的使用
        x = 0;
        y = (Config::kInputHeight - h) / 2;
    } else {
        w = r_h* img_w;
        if(w%2==1)w++;
        h = Config::kInputHeight;
        x = (Config::kInputWidth - w) / 2;
        y = 0;
    }

    image_info.rect_x = x;
    image_info.rect_y = y;
    image_info.rect_w = w;
    image_info.rect_h = h;

    return {r_h,r_w};
}


cv::Mat Pipeline::ReadImage(const std::string& fileName)
{
    cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR);
    if (img.empty())
        return cv::Mat();

    auto [r_h,r_w] = GetXYWHS(img.rows,img.cols);

    //将img resize为(INPUT_W,INPUT_H)
    cv::Mat re(image_info.rect_h, image_info.rect_w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    //resizeByNN(img.data, re.data, img.rows, img.cols, img.channels(), re.rows, re.cols);

    //将图片复制到out中
    cv::Mat out(Config::kInputHeight, Config::kInputWidth, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(image_info.rect_x, image_info.rect_y, re.cols, re.rows)));

    return out;
}


cv::Mat Pipeline::ProcessPad(cv::Mat &img)
{
    TicToc tt;

    auto [r_h,r_w] = GetXYWHS(img.rows,img.cols);

    //将img resize为(INPUT_W,INPUT_H)
    cv::Mat re(image_info.rect_h, image_info.rect_w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    //resizeByNN(img.data, re.data, img.rows, img.cols, img.channels(), re.rows, re.cols);

    tt.TocPrintTic("resize");

    //将图片复制到out中
    cv::Mat out(Config::kInputHeight, Config::kInputWidth, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(image_info.rect_x, image_info.rect_y, re.cols, re.rows)));

    tt.TocPrintTic("copyTo out");

    return out;
}


cv::Mat Pipeline::ProcessPadCuda(cv::Mat &img)
{
    TicToc tt;

    auto [r_h,r_w] = GetXYWHS(img.rows,img.cols);

    static cv::Scalar mag_color(kSoloImgMean[2], kSoloImgMean[1], kSoloImgMean[0]);

    cv::Mat out;
    cv::resize(img,out,cv::Size(image_info.rect_w, image_info.rect_h));
    if (r_h > r_w) { //在图像顶部和下部拼接空白图像
        int cat_h= (Config::kInputHeight - image_info.rect_h) / 2;
        int cat_w=Config::kInputWidth;
        cv::Mat cat_img=cv::Mat(cat_h, cat_w, CV_8UC3,mag_color);
        cv::vconcat(cat_img,out,out);
        cv::vconcat(out,cat_img,out);
    } else {
        int cat_w= (Config::kInputWidth - image_info.rect_w) / 2;
        int cat_h=Config::kInputHeight;
        cv::Mat cat_img=cv::Mat(cat_h, cat_w, CV_8UC3,mag_color);
        cv::hconcat(cat_img,out,out);
        cv::hconcat(out,cat_img,out);
    }

    return out;
}


cv::Mat Pipeline::ProcessPadCuda(cv::cuda::GpuMat &img)
{
    TicToc tt;

    auto [r_h,r_w] = GetXYWHS(img.rows,img.cols);

    static cv::Scalar mag_color(kSoloImgMean[2], kSoloImgMean[1], kSoloImgMean[0]);

    cv::Mat out;
    cv::resize(img,out,cv::Size(image_info.rect_w, image_info.rect_h));
    if (r_h > r_w) { //在图像顶部和下部拼接空白图像
        int cat_h = (Config::kInputHeight - image_info.rect_h) / 2;
        int cat_w = Config::kInputWidth;
        cv::Mat cat_img=cv::Mat(cat_h, cat_w, CV_8UC3,mag_color);
        cv::vconcat(cat_img,out,out);
        cv::vconcat(out,cat_img,out);
    } else {
        int cat_w= (Config::kInputWidth - image_info.rect_w) / 2;
        int cat_h=Config::kInputHeight;
        cv::Mat cat_img=cv::Mat(cat_h, cat_w, CV_8UC3,mag_color);
        cv::hconcat(cat_img,out,out);
        cv::hconcat(out,cat_img,out);
    }

    return out;
}


void* Pipeline::SetInputTensor(cv::Mat &img)
{
    TicToc tt;

    auto [r_h,r_w] = GetXYWHS(img.rows,img.cols);

    cv::Mat out;
    cv::resize(img,out,cv::Size(image_info.rect_w, image_info.rect_h));

    Debugs("SetInputTensor resize:{} ms", tt.TocThenTic());

    ///拼接图像边缘
    static cv::Scalar mag_color(kSoloImgMean[2], kSoloImgMean[1], kSoloImgMean[0]);
    if (r_h > r_w) { //在图像顶部和下部拼接空白图像
        int cat_h = (Config::kInputHeight - image_info.rect_h) / 2;
        int cat_w = Config::kInputWidth;
        cv::Mat cat_img=cv::Mat(cat_h, cat_w, CV_8UC3,mag_color);
        cv::vconcat(cat_img,out,out);
        cv::vconcat(out,cat_img,out);
    } else {
        int cat_w= (Config::kInputWidth - image_info.rect_w) / 2;
        int cat_h=Config::kInputHeight;
        cv::Mat cat_img=cv::Mat(cat_h, cat_w, CV_8UC3,mag_color);
        cv::hconcat(cat_img,out,out);
        cv::hconcat(out,cat_img,out);
    }

    Debugs("SetInputTensor concat:{} ms", tt.TocThenTic());

    cv::cvtColor(out,out,CV_BGR2RGB);

    Debugs("SetInputTensor cvtColor:{} ms", tt.TocThenTic());


    cv::Mat img_float;
    out.convertTo(img_float,CV_32FC3);

    Debugs("SetInputTensor convertTo:{} ms", tt.TocThenTic());


    torch::Tensor input_tensor_cpu = torch::from_blob(img_float.data, { img_float.rows,img_float.cols ,3 }, torch::kFloat32);
    input_tensor = input_tensor_cpu.to(torch::kCUDA).permute({2,0,1});

    Debugs("SetInputTensor from_blob:{} ms", tt.TocThenTic());

    static torch::Tensor mean_t=torch::from_blob(kSoloImgMean, {3, 1, 1}, torch::kFloat).to(torch::kCUDA).expand({3, img_float.rows, img_float.cols});
    static torch::Tensor std_t=torch::from_blob(kSoloImgStd, {3, 1, 1}, torch::kFloat).to(torch::kCUDA).expand({3, img_float.rows, img_float.cols});

    input_tensor = ((input_tensor-mean_t)/std_t).contiguous();

    Debugs("SetInputTensor norm:{} ms", tt.TocThenTic());

    return input_tensor.data_ptr();
}

/**
 * 输入预处理
 * @param img 未经处理的图像张量，shape=[3,h,w],值范围[0-255]，数据类型Float32
 * @return
 */
void* Pipeline::ProcessInput(torch::Tensor &img){
    auto [r_h,r_w] = GetXYWHS(img.sizes()[1],img.sizes()[2]);
    input_tensor= img;
    static torch::Tensor mean_t=torch::from_blob(kSoloImgMean, {3, 1, 1}, torch::kFloat32).to(torch::kCUDA).
            expand({3, image_info.origin_h, image_info.origin_w});
    static torch::Tensor std_t=torch::from_blob(kSoloImgStd, {3, 1, 1}, torch::kFloat32).to(torch::kCUDA).
            expand({3, image_info.origin_h, image_info.origin_w});
    input_tensor = ((input_tensor-mean_t)/std_t);
    ///resize
    auto options=InterpolateFuncOptions().mode(torch::kBilinear).align_corners(true);
    options=options.size(std::vector<int64_t>({image_info.rect_h, image_info.rect_w}));
    input_tensor = torch::nn::functional::interpolate(input_tensor.unsqueeze(0),options).squeeze(0);
    ///拼接图像边缘
    static auto op = torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat32);
    static cv::Scalar mag_color(kSoloImgMean[2], kSoloImgMean[1], kSoloImgMean[0]);
    if (r_h > r_w) { //在图像顶部和下部拼接空白图像
        int cat_w = Config::kInputWidth;
        int cat_h = (Config::kInputHeight - image_info.rect_h) / 2;
        torch::Tensor cat_t = torch::zeros({3,cat_h,cat_w},op);
        input_tensor = torch::cat({cat_t,input_tensor,cat_t},1);
    } else {
        int cat_w= (Config::kInputWidth - image_info.rect_w) / 2;
        int cat_h=Config::kInputHeight;
        torch::Tensor cat_t = torch::zeros({3,cat_h,cat_w},op);
        input_tensor = torch::cat({cat_t,input_tensor,cat_t},2);
    }
    input_tensor = input_tensor.contiguous();
    return input_tensor.data_ptr();
}




cv::Mat Pipeline::ProcessMask(cv::Mat &mask, std::vector<InstInfo> &insts)
{
    cv::Mat rect_img = mask(cv::Rect(image_info.rect_x, image_info.rect_y, image_info.rect_w, image_info.rect_h));
    cv::Mat out;
    cv::resize(rect_img, out, cv::Size(image_info.origin_w, image_info.origin_h), 0, 0, cv::INTER_LINEAR);

    ///调整包围框
    float factor_x = out.cols *1.f / rect_img.cols;
    float factor_y = out.rows *1.f / rect_img.rows;
    for(auto &inst : insts){
        inst.min_pt.x -= image_info.rect_x;
        inst.min_pt.y -= image_info.rect_y;
        inst.max_pt.x -= image_info.rect_x;
        inst.max_pt.y -= image_info.rect_y;

        inst.min_pt.x *= factor_x;
        inst.min_pt.y *= factor_y;
        inst.max_pt.x *= factor_x;
        inst.max_pt.y *= factor_y;
    }


    return out;
}


void Pipeline::ProcessKitti(cv::Mat &input, cv::Mat &output0, cv::Mat &output1)
{
    int input_h=448;
    int input_w=672;

    ///将输入图片划分为两个图片
    int resize_h = input_h;
    float h_factor = resize_h *1.f / input.rows;
    int resize_w = input.cols * h_factor;

    cout<<resize_h<<" "<<resize_w<<endl;

    cv::Mat new_img;
    cv::resize(input, new_img, cv::Size(resize_w,resize_h), 0, 0, cv::INTER_LINEAR);

    int half_width= std::ceil(resize_w/2.f);
    cout<<"half_width:"<<half_width<<endl;

    cv::Mat img0=new_img(cv::Rect(0, 0, half_width,new_img.rows));
    cout<<"img0:"<<img0.size<<endl;

    int start_index = resize_w - half_width;
    cout<<"start_index:"<<start_index<<endl;

    cv::Mat img1=new_img(cv::Rect(start_index, 0, half_width,new_img.rows));
    cout<<"img1:"<<img1.size<<endl;

    cout<<"new_img:"<<new_img.size<<endl;

    image_info.origin_h = img0.rows;
    image_info.origin_w = img0.cols;

    ///将两个半图像进行缩放
    int w, h, x, y;
    float r_w = input_w / (img0.cols * 1.0f);
    float r_h = input_h / (img0.rows * 1.0f);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img0.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h* img0.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    //将img resize为(INPUT_W,INPUT_H)
    cv::Mat re0(h, w, CV_8UC3);
    cv::resize(img0, re0, re0.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat re1(h, w, CV_8UC3);
    cv::resize(img1, re1, re1.size(), 0, 0, cv::INTER_LINEAR);

    //将图片复制到out中
    output0=cv::Mat(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re0.copyTo(output0(cv::Rect(x, y, re0.cols, re0.rows)));

    output1=cv::Mat(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re1.copyTo(output1(cv::Rect(x, y, re1.cols, re1.rows)));
}


cv::Mat Pipeline::ProcessCut(cv::Mat &img)
{
    int resize_h = Config::kInputHeight;
    float h_factor = resize_h *1.f / img.rows;
    int resize_w = img.cols * h_factor;

    cv::Mat new_img;
    cv::resize(img, new_img, cv::Size(resize_w,resize_h), 0, 0, cv::INTER_LINEAR);

    cout<<new_img.size<<endl;

    cv::Mat out(cv::Size(Config::kInputWidth, Config::kInputHeight), CV_8UC3, cv::Scalar(128, 128, 128));

    out=new_img(cv::Rect(0, 0, Config::kInputWidth, Config::kInputHeight));

    cout<<out.size<<endl;

    image_info.origin_h = out.rows;
    image_info.origin_w = out.cols;

    return out;
}


void Pipeline::SetBufferWithNorm(const cv::Mat &img, float *buffer)
{
    int i = 0,b_cnt=0;
    auto rows = std::min(img.rows,Config::kInputHeight);
    auto cols = std::min(img.cols,Config::kInputWidth);
    for (int row = 0; row < rows; ++row) {
        uchar* uc_pixel = img.data + row * img.step;
        for (int col = 0; col < cols; ++col) {
            buffer[b_cnt * 3 * Config::kInputHeight * Config::kInputWidth + i] = (uc_pixel[2] - kSoloImgMean[0]) / kSoloImgStd[0];
            buffer[b_cnt * 3 * Config::kInputHeight * Config::kInputWidth + i + Config::kInputHeight * Config::kInputWidth] = (uc_pixel[1] - kSoloImgMean[1]) / kSoloImgStd[1];
            buffer[b_cnt * 3 * Config::kInputHeight * Config::kInputWidth + i + 2 * Config::kInputHeight * Config::kInputWidth] = (uc_pixel[0] - kSoloImgMean[2]) / kSoloImgStd[2];
            uc_pixel += 3;
            ++i;
        }
    }

}


torch::Tensor Pipeline::ImageToTensor(cv::Mat &img) {
    if(img.empty()){
        return torch::Tensor();
    }
    cv::Mat img_float;
    img.convertTo(img_float,CV_32FC3);
    auto input_tensor = torch::from_blob(img_float.data, {img.rows,img.cols ,3 }, torch::kFloat32).to(torch::kCUDA);
    ///bgr->rgb
    input_tensor = torch::cat({
        input_tensor.index({"...",2}).unsqueeze(2),
        input_tensor.index({"...",1}).unsqueeze(2),
        input_tensor.index({"...",0}).unsqueeze(2)
        },2);

    ///hwc->chw
    input_tensor = input_tensor.permute({2,0,1});
    return input_tensor;
}

torch::Tensor Pipeline::ImageToTensor(cv::cuda::GpuMat &img){
    if(img.empty()){
        return torch::Tensor();
    }
    cv::Mat img_float;
    img.convertTo(img_float,CV_32FC3);
    auto input_tensor = torch::from_blob(img_float.data, {img.rows,img.cols ,3 }, torch::kFloat32).clone();
    ///bgr->rgb
    input_tensor = torch::cat({
        input_tensor.index({"...",2}).unsqueeze(2),
        input_tensor.index({"...",1}).unsqueeze(2),
        input_tensor.index({"...",0}).unsqueeze(2)
        },2);

    ///hwc->chw
    input_tensor = input_tensor.permute({2,0,1});
    return input_tensor;
}


}