//
// Created by chen on 2021/11/7.
//

#include "pipeline.h"
#include <iostream>
#include <opencv2/cudaimgproc.hpp>

#include "../featureTracker/SegmentImage.h"
#include "../utils.h"

using namespace std;
using namespace torch::indexing;
using InterpolateFuncOptions=torch::nn::functional::InterpolateFuncOptions;





template<typename ImageType>
std::tuple<float,float> Pipeline::getXYWHS(const ImageType &img)
{
    imageInfo.originH = img.rows;
    imageInfo.originW = img.cols;

    int w, h, x, y;
    float r_w = Config::inputW / (img.cols*1.0f);
    float r_h = Config::inputH / (img.rows*1.0f);
    if (r_h > r_w) {
        w = Config::inputW;
        h = r_w * img.rows;
        if(h%2==1)h++;//这里确保h为偶数，便于后面的使用
        x = 0;
        y = (Config::inputH - h) / 2;
    } else {
        w = r_h* img.cols;
        if(w%2==1)w++;
        h = Config::inputH;
        x = (Config::inputW - w) / 2;
        y = 0;
    }

    imageInfo.rect_x = x;
    imageInfo.rect_y = y;
    imageInfo.rect_w = w;
    imageInfo.rect_h = h;

    return {r_h,r_w};
}


cv::Mat Pipeline::readImage(const std::string& fileName)
{
    cv::Mat img = cv::imread(fileName, cv::IMREAD_COLOR);
    if (img.empty())
        return cv::Mat();

    auto [r_h,r_w] = getXYWHS(img);

    //将img resize为(INPUT_W,INPUT_H)
    cv::Mat re(imageInfo.rect_h, imageInfo.rect_w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    //resizeByNN(img.data, re.data, img.rows, img.cols, img.channels(), re.rows, re.cols);

    //将图片复制到out中
    cv::Mat out(Config::inputH, Config::inputW, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(imageInfo.rect_x, imageInfo.rect_y, re.cols, re.rows)));

    return out;
}


cv::Mat Pipeline::processPad(cv::Mat &img)
{
    TicToc tt;

    auto [r_h,r_w] = getXYWHS(img);

    //将img resize为(INPUT_W,INPUT_H)
    cv::Mat re(imageInfo.rect_h, imageInfo.rect_w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    //resizeByNN(img.data, re.data, img.rows, img.cols, img.channels(), re.rows, re.cols);

    tt.toc_print_tic("resize");

    //将图片复制到out中
    cv::Mat out(Config::inputH, Config::inputW, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(imageInfo.rect_x, imageInfo.rect_y, re.cols, re.rows)));

    tt.toc_print_tic("copyTo out");

    return out;
}



cv::Mat Pipeline::processPadCuda(cv::Mat &img)
{
    TicToc tt;

    auto [r_h,r_w] = getXYWHS(img);

    static cv::Scalar mag_color(SOLO_IMG_MEAN[2],SOLO_IMG_MEAN[1],SOLO_IMG_MEAN[0]);

    cv::Mat out;
    cv::resize(img,out,cv::Size(imageInfo.rect_w,imageInfo.rect_h));
    if (r_h > r_w) { //在图像顶部和下部拼接空白图像
        int cat_h= (Config::inputH-imageInfo.rect_h)/2;
        int cat_w=Config::inputW;
        cv::Mat cat_img=cv::Mat(cat_h, cat_w, CV_8UC3,mag_color);
        cv::vconcat(cat_img,out,out);
        cv::vconcat(out,cat_img,out);
    } else {
        int cat_w= (Config::inputW-imageInfo.rect_w)/2;
        int cat_h=Config::inputH;
        cv::Mat cat_img=cv::Mat(cat_h, cat_w, CV_8UC3,mag_color);
        cv::hconcat(cat_img,out,out);
        cv::hconcat(out,cat_img,out);
    }

    return out;
}


cv::Mat Pipeline::processPadCuda(cv::cuda::GpuMat &img)
{
    TicToc tt;

    auto [r_h,r_w] = getXYWHS(img);

    static cv::Scalar mag_color(SOLO_IMG_MEAN[2],SOLO_IMG_MEAN[1],SOLO_IMG_MEAN[0]);

    cv::Mat out;
    cv::resize(img,out,cv::Size(imageInfo.rect_w,imageInfo.rect_h));
    if (r_h > r_w) { //在图像顶部和下部拼接空白图像
        int cat_h = (Config::inputH-imageInfo.rect_h)/2;
        int cat_w = Config::inputW;
        cv::Mat cat_img=cv::Mat(cat_h, cat_w, CV_8UC3,mag_color);
        cv::vconcat(cat_img,out,out);
        cv::vconcat(out,cat_img,out);
    } else {
        int cat_w= (Config::inputW-imageInfo.rect_w)/2;
        int cat_h=Config::inputH;
        cv::Mat cat_img=cv::Mat(cat_h, cat_w, CV_8UC3,mag_color);
        cv::hconcat(cat_img,out,out);
        cv::hconcat(out,cat_img,out);
    }

    return out;
}



void* Pipeline::setInputTensor(cv::Mat &img)
{
    TicToc tt;

    auto [r_h,r_w] = getXYWHS(img);

    cv::Mat out;
    cv::resize(img,out,cv::Size(imageInfo.rect_w,imageInfo.rect_h));

    sgLogger->debug("setInputTensor resize:{} ms",tt.toc_then_tic());

    ///拼接图像边缘
    static cv::Scalar mag_color(SOLO_IMG_MEAN[2],SOLO_IMG_MEAN[1],SOLO_IMG_MEAN[0]);
    if (r_h > r_w) { //在图像顶部和下部拼接空白图像
        int cat_h = (Config::inputH-imageInfo.rect_h)/2;
        int cat_w = Config::inputW;
        cv::Mat cat_img=cv::Mat(cat_h, cat_w, CV_8UC3,mag_color);
        cv::vconcat(cat_img,out,out);
        cv::vconcat(out,cat_img,out);
    } else {
        int cat_w= (Config::inputW-imageInfo.rect_w)/2;
        int cat_h=Config::inputH;
        cv::Mat cat_img=cv::Mat(cat_h, cat_w, CV_8UC3,mag_color);
        cv::hconcat(cat_img,out,out);
        cv::hconcat(out,cat_img,out);
    }

    sgLogger->debug("setInputTensor concat:{} ms",tt.toc_then_tic());

    cv::cvtColor(out,out,CV_BGR2RGB);

    sgLogger->debug("setInputTensor cvtColor:{} ms",tt.toc_then_tic());


    cv::Mat img_float;
    out.convertTo(img_float,CV_32FC3);

    sgLogger->debug("setInputTensor convertTo:{} ms",tt.toc_then_tic());


    torch::Tensor input_tensor_cpu = torch::from_blob(img_float.data, { img_float.rows,img_float.cols ,3 }, torch::kFloat32);
    input_tensor = input_tensor_cpu.to(torch::kCUDA).permute({2,0,1});

    sgLogger->debug("setInputTensor from_blob:{} ms",tt.toc_then_tic());

    static torch::Tensor mean_t=torch::from_blob(SOLO_IMG_MEAN,{3,1,1},torch::kFloat).to(torch::kCUDA).expand({3,img_float.rows,img_float.cols});
    static torch::Tensor std_t=torch::from_blob(SOLO_IMG_STD,{3,1,1},torch::kFloat).to(torch::kCUDA).expand({3,img_float.rows,img_float.cols});

    input_tensor = ((input_tensor-mean_t)/std_t).contiguous();

    sgLogger->debug("setInputTensor norm:{} ms",tt.toc_then_tic());

    return input_tensor.data_ptr();
}


void* Pipeline::setInputTensorCuda(cv::Mat &img)
{
    TicToc tt;

    auto [r_h,r_w] = getXYWHS(img);

    /*cv::cuda::GpuMat img_gpu(img);
    img_gpu.convertTo(img_gpu,CV_32FC3);
    input_tensor = torch::from_blob(img_gpu.data, { imageInfo.originH,imageInfo.originW ,3 },torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat32));*/

    cv::Mat img_float;
    img.convertTo(img_float,CV_32FC3);
    sgLogger->debug("setInputTensorCuda convertTo: {} ms",tt.toc_then_tic());
    input_tensor = torch::from_blob(img_float.data, { imageInfo.originH,imageInfo.originW ,3 }, torch::kFloat32).to(torch::kCUDA);


    sgLogger->debug("setInputTensorCuda from_blob:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    ///bgr->rgb
    input_tensor = torch::cat({
        input_tensor.index({"...",2}).unsqueeze(2),
        input_tensor.index({"...",1}).unsqueeze(2),
        input_tensor.index({"...",0}).unsqueeze(2)
        },2);
    sgLogger->debug("setInputTensorCuda bgr->rgb:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    ///hwc->chw
    input_tensor = input_tensor.permute({2,0,1});
    sgLogger->debug("setInputTensorCuda hwc->chw:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    ///norm
    static torch::Tensor mean_t=torch::from_blob(SOLO_IMG_MEAN,{3,1,1},torch::kFloat32).to(torch::kCUDA).expand({3,imageInfo.originH,imageInfo.originW});
    static torch::Tensor std_t=torch::from_blob(SOLO_IMG_STD,{3,1,1},torch::kFloat32).to(torch::kCUDA).expand({3,imageInfo.originH,imageInfo.originW});
    input_tensor = ((input_tensor-mean_t)/std_t);
    sgLogger->debug("setInputTensorCuda norm:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    ///resize
    static auto options=InterpolateFuncOptions().mode(torch::kBilinear).align_corners(true);
    options=options.size(std::vector<int64_t>({imageInfo.rect_h,imageInfo.rect_w}));
    input_tensor = torch::nn::functional::interpolate(input_tensor.unsqueeze(0),options).squeeze(0);
    sgLogger->debug("setInputTensorCuda resize:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    ///拼接图像边缘
    static auto op = torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat32);
    static cv::Scalar mag_color(SOLO_IMG_MEAN[2],SOLO_IMG_MEAN[1],SOLO_IMG_MEAN[0]);
    if (r_h > r_w) { //在图像顶部和下部拼接空白图像
        int cat_w = Config::inputW;
        int cat_h = (Config::inputH-imageInfo.rect_h)/2;
        torch::Tensor cat_t = torch::zeros({3,cat_h,cat_w},op);
        input_tensor = torch::cat({cat_t,input_tensor,cat_t},1);
    } else {
        int cat_w= (Config::inputW-imageInfo.rect_w)/2;
        int cat_h=Config::inputH;
        torch::Tensor cat_t = torch::zeros({3,cat_h,cat_w},op);
        input_tensor = torch::cat({cat_t,input_tensor,cat_t},2);
    }
    sgLogger->debug("setInputTensorCuda cat:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    input_tensor = input_tensor.contiguous();
    sgLogger->debug("setInputTensorCuda contiguous:{} {} ms",dims2str(input_tensor.sizes()),tt.toc_then_tic());

    return input_tensor.data_ptr();
}




cv::Mat Pipeline::processMask(cv::Mat &mask,std::vector<InstInfo> &insts)
{
    cv::Mat rect_img = mask(cv::Rect(imageInfo.rect_x, imageInfo.rect_y, imageInfo.rect_w, imageInfo.rect_h));
    cv::Mat out;
    cv::resize(rect_img, out, cv::Size(imageInfo.originW,imageInfo.originH), 0, 0, cv::INTER_LINEAR);

    ///调整包围框
    float factor_x = out.cols *1.f / rect_img.cols;
    float factor_y = out.rows *1.f / rect_img.rows;
    for(auto &inst : insts){
        inst.min_pt.x -= imageInfo.rect_x;
        inst.min_pt.y -= imageInfo.rect_y;
        inst.max_pt.x -= imageInfo.rect_x;
        inst.max_pt.y -= imageInfo.rect_y;

        inst.min_pt.x *= factor_x;
        inst.min_pt.y *= factor_y;
        inst.max_pt.x *= factor_x;
        inst.max_pt.y *= factor_y;
    }


    return out;
}


void Pipeline::processKitti(cv::Mat &input,cv::Mat &output0,cv::Mat &output1)
{
    int inputH=448;
    int inputW=672;

    ///将输入图片划分为两个图片
    int resize_h = inputH;
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

    imageInfo.originH = img0.rows;
    imageInfo.originW = img0.cols;

    ///将两个半图像进行缩放
    int w, h, x, y;
    float r_w = inputW / (img0.cols*1.0f);
    float r_h = inputH / (img0.rows*1.0f);
    if (r_h > r_w) {
        w = inputW;
        h = r_w * img0.rows;
        x = 0;
        y = (inputH - h) / 2;
    } else {
        w = r_h* img0.cols;
        h = inputH;
        x = (inputW - w) / 2;
        y = 0;
    }
    //将img resize为(INPUT_W,INPUT_H)
    cv::Mat re0(h, w, CV_8UC3);
    cv::resize(img0, re0, re0.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat re1(h, w, CV_8UC3);
    cv::resize(img1, re1, re1.size(), 0, 0, cv::INTER_LINEAR);

    //将图片复制到out中
    output0=cv::Mat(inputH, inputW, CV_8UC3, cv::Scalar(128, 128, 128));
    re0.copyTo(output0(cv::Rect(x, y, re0.cols, re0.rows)));

    output1=cv::Mat(inputH, inputW, CV_8UC3, cv::Scalar(128, 128, 128));
    re1.copyTo(output1(cv::Rect(x, y, re1.cols, re1.rows)));
}


cv::Mat Pipeline::processCut(cv::Mat &img)
{
    int resize_h = Config::inputH;
    float h_factor = resize_h *1.f / img.rows;
    int resize_w = img.cols * h_factor;

    cv::Mat new_img;
    cv::resize(img, new_img, cv::Size(resize_w,resize_h), 0, 0, cv::INTER_LINEAR);

    cout<<new_img.size<<endl;

    cv::Mat out(cv::Size(Config::inputW,Config::inputH),CV_8UC3, cv::Scalar(128, 128, 128));

    out=new_img(cv::Rect(0, 0, Config::inputW, Config::inputH));

    cout<<out.size<<endl;

    imageInfo.originH = out.rows;
    imageInfo.originW = out.cols;

    return out;
}


void Pipeline::setBufferWithNorm(const cv::Mat &img,float *buffer)
{
    //assert(Config::inputH==img.rows);
    //assert(Config::inputW==img.cols);

    int i = 0,b_cnt=0;
    auto rows = std::min(img.rows,Config::inputH);
    auto cols = std::min(img.cols,Config::inputW);
    for (int row = 0; row < rows; ++row) {
        uchar* uc_pixel = img.data + row * img.step;
        for (int col = 0; col < cols; ++col) {
            buffer[b_cnt* 3 * Config::inputH * Config::inputW + i] = (uc_pixel[2]-SOLO_IMG_MEAN[0])/SOLO_IMG_STD[0];
            buffer[b_cnt* 3 * Config::inputH * Config::inputW + i + Config::inputH * Config::inputW] = (uc_pixel[1]-SOLO_IMG_MEAN[1])/SOLO_IMG_STD[1];
            buffer[b_cnt* 3 * Config::inputH * Config::inputW + i + 2 * Config::inputH * Config::inputW] = (uc_pixel[0]-SOLO_IMG_MEAN[2])/SOLO_IMG_STD[2];
            uc_pixel += 3;
            ++i;
        }
    }

}



