/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "detector2d.h"

#include <cstdlib>
#include <iostream>
#include <tuple>
#include <optional>

#include <NvOnnxParser.h>
#include <NvInferPlugin.h>

#include "utils/tensorrt/common.h"
#include "utils/tensorrt/tensorrt_utils.h"

#include "det2d_parameter.h"
#include "utils/parameters.h"
#include "utils/torch_utils.h"

namespace dynamic_vins{\



std::optional<int> GetQueueShapeIndex(int c, int h, int w)
{
    int index=-1;
    for(int i=0;i< (int)det2d_para::kTensorQueueShapes.size(); ++i){
        if(c == det2d_para::kTensorQueueShapes[i][1] && h == det2d_para::kTensorQueueShapes[i][2] && w == det2d_para::kTensorQueueShapes[i][3]){
            index=i;
            break;
        }
    }
    if(index==-1)
        return std::nullopt;
    else
        return index;
}



Detector2D::Detector2D(const std::string& config_path)
{
    ///初始化参数
    det2d_para::SetParameters(config_path);

    if(cfg::slam == SlamType::kRaw || cfg::is_input_seg){
        fmt::print("cfg::slam == SlamType::kRaw || cfg::is_input_seg. So don't need detector\n");
        return;
    }

    ///注册预定义的和自定义的插件
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(),"");
    Infov("start init segmentor");
    std::string model_str;
    if(std::ifstream ifs(det2d_para::kDetectorSerializePath); ifs.is_open()){
        while(ifs.peek() != EOF){
            std::stringstream ss;
            ss<<ifs.rdbuf();
            model_str.append(ss.str());
        }
        ifs.close();
    }
    else{
        throw std::runtime_error(fmt::format("Can not open the kDetectorSerializePath:{}",
                                             det2d_para::kDetectorSerializePath));
    }
    Infov("createInferRuntime");
    ///创建runtime
    runtime_=std::unique_ptr<nvinfer1::IRuntime,InferDeleter>(
            nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    Infov("deserializeCudaEngine");
    ///反序列化模型
    engine_=std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(model_str.data(), model_str.size()) , InferDeleter());
    Infov("createExecutionContext");
    ///创建执行上下文
    context_=std::unique_ptr<nvinfer1::IExecutionContext,InferDeleter>(
            engine_->createExecutionContext());
    if(!context_){
        throw std::runtime_error("can not create context");
    }

    ///创建输入输出的内存
    buffer = std::make_shared<MyBuffer>(*engine_);

    det2d_para::model_input_height=buffer->dims[0].d[2];
    det2d_para::model_input_width=buffer->dims[0].d[3];
    det2d_para::model_input_channel=3;

    pipeline_=std::make_shared<Pipeline>();

    solo_ = std::make_shared<Solov2>();

/*    //cv::Mat warn_up_input(cv::Size(1226,370),CV_8UC3,cv::Scalar(128));
    const std::string warn_up_path= seg_para::kWarnUpImagePath ;
    cv::Mat warn_up_input = cv::imread(warn_up_path);
    if(warn_up_input.empty()){
        Errorv("Can not open warn up image:{}", warn_up_path);
        return;
    }
    cv::resize(warn_up_input,warn_up_input,cv::Size(seg_para::model_input_width, seg_para::model_input_height));
    Warnv("warn up model");
    //[[maybe_unused]] auto result = forward(warn_up_input);
    [[maybe_unused]] torch::Tensor mask_tensor;
    [[maybe_unused]] std::vector<InstInfo> insts_info;
    ForwardTensor(warn_up_input, mask_tensor, insts_info);
    if(insts_info.empty())
        throw std::runtime_error("model not init");*/


    Infov("infer init finished");
}

std::tuple<std::vector<cv::Mat>,std::vector<Box2D::Ptr>>
Detector2D::Forward(cv::Mat &img)
{
/*    TicToc ticToc,tt;
    //cv::Mat input=pipeline_->ProcessPad(img);
    cv::Mat input= pipeline_->ProcessPadCuda(img);

    ///将图片数据复制到输入buffer,同时实现了图像的归一化
    pipeline_->SetBufferWithNorm(input, buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();

    tt.TocPrintTic("prepare:");

    ///推断
    context_->enqueue(kBatchSize, buffer->gpu_buffer, buffer->stream, nullptr);

    buffer->cpyOutputToCPU();
    tt.TocPrintTic("enqueue:");

    std::vector<torch::Tensor> outputs(kTensorQueueShapes.size());
    auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);
    for(int i=1;i<buffer->binding_num;++i){
        torch::Tensor tensor=torch::from_blob(
                buffer->gpu_buffer[i],
                {buffer->dims[i].d[0],buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]},
                opt);
        std::optional<int> index = GetQueueShapeIndex(buffer->dims[i].d[1], buffer->dims[i].d[2], buffer->dims[i].d[3]);
        if(index){
            outputs[*index] = tensor.to(torch::kCUDA);
        }
        else{
            cerr<<"GetQueueShapeIndex failed:"<<buffer->dims[i]<<endl;
            std::terminate();
        }
    }
    tt.TocPrintTic("push_back");
    tt.TocPrintTic("push_back");

    //cv::Mat mask_img=solo_->GetSingleSeg(outputs,torch::kCUDA,insts);
    //auto [masks,insts] = solo_->GetSingleSeg(outputs,pipeline_->image_info);
    torch::Tensor mask_tensor;
    std::vector<InstInfo> insts_info;
    solo_->GetSegTensor(outputs, pipeline_->image_info, mask_tensor, insts_info);

    std::vector<cv::Mat> mask_v;
    if(!insts_info.empty()){
        auto merger_tensor = mask_tensor.sum(0).to(torch::kInt8) * 255;
        merger_tensor = merger_tensor.to(torch::kCPU);
        auto mask = cv::Mat(cv::Size(merger_tensor.sizes()[1],merger_tensor.sizes()[0]), CV_8UC1, merger_tensor.data_ptr()).clone();
        mask_v.push_back(mask);
    }

    tt.TocPrintTic("GetSingleSeg:");

    infer_time_ = ticToc.Toc();
    //ticToc.toc_print_tic("kitti time:");
    //VisualizeResult(img,mask_img,insts);

    return {mask_v,insts_info};*/
}

/**
 * 前向传播图像张量,完成实例分割
 * @param img
 * @param mask_tensor
 * @param insts
 */
void Detector2D::ForwardTensor(torch::Tensor &img, torch::Tensor &mask_tensor,
                               std::vector<Box2D::Ptr> &insts){

    Debugs("ForwardTensor | img_tensor.shape:{}", DimsToStr(img.sizes()));
    buffer->gpu_buffer[0] = pipeline_->ProcessInput(img);
    Debugs("ForwardTensor | input_tensor.shape:{}", DimsToStr(pipeline_->input_tensor.sizes()));
    ///推断
    context_->enqueue(det2d_para::kBatchSize, buffer->gpu_buffer, buffer->stream, nullptr);

    std::vector<torch::Tensor> outputs(det2d_para::kTensorQueueShapes.size());
    auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);
    for(int i=1;i<buffer->binding_num;++i){
        torch::Tensor tensor=torch::from_blob(
                buffer->gpu_buffer[i],
                {buffer->dims[i].d[0],buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]},
                opt);
        std::optional<int> index = GetQueueShapeIndex(buffer->dims[i].d[1], buffer->dims[i].d[2], buffer->dims[i].d[3]);
        if(index){
            outputs[*index] = tensor.to(torch::kCUDA);
        }
        else{
            auto msg=fmt::format("GetQueueShapeIndex failed:({},{},{},{})",buffer->dims[i].d[0],buffer->dims[i].d[1],
                                 buffer->dims[i].d[2],buffer->dims[i].d[3]);
            Errors(msg);
            throw std::runtime_error(msg);
        }
    }

    solo_->GetSegTensor(outputs, pipeline_->image_info, mask_tensor, insts);
}


void Detector2D::ForwardTensor(cv::Mat &img, torch::Tensor &mask_tensor, std::vector<Box2D::Ptr> &insts)
{
    ///将图片数据复制到输入buffer,同时实现了图像的归一化
    /*
    //方式1
     cv::Mat input=pipeline_->ProcessPad(img);
    //cv::Mat input=pipeline_->ProcessPadCuda(img);
    pipeline_->SetBufferWithNorm(input,buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();*/
    /*
    //方式2
     cv::Mat input=pipeline_->ProcessPad(img);
    pipeline_->setBufferWithTensor(input);
    buffer->gpu_buffer[0] = pipeline_->input_tensor.data_ptr();*/
    //方式3
    //buffer->gpu_buffer[0] = pipeline_->SetInputTensor(img);
    //buffer->gpu_buffer[0] = pipeline_->ProcessInput(img);
    //方式4 不做pad
    /*pipeline_->SetBufferWithNorm(img,buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();
    solo_->is_resized_ = false;
    pipeline_->image_info.rect_x=0;
    pipeline_->image_info.rect_y=0;
    pipeline_->image_info.rect_w= std::min(para::kInputWidth,img.cols) ;
    pipeline_->image_info.rect_h= std::min(para::kInputHeight,img.rows) ;*/
    /*cv::Mat input=pipeline_->ProcessPadCuda(img);
    pipeline_->SetBufferWithNorm(input,buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();*/

    auto tensor = Pipeline::ImageToTensor(img);
    ForwardTensor(tensor,mask_tensor,insts);
}



void Detector2D::ForwardTensor(cv::cuda::GpuMat &img, torch::Tensor &mask_tensor, std::vector<Box2D::Ptr> &insts)
{
/*    TicToc tic_toc,tt;

    cv::Mat input= pipeline_->ProcessPadCuda(img);
    pipeline_->SetBufferWithNorm(input, buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();

    Infos("ForwardTensor prepare:{} ms", tt.TocThenTic());

    ///推断
    context_->enqueue(kBatchSize, buffer->gpu_buffer, buffer->stream, nullptr);

    Infos("ForwardTensor enqueue:{} ms", tt.TocThenTic());

    std::vector<torch::Tensor> outputs(kTensorQueueShapes.size());

    auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);
    for(int i=1;i<buffer->binding_num;++i){
        torch::Tensor tensor=torch::from_blob(
                buffer->gpu_buffer[i],
                {buffer->dims[i].d[0],buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]},
                opt);
        std::optional<int> index = GetQueueShapeIndex(buffer->dims[i].d[1], buffer->dims[i].d[2], buffer->dims[i].d[3]);
        if(index){
            outputs[*index] = tensor.to(torch::kCUDA);
        }
        else{
            cerr<<"GetQueueShapeIndex failed:"<<buffer->dims[i]<<endl;
            std::terminate();
        }
        //cout<<index<<" ("<<buffer->dims[i].d[1]<<buffer->dims[i].d[2]<<buffer->dims[i].d[3]<<")"<<endl;
    }

    Infos("ForwardTensor push_back:{} ms", tt.TocThenTic());

    solo_->GetSegTensor(outputs, pipeline_->image_info, mask_tensor, insts);
    Infos("ForwardTensor GetSegTensor:{} ms", tt.TocThenTic());
    infer_time_ = tic_toc.Toc();*/

}




void Detector2D::VisualizeResult(cv::Mat &input, cv::Mat &mask, std::vector<Box2D::Ptr> &insts)
{
/*    if(mask.empty()){
        cv::imshow("test",input);
        cv::waitKey(1);
    }
    else{
        cout<<mask.size<<endl;
        mask = pipeline_->ProcessMask(mask, insts);

        cv::Mat image_test;
        cv::add(input,mask,image_test);
        for(auto &inst : insts){
            if(inst.prob < 0.2)
                continue;
            inst.name = Config::CocoLabelVector[inst.label_id + 1];
            cv::Point2i center = (inst.min_pt + inst.max_pt)/2;
            std::string show_text = fmt::format("{} {:.2f}",inst.name,inst.prob);
            cv::putText(image_test,show_text,center,CV_FONT_HERSHEY_SIMPLEX,0.8,
                        cv::Scalar(255,0,0),2);
            cv::rectangle(image_test, inst.min_pt, inst.max_pt, cv::Scalar(255, 0, 0), 2);
        }
        cv::putText(image_test, fmt::format("{:.2f} ms", infer_time_), cv::Point2i(20, 20),
                    CV_FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 255, 255));

        cv::imshow("test",image_test);
        cv::waitKey(1);
    }*/
}

}
