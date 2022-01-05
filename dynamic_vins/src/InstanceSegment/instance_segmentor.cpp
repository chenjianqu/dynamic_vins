/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <cstdlib>
#include <iostream>
#include <tuple>

#include <NvOnnxParser.h>
#include <NvInferPlugin.h>

#include "instance_segmentor.h"
#include "parameters.h"
#include "utils.h"

namespace dynamic_vins{\



std::optional<int> GetQueueShapeIndex(int c, int h, int w)
{
    int index=-1;
    for(int i=0;i< (int)kTensorQueueShapes.size(); ++i){
        if(c == kTensorQueueShapes[i][1] && h == kTensorQueueShapes[i][2] && w == kTensorQueueShapes[i][3]){
            index=i;
            break;
        }
    }
    if(index==-1)
        return std::nullopt;
    else
        return index;
}



InstanceSegmentor::InstanceSegmentor()
{
    if(Config::is_input_seg || Config::slam == SlamType::kRaw){
        auto msg="set input seg, the segmentor does not initial";
        WarnV(msg);
        cerr<<msg<<endl;
        return;
    }

    ///注册预定义的和自定义的插件
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(),"");
    InfoV("读取模型文件");
    std::string model_str;
    if(std::ifstream ifs(Config::kDetectorSerializePath);ifs.is_open()){
        while(ifs.peek() != EOF){
            std::stringstream ss;
            ss<<ifs.rdbuf();
            model_str.append(ss.str());
        }
        ifs.close();
    }
    else{
        auto msg=fmt::format("Can not open the kDetectorSerializePath:{}",Config::kDetectorSerializePath);
        vio_logger->critical(msg);
        throw std::runtime_error(msg);
    }

    InfoV("createInferRuntime");

    ///创建runtime
    runtime_=std::unique_ptr<nvinfer1::IRuntime,InferDeleter>(
            nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));

    InfoV("deserializeCudaEngine");

    ///反序列化模型
    engine_=std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(model_str.data(), model_str.size()) , InferDeleter());

    InfoV("createExecutionContext");

    ///创建执行上下文
    context_=std::unique_ptr<nvinfer1::IExecutionContext,InferDeleter>(
            engine_->createExecutionContext());

    if(!context_){
        throw std::runtime_error("can not create context");
    }

    ///创建输入输出的内存
    buffer = std::make_shared<MyBuffer>(*engine_);

    Config::kInputHeight=buffer->dims[0].d[2];
    Config::kInputWidth=buffer->dims[0].d[3];
    Config::kInputChannel=3;

    pipeline_=std::make_shared<Pipeline>();
    solo_ = std::make_shared<Solov2>();


    //cv::Mat warn_up_input(cv::Size(1226,370),CV_8UC3,cv::Scalar(128));
    const std::string warn_up_path="/home/chen/ws/vio_ws/src/dynamic_vins/config/kitti.png";
    cv::Mat warn_up_input = cv::imread(warn_up_path);

    if(warn_up_input.empty()){
        vio_logger->error("Can not open warn up image:{}", warn_up_path);
        return;
    }

    cv::resize(warn_up_input,warn_up_input,cv::Size(Config::kCol, Config::kRow));

    WarnV("warn up model");

    //[[maybe_unused]] auto result = forward(warn_up_input);

    [[maybe_unused]] torch::Tensor mask_tensor;
    [[maybe_unused]] std::vector<InstInfo> insts_info;
    ForwardTensor(warn_up_input, mask_tensor, insts_info);

    if(insts_info.empty()){
        throw std::runtime_error("model not init");
    }

    InfoV("infer init finished");
}






std::tuple<std::vector<cv::Mat>,std::vector<InstInfo>>
InstanceSegmentor::Forward(cv::Mat &img)
{
    TicToc ticToc,tt;

    //cv::Mat input=pipeline_->ProcessPad(img);
    cv::Mat input= pipeline_->ProcessPadCuda(img);

    ///将图片数据复制到输入buffer,同时实现了图像的归一化
    pipeline_->SetBufferWithNorm(input, buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();

    tt.toc_print_tic("prepare:");

    ///推断
    context_->enqueue(kBatchSize, buffer->gpu_buffer, buffer->stream, nullptr);

    buffer->cpyOutputToCPU();

    tt.toc_print_tic("enqueue:");


    /*std::vector<torch::Tensor> outputs;
    static auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);
    for(int i=1;i<buffer->binding_num;++i){
        torch::Tensor tensor=torch::from_blob(
                buffer->gpu_buffer[i],
                {buffer->dims[i].d[0],buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]},
                opt);
        outputs.push_back(tensor);
    }*/

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
    tt.toc_print_tic("push_back");
    cout<<endl;




    tt.toc_print_tic("push_back");

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

    tt.toc_print_tic("GetSingleSeg:");

    infer_time_ = ticToc.toc();
    //ticToc.toc_print_tic("kitti time:");
    //VisualizeResult(img,mask_img,insts);

    return {mask_v,insts_info};
}

void InstanceSegmentor::ForwardTensor(torch::Tensor &img, torch::Tensor &mask_tensor, std::vector<InstInfo> &insts){
    buffer->gpu_buffer[0] = pipeline_->ProcessInput(img);
    ///推断
    context_->enqueue(kBatchSize, buffer->gpu_buffer, buffer->stream, nullptr);

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
            auto msg=fmt::format("GetQueueShapeIndex failed:({},{},{},{})",buffer->dims[i].d[0],buffer->dims[i].d[1],
                                 buffer->dims[i].d[2],buffer->dims[i].d[3]);
            sg_logger->error(msg);
            throw std::runtime_error(msg);
        }
    }

    solo_->GetSegTensor(outputs, pipeline_->image_info, mask_tensor, insts);
}


void InstanceSegmentor::ForwardTensor(cv::Mat &img, torch::Tensor &mask_tensor, std::vector<InstInfo> &insts)
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
    pipeline_->image_info.rect_w= std::min(Config::kInputWidth,img.cols) ;
    pipeline_->image_info.rect_h= std::min(Config::kInputHeight,img.rows) ;*/
    /*cv::Mat input=pipeline_->ProcessPadCuda(img);
    pipeline_->SetBufferWithNorm(input,buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();*/

    auto tensor = Pipeline::ImageToTensor(img);
    ForwardTensor(tensor,mask_tensor,insts);
}



void InstanceSegmentor::ForwardTensor(cv::cuda::GpuMat &img, torch::Tensor &mask_tensor, std::vector<InstInfo> &insts)
{
    TicToc tic_toc,tt;

    cv::Mat input= pipeline_->ProcessPadCuda(img);
    pipeline_->SetBufferWithNorm(input, buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();

    InfoS("ForwardTensor prepare:{} ms", tt.toc_then_tic());

    ///推断
    context_->enqueue(kBatchSize, buffer->gpu_buffer, buffer->stream, nullptr);

    InfoS("ForwardTensor enqueue:{} ms", tt.toc_then_tic());

    std::vector<torch::Tensor> outputs(kTensorQueueShapes.size());


    //方法1
    /*buffer->cpyOutputToCPU();
    for(int i=1;i<buffer->binding_num;++i){
        torch::Tensor tensor=torch::from_blob(
                buffer->cpu_buffer[i],
                {buffer->dims[i].d[0],buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]},
                torch::kFloat);
        outputs.push_back(tensor.to(torch::kCUDA));
        //cout<<tensor.sizes()<<endl;
    }*/

    //方法2
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

    InfoS("ForwardTensor push_back:{} ms", tt.toc_then_tic());

    solo_->GetSegTensor(outputs, pipeline_->image_info, mask_tensor, insts);

    InfoS("ForwardTensor GetSegTensor:{} ms", tt.toc_then_tic());

    infer_time_ = tic_toc.toc();

}




void InstanceSegmentor::VisualizeResult(cv::Mat &input, cv::Mat &mask, std::vector<InstInfo> &insts)
{
    if(mask.empty()){
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
    }
}

}
