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
#include "utils/dataset/coco_utils.h"
#include "utils/dataset/kitti_utils.h"

namespace dynamic_vins{\



std::optional<int> GetQueueShapeIndex(int c, int h, int w)
{
    int index=-1;
    for(int i=0;i< (int)det2d_para::kTensorQueueShapes.size(); ++i){
        if(c == det2d_para::kTensorQueueShapes[i][1] && h == det2d_para::kTensorQueueShapes[i][2]
        && w == det2d_para::kTensorQueueShapes[i][3]){
            index=i;
            break;
        }
    }
    if(index==-1)
        return std::nullopt;
    else
        return index;
}



/**
 * 根据实例分割的mask构建box2d
 * @param seg_label 实例分割mask,大小:[num_mask, cols, rows]
 * @param cate_label 类别标签:[num_mask]
 * @param cate_score 类别分数
 * @return
 */
vector<Box2D::Ptr> BuildBoxes2D(torch::Tensor &seg_label,torch::Tensor &cate_label,torch::Tensor &cate_score,
                                double height_threshold){
    vector<Box2D::Ptr> insts;

    ///根据mask计算包围框
    for(int i=0;i<seg_label.sizes()[0];++i){
        auto nz=seg_label[i].nonzero();
        auto max_xy =std::get<0>( torch::max(nz,0) );
        auto min_xy =std::get<0>( torch::min(nz,0) );

        cv::Point2f max_pt(max_xy[1].item().toInt(),max_xy[0].item().toInt());
        cv::Point2f min_pt(min_xy[1].item().toInt(),min_xy[0].item().toInt());

        if(max_pt.y - min_pt.y < height_threshold){
            //删除对应的mask
            ///TODO
            continue;
        }

        Box2D::Ptr inst = std::make_shared<Box2D>();
        inst->id = i;

        int coco_id = cate_label[i].item().toInt();
        string coco_name = coco::CocoLabel[coco_id];
        if(auto it=coco::CocoToKitti.find(coco_name);it!=coco::CocoToKitti.end()){
            string kitti_name = *(it->second.begin());
            int kitti_id = kitti::GetKittiLabelIndex(kitti_name);
            inst->class_id =kitti_id;
            inst->class_name = kitti_name;
        }
        else{
            inst->class_id =coco_id;
            inst->class_name = coco_name;
        }

        inst->max_pt = max_pt;
        inst->min_pt = min_pt;
        inst->rect = cv::Rect2f(inst->min_pt,inst->max_pt);

        inst->score = cate_score[i].item().toFloat();

        insts.push_back(inst);
    }

    return insts;
}




Detector2D::Detector2D(const std::string& config_path)
{
    ///初始化参数
    det2d_para::SetParameters(config_path);

    if(!(cfg::slam == SLAM::kNaive || cfg::slam == SLAM::kDynamic) || cfg::is_input_seg){
        fmt::print("cfg::slam == SlamType::kRaw || cfg::is_input_seg. So don't need detector\n");
        return;
    }
    if(det2d_para::use_offline){
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

    ///SOLO的后处理
    torch::Tensor cate_labels,cate_scores;
    solo_->GetSegTensor(outputs, pipeline_->image_info, mask_tensor, cate_labels,cate_scores);

    ///构造Box2D
    insts = BuildBoxes2D(mask_tensor,cate_labels,cate_scores,det2d_para::kBoxMinHeight);

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




torch::Tensor Detector2D::LoadTensor(const string &load_path){

    std::ifstream input(load_path, std::ios::binary);
    if(!input.is_open()){
        string msg=fmt::format("Detector2D::LoadTensor failed in:{}",load_path);
        Errors(msg);
        std::cerr<<msg<<std::endl;
        return {};
    }

    std::vector<char> bytes( (std::istreambuf_iterator<char>(input)),
                             (std::istreambuf_iterator<char>()));
    input.close();

    torch::IValue x = torch::pickle_load(bytes);
    torch::Tensor tensor = x.toTensor();
    return tensor;
}





void Detector2D::Launch(SemanticImage &img){

    if(det2d_para::use_offline){
        image_seq_id = img.seq;

        std::string seq_str = PadNumber(image_seq_id,6);

        torch::Tensor seg_label = LoadTensor(det2d_para::kDet2dPreprocessPath +
                fmt::format("seg_label_{}.pt",seq_str));
        torch::Tensor cate_score = LoadTensor(det2d_para::kDet2dPreprocessPath +
                fmt::format("cate_score_{}.pt",seq_str));
        torch::Tensor cate_label = LoadTensor(det2d_para::kDet2dPreprocessPath +
                fmt::format("cate_label_{}.pt",seq_str));

        if(! seg_label.defined()){
            return;
        }

        seg_label=seg_label.to(torch::kCUDA);
        cate_score=cate_score.to(torch::kCUDA);
        cate_label=cate_label.to(torch::kCUDA);

        seg_label = seg_label > det2d_para::kSoloMaskThr;

        ///根据mask计算包围框
        img.boxes2d = BuildBoxes2D(seg_label,cate_label,cate_score,det2d_para::kBoxMinHeight);
        img.mask_tensor = seg_label;
    }
    else{
        ForwardTensor(img.img_tensor, img.mask_tensor, img.boxes2d);
    }

}











}
