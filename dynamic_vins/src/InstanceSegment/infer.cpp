//
// Created by chen on 2021/11/7.
//
#include <cstdlib>
#include <iostream>
#include <tuple>

#include "infer.h"

#include "../parameters.h"
#include "../utils.h"

#include <NvOnnxParser.h>
#include <NvInferPlugin.h>


using namespace std;


std::optional<int> getQueueShapeIndex(int c,int h,int w)
{
    int index=-1;
    for(int i=0;i< (int)TENSOR_QUEUE_SHAPE.size();++i){
        if(c==TENSOR_QUEUE_SHAPE[i][1] && h==TENSOR_QUEUE_SHAPE[i][2] && w==TENSOR_QUEUE_SHAPE[i][3]){
            index=i;
            break;
        }
    }
    if(index==-1)
        return std::nullopt;
    else
        return index;
}



Infer::Infer()
{
    if(Config::isInputSeg || Config::SLAM == SlamType::RAW){
        auto msg="set input seg, the segmentor does not initial";
        vioLogger->warn(msg);
        cerr<<msg<<endl;
        return;
    }

    ///注册预定义的和自定义的插件
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(),"");
    info_v("读取模型文件");
    std::string model_str;
    if(std::ifstream ifs(Config::DETECTOR_SERIALIZE_PATH);ifs.is_open()){
        while(ifs.peek() != EOF){
            std::stringstream ss;
            ss<<ifs.rdbuf();
            model_str.append(ss.str());
        }
        ifs.close();
    }
    else{
        auto msg=fmt::format("Can not open the DETECTOR_SERIALIZE_PATH:{}",Config::DETECTOR_SERIALIZE_PATH);
        vioLogger->critical(msg);
        throw std::runtime_error(msg);
    }

    info_v("createInferRuntime");

    ///创建runtime
    runtime=std::unique_ptr<nvinfer1::IRuntime,InferDeleter>(
            nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));

    info_v("deserializeCudaEngine");

    ///反序列化模型
    engine=std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(model_str.data(),model_str.size()) ,InferDeleter());

    info_v("createExecutionContext");

    ///创建执行上下文
    context=std::unique_ptr<nvinfer1::IExecutionContext,InferDeleter>(
            engine->createExecutionContext());

    if(!context){
        throw std::runtime_error("can not create context");
    }

    ///创建输入输出的内存
    buffer = std::make_shared<MyBuffer>(*engine);

    Config::inputH=buffer->dims[0].d[2];
    Config::inputW=buffer->dims[0].d[3];
    Config::inputC=3;

    pipeline=std::make_shared<Pipeline>();
    solo = std::make_shared<Solov2>();


    //cv::Mat warn_up_input(cv::Size(1226,370),CV_8UC3,cv::Scalar(128));
    const std::string warn_up_path="/home/chen/ws/vio_ws/src/dynamic_vins/config/kitti.png";
    cv::Mat warn_up_input = cv::imread(warn_up_path);

    if(warn_up_input.empty()){
        vioLogger->error("Can not open warn up image:{}", warn_up_path);
        return;
    }

    cv::resize(warn_up_input,warn_up_input,cv::Size(Config::COL,Config::ROW));

    vioLogger->warn("warn up model");

    //[[maybe_unused]] auto result = forward(warn_up_input);

    [[maybe_unused]] torch::Tensor mask_tensor;
    [[maybe_unused]] std::vector<InstInfo> insts_info;
    forward_tensor(warn_up_input,mask_tensor,insts_info);

    if(insts_info.empty()){
        throw std::runtime_error("model not init");
    }

    info_v("infer init finished");
}






std::tuple<std::vector<cv::Mat>,std::vector<InstInfo>>
Infer::forward(cv::Mat &img)
{
    TicToc ticToc,tt;

    //cv::Mat input=pipeline->processPad(img);
    cv::Mat input=pipeline->processPadCuda(img);

    ///将图片数据复制到输入buffer,同时实现了图像的归一化
    pipeline->setBufferWithNorm(input,buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();

    tt.toc_print_tic("prepare:");

    ///推断
    context->enqueue(BATCH_SIZE, buffer->gpu_buffer, buffer->stream, nullptr);

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

    std::vector<torch::Tensor> outputs(TENSOR_QUEUE_SHAPE.size());
    auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);
    for(int i=1;i<buffer->binding_num;++i){
        torch::Tensor tensor=torch::from_blob(
                buffer->gpu_buffer[i],
                {buffer->dims[i].d[0],buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]},
                opt);
        std::optional<int> index = getQueueShapeIndex(buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]);
        if(index){
            outputs[*index] = tensor.to(torch::kCUDA);
        }
        else{
            cerr<<"getQueueShapeIndex failed:"<<buffer->dims[i]<<endl;
            std::terminate();
        }
    }
    tt.toc_print_tic("push_back");
    cout<<endl;




    tt.toc_print_tic("push_back");

    //cv::Mat mask_img=solo->getSingleSeg(outputs,torch::kCUDA,insts);
    //auto [masks,insts] = solo->getSingleSeg(outputs,pipeline->imageInfo);
    torch::Tensor mask_tensor;
    std::vector<InstInfo> insts_info;
    solo->getSegTensor(outputs,pipeline->imageInfo,mask_tensor,insts_info);

    std::vector<cv::Mat> mask_v;
    if(!insts_info.empty()){
        auto merger_tensor = mask_tensor.sum(0).to(torch::kInt8) * 255;
        merger_tensor = merger_tensor.to(torch::kCPU);
        auto mask = cv::Mat(cv::Size(merger_tensor.sizes()[1],merger_tensor.sizes()[0]), CV_8UC1, merger_tensor.data_ptr()).clone();
        mask_v.push_back(mask);
    }

    tt.toc_print_tic("getSingleSeg:");

    infer_time = ticToc.toc();
    //ticToc.toc_print_tic("kitti time:");
    //visualizeResult(img,mask_img,insts);

    return {mask_v,insts_info};
}


void Infer::forward_tensor(cv::Mat &img,torch::Tensor &mask_tensor,std::vector<InstInfo> &insts)
{
    TicToc ticToc,tt;

    ///将图片数据复制到输入buffer,同时实现了图像的归一化
    //方式1
    /*cv::Mat input=pipeline->processPad(img);
    //cv::Mat input=pipeline->processPadCuda(img);
    pipeline->setBufferWithNorm(input,buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();*/

    //方式2
    /*cv::Mat input=pipeline->processPad(img);
    pipeline->setBufferWithTensor(input);
    buffer->gpu_buffer[0] = pipeline->input_tensor.data_ptr();*/

    //方式3
    //buffer->gpu_buffer[0] = pipeline->setInputTensor(img);
    buffer->gpu_buffer[0] = pipeline->setInputTensorCuda(img);

    //方式4 不做pad
    /*pipeline->setBufferWithNorm(img,buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();
    solo->isResized = false;
    pipeline->imageInfo.rect_x=0;
    pipeline->imageInfo.rect_y=0;
    pipeline->imageInfo.rect_w= std::min(Config::inputW,img.cols) ;
    pipeline->imageInfo.rect_h= std::min(Config::inputH,img.rows) ;*/

    /*cv::Mat input=pipeline->processPadCuda(img);
    pipeline->setBufferWithNorm(input,buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();*/

    info_s(fmt::format("forward_tensor prepare:{} ms",tt.toc_then_tic()));

    ///推断
    context->enqueue(BATCH_SIZE, buffer->gpu_buffer, buffer->stream, nullptr);

    info_s(fmt::format("forward_tensor enqueue:{} ms",tt.toc_then_tic()));

    std::vector<torch::Tensor> outputs(TENSOR_QUEUE_SHAPE.size());


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
        std::optional<int> index = getQueueShapeIndex(buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]);
        if(index){
            outputs[*index] = tensor.to(torch::kCUDA);
        }
        else{
            auto msg=fmt::format("getQueueShapeIndex failed:({},{},{},{})",buffer->dims[i].d[0],buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]);
            sgLogger->error(msg);
            throw std::runtime_error(msg);
        }
        //cout<<index<<" ("<<buffer->dims[i].d[1]<<buffer->dims[i].d[2]<<buffer->dims[i].d[3]<<")"<<endl;
    }

    info_s("forward_tensor push_back:{} ms",tt.toc_then_tic());

    solo->getSegTensor(outputs,pipeline->imageInfo,mask_tensor,insts);

    info_s("forward_tensor getSegTensor:{} ms",tt.toc_then_tic());
    info_s("forward_tensor inst number:{}",insts.size());
    

    infer_time = ticToc.toc();

}



void Infer::forward_tensor(cv::cuda::GpuMat &img,torch::Tensor &mask_tensor,std::vector<InstInfo> &insts)
{
    TicToc ticToc,tt;

    cv::Mat input=pipeline->processPadCuda(img);
    pipeline->setBufferWithNorm(input,buffer->cpu_buffer[0]);
    buffer->cpyInputToGPU();

    info_s("forward_tensor prepare:{} ms",tt.toc_then_tic());

    ///推断
    context->enqueue(BATCH_SIZE, buffer->gpu_buffer, buffer->stream, nullptr);

    info_s("forward_tensor enqueue:{} ms",tt.toc_then_tic());

    std::vector<torch::Tensor> outputs( TENSOR_QUEUE_SHAPE.size());


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
        std::optional<int> index =getQueueShapeIndex(buffer->dims[i].d[1],buffer->dims[i].d[2],buffer->dims[i].d[3]);
        if(index){
            outputs[*index] = tensor.to(torch::kCUDA);
        }
        else{
            cerr<<"getQueueShapeIndex failed:"<<buffer->dims[i]<<endl;
            std::terminate();
        }
        //cout<<index<<" ("<<buffer->dims[i].d[1]<<buffer->dims[i].d[2]<<buffer->dims[i].d[3]<<")"<<endl;
    }

    info_s("forward_tensor push_back:{} ms",tt.toc_then_tic());

    solo->getSegTensor(outputs,pipeline->imageInfo,mask_tensor,insts);

    info_s("forward_tensor getSegTensor:{} ms",tt.toc_then_tic());

    infer_time = ticToc.toc();

}




void Infer::visualizeResult(cv::Mat &input,cv::Mat &mask,std::vector<InstInfo> &insts)
{
    if(mask.empty()){
        cv::imshow("test",input);
        cv::waitKey(1);
    }
    else{
        cout<<mask.size<<endl;
        mask = pipeline->processMask(mask,insts);

        cv::Mat image_test;
        cv::add(input,mask,image_test);
        for(auto &inst : insts){
            if(inst.prob < 0.2)
                continue;
            inst.name = CocoLabelMap[inst.label_id + 1];
            cv::Point2i center = (inst.min_pt + inst.max_pt)/2;
            std::string show_text = fmt::format("{} {:.2f}",inst.name,inst.prob);
            cv::putText(image_test,show_text,center,CV_FONT_HERSHEY_SIMPLEX,0.8,cv::Scalar(255,0,0),2);
            cv::rectangle(image_test, inst.min_pt, inst.max_pt, cv::Scalar(255, 0, 0), 2);
        }
        cv::putText(image_test,fmt::format("{:.2f} ms",infer_time),cv::Point2i(20,20),CV_FONT_HERSHEY_SIMPLEX,2,cv::Scalar(0,255,255));

        cv::imshow("test",image_test);
        cv::waitKey(1);
    }
}
