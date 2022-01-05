/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <iostream>
#include <memory>

#include <torch/torch.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "tensorrt_utils.h"
#include "common.h"


namespace dynamic_vins{\


int BuildTensorRT(const string &onnx_path,const string &tensorrt_path)
{
    cout<<"createInferBuilder"<<endl;

    ///创建builder
    auto builder=std::unique_ptr<nvinfer1::IBuilder,InferDeleter>(
            nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if(!builder)
        return -1;

    ///创建网络定义
    cout<<"createNetwork"<<endl;
    uint32_t flag=1U<<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network=std::unique_ptr<nvinfer1::INetworkDefinition,InferDeleter>(
            builder->createNetworkV2(flag));
    if(!network)
        return -1;

    cout<<"createBuilderConfig"<<endl;
    auto config=std::unique_ptr<nvinfer1::IBuilderConfig,InferDeleter>(
            builder->createBuilderConfig());
    if(!config)
        return -1;

    ///创建parser
    cout<<"createParser"<<endl;
    auto parser=std::unique_ptr<nvonnxparser::IParser,InferDeleter>(
            nvonnxparser::createParser(*network,sample::gLogger.getTRTLogger()));
    if(!parser)
        return -1;

    ///读取模型文件

    cout<<"parseFromFile:"<<onnx_path<<endl;
    auto verbosity=sample::gLogger.getReportableSeverity();
    auto parsed=parser->parseFromFile(onnx_path.c_str(),static_cast<int>(verbosity));
    if(!parsed)
        return -1;

    //设置层工作空间大小
    config->setMaxWorkspaceSize(1_GiB);
    //使用FP16精度
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    cout<<"input shape:"<<network->getInput(0)->getName()<<" "<<network->getInput(0)->getDimensions()<<endl;
    cout<<"output shape:"<<network->getOutput(0)->getName()<<" "<<network->getOutput(0)->getDimensions()<<endl;

    cout<<"enableDLA"<<endl;

    ///DLA
    const int useDLACore=-1;
    samplesCommon::enableDLA(builder.get(),config.get(),useDLACore);

    ///构建engine
    cout<<"buildEngineWithConfig"<<endl;
    auto engine=std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network,*config),InferDeleter());

    if(!engine)
        return -1;

    cout<<"serializeModel"<<endl;
    auto serializeModel=engine->serialize();

    //将序列化模型拷贝到字符串
    std::string serialize_str;
    serialize_str.resize(serializeModel->size());
    memcpy((void*)serialize_str.data(),serializeModel->data(),serializeModel->size());
    //将字符串输出到文件中
    std::ofstream serialize_stream(tensorrt_path);
    serialize_stream<<serialize_str;
    serialize_stream.close();

    cout<<"done"<<endl;

    return 0;
}



}