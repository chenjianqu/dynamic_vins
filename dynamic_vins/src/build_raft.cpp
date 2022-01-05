/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_CPP. Created by chen on 2021/12/24.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <iostream>
#include <memory>

#include <torch/torch.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include "TensorRT/tensorrt_utils.h"
#include "parameters.h"
#include "utils.h"

using namespace std;
namespace dv = dynamic_vins;

int main(int argc, char **argv)
{
    if(argc != 2){
        cerr<<"please input: [config file]"<<endl;
        return 1;
    }
    string config_file = argv[1];
    fmt::print("config_file:{}\n",argv[1]);

    try{
        dv::Config cfg(config_file);
    }
    catch(std::runtime_error &e){
        dv::CriticalS(e.what());
        cerr<<e.what()<<endl;
        return -1;
    }

    fmt::print("start build fnet_onnx_path\n");
    if(dv::BuildTensorRT(dv::cfg::kRaftFnetOnnxPath,dv::cfg::kRaftFnetTensorrtPath)!=0){
        return -1;
    }
    fmt::print("start build cnet_onnx_path\n");
    if(dv::BuildTensorRT(dv::cfg::kRaftCnetOnnxPath,dv::cfg::kRaftCnetTensorrtPath)!=0){
        return -1;
    }
    fmt::print("start build update_onnx_path\n");
    if(dv::BuildTensorRT(dv::cfg::kRaftUpdateOnnxPath,dv::cfg::kRaftUpdateTensorrtPath)!=0){
        return -1;
    }

    return 0;
}


