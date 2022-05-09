/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/
#include <iostream>
#include <memory>

#include "utils/tensorrt/tensorrt_utils.h"
#include "det2d/det2d_parameter.h"

using namespace std;
namespace dv = dynamic_vins;


int main(int argc,char** argv)
{
    if(argc != 2){
        cerr<<"please intput: 参数文件"<<endl;
        return 1;
    }
    string config_file = argv[1];
    dv::det2d_para::SetParameters(config_file);
    return dv::BuildTensorRT(dv::det2d_para::kDetectorOnnxPath,dv::det2d_para::kDetectorSerializePath);
}

