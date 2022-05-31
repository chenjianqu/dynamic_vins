/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "detector3d.h"

#include <iostream>
#include <filesystem>
#include <spdlog/logger.h>

#include "det3d_parameter.h"
#include "utils/utils.h"
#include "utils/log_utils.h"

namespace dynamic_vins{\


Detector3D::Detector3D(const std::string& config_path){
    det3d_para::SetParameters(config_path);
}


void Detector3D::Launch(SemanticImage &img){
    if(det3d_para::use_offline){
        image_seq_id = img.seq;
        return;
    }
    else{
        Criticals("det3d_para::use_offline = false, but not implement");
        std::terminate();
    }
}

std::vector<Box3D::Ptr>  Detector3D::WaitResult(){
    if(det3d_para::use_offline){
        return Detector3D::ReadBox3D(image_seq_id);
    }
    else{
        Criticals("det3d_para::use_offline = false, but not implement");
        std::terminate();
    }
}



std::vector<Box3D::Ptr> Detector3D::ReadBox3dFromTxt(const std::string &txt_path,double score_threshold)
{
    std::vector<Box3D::Ptr> boxes;

    std::ifstream fp(txt_path);
    string line;
    int index=0;
    while (getline(fp,line)){ //循环读取每行数据
        vector<string> tokens;
        split(line,tokens," ");

        if(std::stod(tokens[2]) < score_threshold)
            continue;

        Box3D::Ptr box = std::make_shared<Box3D>(tokens);

        boxes.push_back(box);

        index++;
    }
    fp.close();

    return boxes;
}




std::vector<Box3D::Ptr> Detector3D::ReadBox3D(unsigned int seq_id){
    string target_name = PadNumber(seq_id,6);///补零

    ///获取目录中所有的文件名
    static vector<fs::path> names;
    if(names.empty()){
        fs::path dir_path(det3d_para::kDet3dPreprocessPath);
        if(!fs::exists(dir_path))
            return {};
        fs::directory_iterator dir_iter(dir_path);
        for(auto &it : dir_iter)
            names.emplace_back(it.path().filename());
        std::sort(names.begin(),names.end());
    }

    vector<Box3D::Ptr> boxes;
    bool success_read= false;

    ///二分查找
    int low=0,high=names.size()-1;
    while(low<=high){
        int mid=(low+high)/2;
        string name_stem = names[mid].stem().string();
        if(name_stem == target_name){
            string n_path = (det3d_para::kDet3dPreprocessPath/names[mid]).string();
            Debugs("det3d read:{}",n_path);
            boxes = ReadBox3dFromTxt(n_path,det3d_para::kDet3dScoreThreshold);
            success_read = true;
            break;
        }
        else if(name_stem > target_name){
            high = mid-1;
        }
        else{
            low = mid+1;
        }
    }

    if(!success_read){
        string msg=fmt::format("Can not find the target name:{} in dir:{}",target_name,det3d_para::kDet3dPreprocessPath);
        Errors(msg);
        std::cerr<<msg<<std::endl;
    }

    return boxes;
}


}
