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
#include <unordered_map>
#include <sstream>

#include <spdlog/logger.h>

#include "det3d_parameter.h"
#include "utils/log_utils.h"
#include "utils/io_utils.h"
#include "image_process/deeplearning_utils.h"

namespace dynamic_vins{\


Detector3D::Detector3D(const std::string& config_path){
    det3d_para::SetParameters(config_path);
}


void Detector3D::Launch(SemanticImage &img){
    if(det3d_para::use_offline){
        image_seq_id = img.seq;
        image_time = img.time0;
        return;
    }
    else{
        Criticals("det3d_para::use_offline = false, but not implement");
        std::terminate();
    }
}

std::vector<Box3D::Ptr>  Detector3D::WaitResult(){
    if(det3d_para::use_offline){
        if(cfg::dataset == DatasetType::kViode){
            return Detector3D::ReadBox3D(DoubleToStr(image_time,6));
        }
        else if(cfg::dataset == DatasetType::kKitti){
            string target_name = PadNumber(image_seq_id,6);//补零
            return Detector3D::ReadBox3D(target_name);
        }
        else{
            std::cerr<<"Detector3D::WaitResult() not is implemented, as dataset is "<<cfg::dataset_name<<endl;
            return {};
        }
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

        boxes.push_back(Box3D::Box3dFromFCOS3D(tokens,cam_s.cam0));

        index++;
    }
    fp.close();

    return boxes;
}


/**
 * 获取第 frame 帧下的Kitti Tracking数据集的ground truth 3D框
 * @param frame
 * @return
 */
vector<Box3D::Ptr> Detector3D::ReadGroundtruthFromKittiTracking(int frame){
    static std::unordered_map<int,vector<Box3D::Ptr>> boxes_gt;
    static bool is_first_run=true;

    if(det3d_para::kGroundTruthPath.empty()){
        cerr<<"Detector3D::ReadGroundtruthFromKittiTracking(), \n"
                  "det3d_para::kGroundTruthPath is empty"<<endl;
        std::terminate();
    }

    if(is_first_run){
        is_first_run=false;
        std::ifstream fp_gt(det3d_para::kGroundTruthPath);
        if(!fp_gt.is_open()){
            cerr<<"Detector3D::ReadGroundtruthFromKittiTracking(), \n"
                  "open:"<<det3d_para::kGroundTruthPath<<" is failed!"<<endl;
            std::terminate();
        }

        string line_gt;
        while (getline(fp_gt,line_gt)){ //循环读取每行数据
            vector<string> tokens;
            split(line_gt,tokens," ");
            Box3D::Ptr box = Box3D::Box3dFromKittiTracking(tokens,cam_s.cam0);
            int curr_frame = std::stoi(tokens[0]);
            boxes_gt[curr_frame].push_back(box);
        }
        fp_gt.close();
    }

    if(boxes_gt.count(frame)==0){
        return {};
    }
    else{
        return boxes_gt[frame];
    }
}




std::vector<Box3D::Ptr> Detector3D::ReadBox3D(const string &target_name){

    ///获取目录中所有的文件名
    static vector<fs::path> names = GetDirectoryFileNames(det3d_para::kDet3dPreprocessPath);

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
