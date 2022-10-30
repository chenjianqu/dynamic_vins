/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "flow_estimator.h"

#include <filesystem>
#include <opencv2/optflow.hpp>
#include "flow_parameter.h"
#include "utils/log_utils.h"
#include "utils/parameters.h"
#include "utils/io_utils.h"

namespace dynamic_vins{\

using Tensor = torch::Tensor;

FlowEstimator::FlowEstimator(const std::string& config_path){
    flow_para::SetParameters(config_path);
    if(cfg::use_dense_flow && !flow_para::use_offline_flow){
        raft_ = std::make_unique<RAFT>();
        data_ = std::make_shared<RaftData>();
    }
}


void FlowEstimator::Launch(SemanticImage &img){
    if(flow_para::use_offline_flow){
        SynchronizeReadFlow(img.seq);
    }
    else{
        SynchronizeForward(img.img_tensor);
    }
}

cv::Mat FlowEstimator::WaitResult(){
    cv::Mat flow_cv;
    if(flow_para::use_offline_flow){
        return WaitingReadFlowImage();
    }
    else{
        auto flow_tensor = WaitingForwardResult();
        flow_tensor = flow_tensor.to(torch::kCPU);
        return cv::Mat(flow_tensor.sizes()[1],flow_tensor.sizes()[2],CV_8UC2,flow_tensor.data_ptr()).clone();
    }
}




///异步检测光流
void FlowEstimator::SynchronizeForward(Tensor &img){
    if(is_running_){
        return;
    }
    if(flow_para::use_offline_flow){
        cerr<<"because use_preprocess_flow=true,so can not launch FlowEstimator::SynchronizeForward()";
        std::terminate();
    }
    flow_thread_ = std::thread([this](torch::Tensor &img){
            TicToc tt;
            this->is_running_=true;
            this->output = this->Forward(img);
            this->is_running_=false;
            Infos("FlowEstimator forward time:{} ms",tt.Toc());
        }
        ,std::ref(img));
}

Tensor FlowEstimator::WaitingForwardResult(){
    flow_thread_.join();
    return output;
}

///异步读取光流图像
void FlowEstimator::SynchronizeReadFlow(unsigned int seq_id){
    if(is_running_){
        return;
    }
    if(!flow_para::use_offline_flow){
        cerr<<"because use_preprocess_flow=false,so can not launch FlowEstimator::SynchronizeReadFlow()";
        std::terminate();
    }
    flow_thread_ = std::thread([this](unsigned int seq){
            TicToc tt;
            this->is_running_=true;
            this->img_flow = FlowEstimator::ReadFlowImage(seq);
            this->is_running_=false;
            Infos("FlowEstimator read flow time:{} ms",tt.Toc());
        }
        ,seq_id);
}

cv::Mat FlowEstimator::WaitingReadFlowImage(){
    flow_thread_.join();
    return img_flow;
}



/**
 * 进行前向的光流估计
 * @param img 未经处理的图像张量，shape=[3,h,w],值范围[0-255]，数据类型Float32
 * @return 估计得到三光流张量，[2,h,w]
 */
Tensor FlowEstimator::Forward(Tensor &img) {
    auto curr_img = data_->Process(img);
    if(!last_img_.defined()){
        last_img_ = curr_img;
        auto opt = torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat32);
        return torch::zeros({2,img.sizes()[1],img.sizes()[2]},opt);
    }

    vector<Tensor> pred = raft_->Forward(last_img_,curr_img);

    auto flow = pred.back();//[1,2,h,w]
    flow = flow.squeeze();
    flow = data_->Unpad(flow);
    last_img_ = curr_img;
    return flow;
}

/**
 * 读取离线估计的光流数据
 * @param seq_id
 * @return
 */
cv::Mat FlowEstimator::ReadFlowImage(unsigned int seq_id){
    ///补零
    int name_width=6;
    std::stringstream ss;
    ss<<std::setw(name_width)<<std::setfill('0')<<seq_id;
    string target_name;
    ss >> target_name;

    ///获取目录中所有的文件名
    static vector<fs::path> names = GetDirectoryFileNames(flow_para::kFlowOfflinePath);


    cv::Mat flow_img;

    ///二分查找
    int low=0,high=names.size()-1;
    while(low<=high){
        int mid=(low+high)/2;
        string name_stem = names[mid].stem().string();
        if(name_stem == target_name){
            string n_path = (flow_para::kFlowOfflinePath / names[mid]).string();
            flow_img = cv::optflow::readOpticalFlow(n_path);
            break;
        }
        else if(name_stem > target_name){
            high = mid-1;
        }
        else{
            low = mid+1;
        }
    }

    if(flow_img.empty()){
        string msg=fmt::format("Can not find the target name:{} in dir:{}",target_name,flow_para::kFlowOfflinePath);
        Errors(msg);
        cerr<<msg<<endl;
    }

    return flow_img;
    /*if(read_flow.empty())
        return {};
    torch::Tensor tensor = torch::from_blob(read_flow.data, {read_flow.rows,read_flow.cols ,2}, torch::kFloat32).to(torch::kCUDA);
    tensor = tensor.permute({2,0,1});
    return tensor;*/
}



}

