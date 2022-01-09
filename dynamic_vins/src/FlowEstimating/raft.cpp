/*******************************************************
 *
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of RAFT_CPP. Created by chen on 2021/12/24.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "raft.h"

#include <torch/script.h>

#include "utils.h"
#include "TensorRT/common.h"
#include "TensorRT/logger.h"

namespace dynamic_vins{\


namespace F = torch::nn::functional;
using Tensor = torch::Tensor;
using Slice = torch::indexing::Slice;

/**
 * 图像预处理
 * @param img [3,h,w]的图像张量,值范围为0-255
 * @return
 */
Tensor RaftData::Process(Tensor &img) {
    int h = img.sizes()[1];
    int w = img.sizes()[2];

    ///预处理
    Tensor input_tensor = 2*(img/255.0f) - 1.0f;

    ///pad
    h_pad = ((int(h/8)+1)*8 - h)%8;
    w_pad = ((int(w/8)+1)*8 - w)%8;

    //前两个数pad是2维度，中间两个数pad第1维度，后两个数pad 第0维度
    input_tensor = F::pad(input_tensor, F::PadFuncOptions({w_pad, 0, h_pad, 0, 0, 0}).mode(torch::kConstant));

    return input_tensor.unsqueeze(0).contiguous();
}

Tensor RaftData::Process(cv::Mat &img)
{
    cv::Mat img_float;
    img.convertTo(img_float,CV_32FC3);
    auto input_tensor = torch::from_blob(img_float.data, {img.rows,img.cols ,3 }, torch::kFloat32).to(torch::kCUDA);
    return Process(input_tensor);
}

/**
 *
 * @param tensor shape:[c,h,w]
 * @return
 */
Tensor RaftData::Unpad(Tensor &tensor)
{
    return tensor.index({"...",
                         Slice(h_pad,at::indexing::None),
                         Slice(w_pad,at::indexing::None)});
}



RAFT::RAFT(){
    ///注册预定义的和自定义的插件
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(),"");
    Infos("Read model param");

    auto CreateModel = [](std::unique_ptr<nvinfer1::IRuntime,InferDeleter> &runtime,
            std::shared_ptr<nvinfer1::ICudaEngine> &engine,
            std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> &context,
            const string& path){
        std::string model_str;
        if(std::ifstream ifs(path);ifs.is_open()){
            while(ifs.peek() != EOF){
                std::stringstream ss;
                ss<<ifs.rdbuf();
                model_str.append(ss.str());
            }
            ifs.close();
        }
        else{
            auto msg=fmt::format("Can not open the DETECTOR_SERIALIZE_PATH:{}",path);
            throw std::runtime_error(msg);
        }

        Infos("createInferRuntime");

        ///创建runtime
        runtime=std::unique_ptr<nvinfer1::IRuntime,InferDeleter>(
                nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));

        Infos("deserializeCudaEngine");

        ///反序列化模型
        engine=std::shared_ptr<nvinfer1::ICudaEngine>(
                runtime->deserializeCudaEngine(model_str.data(),model_str.size()) ,InferDeleter());

        Infos("createExecutionContext");

        ///创建执行上下文
        context=std::unique_ptr<nvinfer1::IExecutionContext,InferDeleter>(
                engine->createExecutionContext());

        if(!context){
            throw std::runtime_error("can not create context");
        }
    };

    CreateModel(fnet_runtime_,fnet_engine_,fnet_context_,Config::kRaftFnetTensorrtPath);
    CreateModel(cnet_runtime_,cnet_engine_,cnet_context_,Config::kRaftCnetTensorrtPath);
    CreateModel(update_runtime_,update_engine_,update_context_,Config::kRaftUpdateTensorrtPath);

}

tuple<Tensor, Tensor> RAFT::ForwardFNet(Tensor &tensor0, Tensor &tensor1) {
    void *buffer[4]{};
    buffer[fnet_engine_->getBindingIndex("img0")] = tensor0.data_ptr();
    buffer[fnet_engine_->getBindingIndex("img1")] = tensor1.data_ptr();

    auto opt=torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);

    int index0 = fnet_engine_->getBindingIndex("feat0");
    int index1 = fnet_engine_->getBindingIndex("feat1");
    auto dim2=fnet_engine_->getBindingDimensions(index0);
    auto dim3=fnet_engine_->getBindingDimensions(index1);
    Tensor fmat0 = torch::zeros({dim2.d[0],dim2.d[1],dim2.d[2],dim2.d[3]},opt);
    Tensor fmat1 = torch::zeros({dim3.d[0],dim3.d[1],dim3.d[2],dim3.d[3]},opt);

    buffer[index0] = fmat0.data_ptr();
    buffer[index1] = fmat1.data_ptr();

    fnet_context_->enqueue(1, buffer, stream_, nullptr);

    return {fmat0.to(torch::kFloat),fmat1.to(torch::kFloat)};
}


tuple<Tensor, Tensor> RAFT::ForwardCnet(Tensor &tensor) {
    void *buffer[2];
    buffer[0] = tensor.data_ptr();

    static auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);
    auto dim1=cnet_engine_->getBindingDimensions(1);
    Tensor fmat = torch::zeros({dim1.d[0],dim1.d[1],dim1.d[2],dim1.d[3]},opt);//[1,256,47,154]
    buffer[1] = fmat.data_ptr();

    cnet_context_->enqueue(1, buffer, stream_, nullptr);

    auto t_vector = torch::split_with_sizes(fmat,{128,128},1);

    auto net = torch::tanh(t_vector[0]);//[1,128,47,154]
    auto inp = torch::relu(t_vector[1]);

    return {net,inp};
}


tuple<Tensor, Tensor, Tensor> RAFT::ForwardUpdate(Tensor &net, Tensor &inp, Tensor &corr, Tensor &flow) {
    //input_names=["net_in","inp","corr","flow"],output_names=["net_out", "mask", "delta_flow"]
    void *buffer[7];
    buffer[update_engine_->getBindingIndex("net_in")] = net.data_ptr();
    buffer[update_engine_->getBindingIndex("inp")] = inp.data_ptr();
    buffer[update_engine_->getBindingIndex("corr")] = corr.data_ptr();
    buffer[update_engine_->getBindingIndex("flow")] = flow.data_ptr();

    static auto opt=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat);

    int index_net_out = update_engine_->getBindingIndex("net_out");
    auto net_size= update_engine_->getBindingDimensions(index_net_out);//[1,128,47,154]
    Tensor net_output = torch::zeros({net_size.d[0],net_size.d[1],net_size.d[2],net_size.d[3]},opt);
    buffer[index_net_out]=net_output.data_ptr();

    int index_mask = update_engine_->getBindingIndex("mask");
    auto mask_size= update_engine_->getBindingDimensions(index_mask);//[1,576,47,154]
    Tensor up_mask = torch::zeros({mask_size.d[0],mask_size.d[1],mask_size.d[2],mask_size.d[3]},opt);
    buffer[index_mask] = up_mask.data_ptr();

    int index_delta = update_engine_->getBindingIndex("delta_flow");
    auto delta_size= update_engine_->getBindingDimensions(index_delta);//[1,2,47,154]
    Tensor delta_flow = torch::zeros({delta_size.d[0],delta_size.d[1],delta_size.d[2],delta_size.d[3]},opt);
    buffer[index_delta]=delta_flow.data_ptr();

    update_context_->enqueue(1, buffer, stream_, nullptr);
    return {net_output,up_mask,delta_flow};
}

/**
 * 计算相关金字塔
 * @param tensor0
 * @param tensor1
 * @return
 */
void RAFT::ComputeCorrPyramid(Tensor &tensor0, Tensor &tensor1) {
    auto size = tensor0.sizes();
    const int batch = size[0];
    const int dim = size[1];
    const int h = size[2];
    const int w = size[3];
    const int num_level = 4;
    corr_pyramid.clear();

    ///计算corr张量
    Tensor t0_view = tensor0.view({batch,dim,h*w});//[1, 256, 47*154]
    Tensor t1_view = tensor1.view({batch,dim,h*w});//[1, 256, 47*154]
    Tensor corr = torch::matmul(t0_view.transpose(1,2),t1_view);//[1, 7238,7238]
    corr = corr.view({batch,h,w,1,h,w});//[1,47,154,1,47,154]
    corr = corr / std::sqrt(dim);

    corr = corr.reshape({batch*h*w,1,h,w}).to(torch::kFloat);

    ///构造corr volume金字塔
    corr_pyramid.push_back(corr);//[batch*h*w,1,h,w]
    static auto opt =F::AvgPool2dFuncOptions(2).stride(2);//kernel size =2 ,stride =2;

    for(int i=1;i<num_level;++i){
        corr = F::avg_pool2d(corr,opt);
        corr_pyramid.push_back(corr);
    }
}

/**
 * 初始化光流
 * @param tensor [1,3,376, 1232]
 * @return {coords0,coords1},shape:[1,2,47,154]
 */
tuple<Tensor, Tensor> RAFT::InitializeFlow(Tensor &tensor) {
    auto size = tensor.sizes();
    static auto opt = torch::TensorOptions(torch::kCUDA);
    auto coords_grid = [](int batch,int h,int w){
        auto coords_vector = torch::meshgrid({torch::arange(h,opt),torch::arange(w,opt)});//[h,w]
        auto coords = torch::stack({coords_vector[1],coords_vector[0]},0).to(torch::kFloat);//[2,h,w]
        return coords.unsqueeze(0).expand({batch,2,h,w});//(1,2,h,w)
    };

    auto coords0 = coords_grid(size[0],size[2]/8,size[3]/8);
    auto coords1 = coords0.clone();
    return {coords0,coords1};
}

/**
 * 索引相关性张量
 * @param tensor [1,2,47,154]
 * @param pyramid
 * @return [batch*h*w,1,9,9]
 */
Tensor RAFT::IndexCorrVolume(Tensor &tensor){
    auto bilinear_sampler = [](Tensor &img,Tensor &coords){ //img:[7238,1,h,w], coords:[7238, 9, 9, 2]
        int H = img.sizes()[2];
        int W = img.sizes()[3];
        auto grids = coords.split_with_sizes({1,1},-1);//划分为两个[7238,9,9,1]的张量
        Tensor xgrid = 2*grids[0]/(W-1) -1;//归一化到[-1,1]
        Tensor ygrid = 2*grids[1]/(H-1) -1;
        Tensor grid = torch::cat({xgrid,ygrid},-1);//[7238, 9, 9, 2]
        static auto opt = F::GridSampleFuncOptions().align_corners(true);
        return grid_sample(img,grid,opt);
    };

    static auto gpu = torch::TensorOptions(torch::kCUDA);

    const int r = 4;
    const int rr = 2*r+1;
    const int num_level = 4;
    auto coords = tensor.permute({0,2,3,1});//[batch,h,w,2]
    auto size = coords.sizes();
    const int batch = size[0], h = size[1],w = size[2];

    vector<Tensor> out_pyramid;
    for(int i=0;i<num_level;++i){
        auto corr = corr_pyramid[i];//层i的相关性张量，[7238,1,h,w]
        //每个像素在该金字塔层搜索的范围
        auto delta = torch::stack(torch::meshgrid( //[9,9,2],  [2*r+1, 2*r+1,2]
                {torch::linspace(-r,r,rr,gpu),torch::linspace(-r,r,rr,gpu)}),-1);
        auto delta_lvl = delta.view({1,rr,rr,2});//[1,9,9,2]
        //将坐标值缩放到层i的值
        auto centroid_lvl = coords.reshape({batch*h*w,1,1,2}) / std::pow(2,i);//[batch*h*w,1,1,2]
        auto coords_lvl = centroid_lvl + delta_lvl;//[7238, 9, 9, 2]，7238表示每个像素，9 9表示每个像素检索的范围，2表示xy值
        corr = bilinear_sampler(corr,coords_lvl);//[batch*h*w,1,9,9]
        corr = corr.view({batch,h,w,-1});//[batch,h,w,81]
        out_pyramid.push_back(corr);
    }
    Tensor out = torch::cat(out_pyramid,-1);
    return out.permute({0,3,1,2}).contiguous().to(torch::kFloat);
}





/**
 * 光流估计
 * @param tensor0 图像1 [1,3,376, 1232]
 * @param tensor1 图像2 [1,3,376, 1232]
 * @return
 */
vector<Tensor> RAFT::Forward(Tensor& tensor0, Tensor& tensor1) {
    TicToc tt;
    static auto gpu=torch::TensorOptions(torch::kCUDA).dtype(torch::kFloat);
    const int num_iter = 20;

    auto [fmat0,fmat1] = ForwardFNet(tensor0, tensor1);//fmat0和fmat1:[1, 256, 47, 154]
        Debugs("ForwardFNet:{} ms", tt.TocThenTic());

    /**
     * [7238, 1, 47, 154]
     * [7238, 1, 23, 77]
     * [7238, 1, 11, 38]
     * [7238, 1, 5, 19]
     */
    ComputeCorrPyramid(fmat0,fmat1);
        Debugs("corr_pyramid:{} ms", tt.TocThenTic());
    //for(auto &p : corr_pyramid) DebugS("corr_pyramid.shape:{}", dims2str(p.sizes()));
    auto [net,inp] = ForwardCnet(tensor1);//net和inp:[1,128,47,154]
        Debugs("forward_cnet:{} ms", tt.TocThenTic());

    auto [coords0,coords1] = InitializeFlow(tensor1);//coords0和coords1：[1,2,47,154]
        Debugs("initialize_flow:{} ms", tt.TocThenTic());

    if(last_flow.defined()){
        coords1 = coords1 + last_flow;
    }

    vector<Tensor> flow_prediction;
    for(int i=0;i<num_iter;++i){
        //DebugS("{}",i);
        auto corr = IndexCorrVolume(coords1);//[batch*h*w, 1, 9, 9]
        auto flow = coords1 - coords0;//[1,2,47,154]
        auto [net1,up_mask,delta_flow] = ForwardUpdate(net,inp,corr,flow);
        net = net1;
        coords1 = coords1 + delta_flow;

        if(i==num_iter-1){///上采样
            flow = coords1 - coords0;
            static auto opt = F::InterpolateFuncOptions().size(
                    vector<int64_t>({8*coords1.sizes()[2],8*coords1.sizes()[3]})).
                            mode(torch::kBilinear).align_corners(true);
            auto flow_up = 8 * interpolate(flow,opt) ;//[1, 2, 376, 1232]
            flow_prediction.push_back(flow_up);
            last_flow = flow;
        }
    }
        Debugs("iter all:{} ms", tt.TocThenTic());

    corr_pyramid.clear();

    return flow_prediction;
}


}

