/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "solo_head.h"
#include "det2d_parameter.h"
#include "utils/dataset/coco_utils.h"
#include "utils/dataset/kitti_utils.h"
#include "utils/log_utils.h"
#include "utils/torch_utils.h"

namespace dynamic_vins{\


using Slice=torch::indexing::Slice;
using InterpolateFuncOptions=torch::nn::functional::InterpolateFuncOptions;
namespace idx=torch::indexing;


Solov2::Solov2(){
    size_trans_=torch::from_blob(det2d_para::kSoloNumGrids.data(), {int(det2d_para::kSoloNumGrids.size())},
                                 torch::kFloat).clone();
    size_trans_=size_trans_.pow(2).cumsum(0);
}


torch::Tensor Solov2::MatrixNMS(torch::Tensor &seg_masks,torch::Tensor &cate_labels,torch::Tensor &cate_scores,
                                torch::Tensor &sum_mask)
{
    int n_samples=cate_labels.sizes()[0];

    //seg_masks.shape [n,h,w] -> [n,h*w]
    seg_masks = seg_masks.reshape({n_samples,-1}).to(torch::kFloat);

    ///计算两个实例之间的内积，即相交的像素数
    auto inter_matrix=torch::mm(seg_masks,seg_masks.transpose(1,0));
    auto sum_mask_x=sum_mask.expand({n_samples,n_samples});

    ///两个两两实例之间的IOU
    auto iou_matrix = (inter_matrix / (sum_mask_x + sum_mask_x.transpose(1,0) - inter_matrix ) ).triu(1);
    auto cate_label_x = cate_labels.expand({n_samples,n_samples});

    auto label_matrix= (cate_label_x==cate_label_x.transpose(1,0)).to(torch::kFloat).triu(1);

    ///计算IoU补偿
    auto compensate_iou = std::get<0>( (iou_matrix * label_matrix).max(0) );//max()返回两个张量(最大值和最大值索引)组成的tuple
    compensate_iou = compensate_iou.expand({n_samples,n_samples}).transpose(1,0);
    auto decay_iou = iou_matrix * label_matrix;

    ///计算实例置信度的衰减系数
    torch::Tensor decay_coefficient;
    if(det2d_para::kSoloNmsKernel == "gaussian"){
        auto decay_matrix = torch::exp(-1 * det2d_para::kSoloNmsSigma * (decay_iou.pow(2)));
        auto compensate_matrix= torch::exp(-1 * det2d_para::kSoloNmsSigma * (compensate_iou.pow(2)));
        decay_coefficient = std::get<0>( (decay_matrix / compensate_matrix).min(0) );
    }
    else if(det2d_para::kSoloNmsKernel == "linear"){
        auto decay_matrix = (1-decay_iou) / (1-compensate_iou) ;
        decay_coefficient = std::get<0>( (decay_matrix).min(0) );
    }
    else{
        throw;
    }
    ///更新置信度
    auto cate_scores_update = cate_scores * decay_coefficient;
    return  cate_scores_update;
}




/**
 * 处理solo的输出
 * @param outputs
 */
cv::Mat Solov2::GetSingleSeg(std::vector<torch::Tensor> &outputs, torch::Device device, std::vector<Box2D::Ptr> &insts)
{
    /*TicToc ticToc;
    const int batch=0;
    const int level_num=5;//FPN共输出5个层级
    auto kernel_tensor=outputs[0][batch].view({kSoloTensorChannel, -1}).permute({1, 0});
    for(int i=1;i<level_num;++i){
        auto kt=outputs[i][batch].view({kSoloTensorChannel, -1}); //kt的维度是(128,h*w)
        kernel_tensor = torch::cat({kernel_tensor,kt.permute({1,0})},0);
    }

    const int cate_channel=80;
    auto cate_tensor=outputs[level_num][batch].view({cate_channel,-1}).permute({1,0});
    for(int i=level_num+1;i<2*level_num;++i){
        auto ct=outputs[i][batch].view({cate_channel,-1}); //kt的维度是(h*w, 80)
        cate_tensor = torch::cat({cate_tensor,ct.permute({1,0})},0);
    }
    auto feat_tensor=outputs[2*level_num][batch];

    const int feat_h=feat_tensor.sizes()[1];
    const int feat_w=feat_tensor.sizes()[2];
    const int pred_num=cate_tensor.sizes()[0];//所有的实例数量(3872)

    ticToc.TocPrintTic("input reshape:");
    ///过滤掉低于0.1置信度的实例
    auto inds= cate_tensor > para::kSoloScoreThr;
    torch::IntArrayRef dims={0,1};
    if(inds.sum(dims).item().toInt() == 0){
        Warns("inds.sum(dims) == 0");
        return {};
    }
    cate_tensor=cate_tensor.masked_select(inds);
    ///获得所有满足阈值的，得到的inds中的元素inds[i,j]表示第i个实例是属于j类
    inds=inds.nonzero();
    ///获得每个实例的类别
    auto cate_labels=inds.index({"...",1});
    ///获得满足阈值的kernel预测
    auto pred_index=inds.index({"...",0});
    auto kernel_preds=kernel_tensor.index({pred_index});
    ticToc.TocPrintTic("过滤掉低于0.1置信度的实例:");
    ///计算每个实例的stride
    //首先计算各个层级的分界
    auto strides=torch::ones({pred_num},device);
    const int n_stage=kSoloNumGrids.size();
    //计算各个层级上的实例的strides
    int index0=size_trans_[0].item().toInt();
    strides.index_put_({idx::Slice(idx::None,index0)}, kSoloStrides[0]);
    for(int i=1;i<n_stage;++i){
        int index_start=size_trans_[i - 1].item().toInt();
        int index_end=size_trans_[i].item().toInt();
        strides.index_put_({idx::Slice(index_start,index_end)}, kSoloStrides[i]);
    }
    //保留满足阈值的实例的strides
    strides=strides.index({pred_index});
    ticToc.TocPrintTic("计算每个实例的stride:");
    ///将mask_feat和kernel进行卷积
    auto seg_preds=feat_tensor.unsqueeze(0);
    //首先将kernel改变为1x1卷积核的形状
    kernel_preds=kernel_preds.view({kernel_preds.sizes()[0],kernel_preds.sizes()[1],1,1});
    //然后进行卷积
    seg_preds=torch::conv2d(seg_preds,kernel_preds,{},1);
    seg_preds=torch::squeeze(seg_preds,0).sigmoid();
    ticToc.TocPrintTic("将mask_feat和kernel进行卷积:");

    ///计算mask
    auto seg_masks=seg_preds > para::kSoloMaskThr;
    auto sum_masks=seg_masks.sum({1,2}).to(torch::kFloat);
    ticToc.TocPrintTic("计算mask:");
    ///根据strides过滤掉像素点太少的实例
    auto keep=sum_masks > strides;
    if(keep.sum(0).item().toInt()==0){
        cerr<<"keep.sum(0) == 0"<<endl;
        return {};
    }
    seg_masks = seg_masks.index({keep,"..."});
    seg_preds = seg_preds.index({keep,"..."});
    sum_masks=sum_masks.index({keep});
    cate_tensor=cate_tensor.index({keep});
    cate_labels=cate_labels.index({keep});

    ticToc.TocPrintTic("根据strides过滤掉像素点太少的实例:");
    ///根据mask预测设置实例的置信度
    auto seg_scores=(seg_preds * seg_masks.to(torch::kFloat)).sum({1,2}) / sum_masks;
    cate_tensor *= seg_scores;
    ///根据cate_score进行排序，用于NMS
    auto sort_inds = torch::argsort(cate_tensor,-1,true);
    if(sort_inds.sizes()[0] >  para::kSoloNmsPre){
        sort_inds=sort_inds.index({idx::Slice(idx::None,para::kSoloNmsPre)});
    }
    seg_masks=seg_masks.index({sort_inds,"..."});
    seg_preds=seg_preds.index({sort_inds,"..."});
    sum_masks=sum_masks.index({sort_inds});
    cate_tensor=cate_tensor.index({sort_inds});
    cate_labels=cate_labels.index({sort_inds});
    ticToc.TocPrintTic("NMS准备:");
    ///执行Matrix NMS
    auto cate_scores = MatrixNMS(seg_masks,cate_labels,cate_tensor,sum_masks);
    ticToc.TocPrintTic("NMS执行:");

    ///根据新的置信度过滤结果
    keep = cate_scores >= para::kSoloUpdateThr;
    if(keep.sum(0).item().toInt() == 0){
        cout<<"keep.sum(0) == 0"<<endl;
        return {};
    }
    seg_preds = seg_preds.index({keep,"..."});
    cate_scores = cate_scores.index({keep});
    cate_labels = cate_labels.index({keep});

    ///再次根据置信度进行排序
    sort_inds = torch::argsort(cate_scores,-1,true);
    if(sort_inds.sizes()[0] >  para::kSoloMaxPerImg){
        sort_inds=sort_inds.index({idx::Slice(idx::None,para::kSoloMaxPerImg)});
    }
    seg_preds=seg_preds.index({sort_inds,"..."});
    cate_scores=cate_scores.index({sort_inds});
    cate_labels=cate_labels.index({sort_inds});
    ticToc.TocPrintTic("NMS执行:");
    //F::InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({image.rows, image.cols})).align_corners(true)
    ///对mask进行双线性上采样,
    auto options=InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({feat_h*4,feat_w*4}));
    seg_preds = torch::nn::functional::interpolate(seg_preds.unsqueeze(0),options);

    seg_preds =seg_preds.index({"...", Slice(idx::None,para::kInputHeight), Slice(idx::None, para::kInputWidth)});
    //再次上采样到原始的图片大小
    //options=InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({cfg.imgOriginH,cfg.imgOriginW}));
    //seg_preds = torch::nn::functional::interpolate(seg_preds,options);

    seg_preds=seg_preds.squeeze(0);

    ///阈值化
    seg_masks = (seg_preds > para::kSoloMaskThr).to(torch::kFloat );
    //cout<<"seg_masks.sizes"<<seg_masks.sizes()<<endl;

    ticToc.TocPrintTic("上采样和阈值化:");

    ///根据mask计算包围框

    for(int i=0;i<seg_masks.sizes()[0];++i){
        auto nz=seg_masks[i].nonzero();
        auto max_xy =std::get<0>( torch::max(nz,0) );
        auto min_xy =std::get<0>( torch::min(nz,0) );

        InstInfo inst;
        inst.id = i;
        inst.label_id =cate_labels[i].item().toInt();
        inst.max_pt.x = max_xy[1].item().toInt();
        inst.max_pt.y = max_xy[0].item().toInt();
        inst.min_pt.x = min_xy[1].item().toInt();
        inst.min_pt.y = min_xy[0].item().toInt();
        inst.prob = cate_scores[i].item().toFloat();
        insts.push_back(inst);
    }

    ///可视化
    seg_masks = seg_masks.unsqueeze(3).expand({seg_masks.sizes()[0],seg_masks.sizes()[1],seg_masks.sizes()[2],3});
    auto rand_color = torch::randint(0, 255, { seg_masks.sizes()[0], 3},kernel_tensor.device());
    rand_color = rand_color.unsqueeze(1).unsqueeze(2).expand({seg_masks.sizes()[0],seg_masks.sizes()[1],seg_masks.sizes()[2],3});
    auto show_tensor = seg_masks * rand_color;    //非常耗时18ms
    auto show_img = show_tensor.sum(0).clip(0,255).squeeze(0) * 0.6f; //非常耗时7ms
    show_img=show_img.to(torch::kInt8).detach();
    show_img=show_img.to(torch::kCPU);
    cv::Mat timg=cv::Mat(cv::Size(show_img.sizes()[1], show_img.sizes()[0]), CV_8UC3, show_img.data_ptr()).clone();

    return timg;*/
}


std::tuple<std::vector<cv::Mat>,std::vector<Box2D::Ptr>> Solov2::GetSingleSeg(std::vector<torch::Tensor> &outputs,
                                                                              ImageInfo& img_info){
    /*torch::Device device = outputs[0].device();
    constexpr int batch=0;
    constexpr int level_num=5;//FPN共输出5个层级
    auto kernel_tensor=outputs[0][batch].view({kSoloTensorChannel, -1}).permute({1, 0});
    for(int i=1;i<level_num;++i){
        auto kt=outputs[i][batch].view({kSoloTensorChannel, -1}); //kt的维度是(128,h*w)
        kernel_tensor = torch::cat({kernel_tensor,kt.permute({1,0})},0);
    }
    constexpr int cate_channel=80;
    auto cate_tensor=outputs[level_num][batch].view({cate_channel,-1}).permute({1,0});
    for(int i=level_num+1;i<2*level_num;++i){
        auto ct=outputs[i][batch].view({cate_channel,-1}); //kt的维度是(h*w, 80)
        cate_tensor = torch::cat({cate_tensor,ct.permute({1,0})},0);
    }

    auto feat_tensor=outputs[2*level_num][batch];

    const int feat_h=feat_tensor.sizes()[1];
    const int feat_w=feat_tensor.sizes()[2];
    const int pred_num=cate_tensor.sizes()[0];//所有的实例数量(3872)

    ///过滤掉低于0.1置信度的实例
    auto inds= cate_tensor > para::kSoloScoreThr;
    if(inds.sum(torch::IntArrayRef({0,1})).item().toInt() == 0){
        Warns("inds.sum(dims) == 0");
        return {std::vector<cv::Mat>(),std::vector<InstInfo>()};
    }
    cate_tensor=cate_tensor.masked_select(inds);
    ///获得所有满足阈值的，得到的inds中的元素inds[i,j]表示第i个实例是属于j类
    inds=inds.nonzero();
    ///获得每个实例的类别
    auto cate_labels=inds.index({"...",1});
    ///获得满足阈值的kernel预测
    auto pred_index=inds.index({"...",0});
    auto kernel_preds=kernel_tensor.index({pred_index});
    cout<<"过滤掉低于0.1置信度的实例,kernel_preds.sizes:"<<kernel_preds.sizes()<<endl;
    ///计算每个实例的stride
    auto strides=torch::ones({pred_num},device);
    const int n_stage=kSoloNumGrids.size();

    //计算各个层级上的实例的strides
    int index0=size_trans_[0].item().toInt();
    strides.index_put_({idx::Slice(idx::None,index0)}, kSoloStrides[0]);
    for(int i=1;i<n_stage;++i){
        int index_start=size_trans_[i - 1].item().toInt();
        int index_end=size_trans_[i].item().toInt();
        strides.index_put_({idx::Slice(index_start,index_end)}, kSoloStrides[i]);
    }
    //保留满足阈值的实例的strides
    strides=strides.index({pred_index});

    ///将mask_feat和kernel进行卷积
    auto seg_preds=feat_tensor.unsqueeze(0);
    //首先将kernel改变为1x1卷积核的形状
    kernel_preds=kernel_preds.view({kernel_preds.sizes()[0],kernel_preds.sizes()[1],1,1});
    //然后进行卷积
    seg_preds=torch::conv2d(seg_preds,kernel_preds,{},1);
    seg_preds=torch::squeeze(seg_preds,0).sigmoid();

    ///计算mask
    auto seg_masks=seg_preds > para::kSoloMaskThr;
    auto sum_masks=seg_masks.sum({1,2}).to(torch::kFloat);

    ///根据strides过滤掉像素点太少的实例
    auto keep=sum_masks > strides;
    if(keep.sum(0).item().toInt()==0){
        cerr<<"keep.sum(0) == 0"<<endl;
        return {std::vector<cv::Mat>(),std::vector<InstInfo>()};
    }
    seg_masks = seg_masks.index({keep,"..."});
    seg_preds = seg_preds.index({keep,"..."});
    sum_masks = sum_masks.index({keep});
    cate_tensor = cate_tensor.index({keep});
    cate_labels = cate_labels.index({keep});
    ///根据mask预测设置实例的置信度
    auto seg_scores=(seg_preds * seg_masks.to(torch::kFloat)).sum({1,2}) / sum_masks;
    cate_tensor *= seg_scores;
    ///根据cate_score进行排序，用于NMS
    auto sort_inds = torch::argsort(cate_tensor,-1,true);
    if(sort_inds.sizes()[0] >  para::kSoloNmsPre){
        sort_inds=sort_inds.index({idx::Slice(idx::None,para::kSoloNmsPre)});
    }
    seg_masks=seg_masks.index({sort_inds,"..."});
    seg_preds=seg_preds.index({sort_inds,"..."});
    sum_masks=sum_masks.index({sort_inds});
    cate_tensor=cate_tensor.index({sort_inds});
    cate_labels=cate_labels.index({sort_inds});
    ///执行Matrix NMS
    auto cate_scores = MatrixNMS(seg_masks,cate_labels,cate_tensor,sum_masks);
    ///根据新的置信度过滤结果
    keep = cate_scores >= para::kSoloUpdateThr;
    if(keep.sum(0).item().toInt() == 0){
        cout<<"keep.sum(0) == 0"<<endl;
        return {std::vector<cv::Mat>(),std::vector<InstInfo>()};
    }
    seg_preds = seg_preds.index({keep,"..."});
    cate_scores = cate_scores.index({keep});
    cate_labels = cate_labels.index({keep});
    sum_masks = sum_masks.index({keep});

    ///再次根据置信度进行排序
    sort_inds = torch::argsort(cate_scores,-1,true);
    if(sort_inds.sizes()[0] >  para::kSoloMaxPerImg){
        sort_inds=sort_inds.index({idx::Slice(idx::None,para::kSoloMaxPerImg)});
    }
    seg_preds=seg_preds.index({sort_inds,"..."});
    cate_scores=cate_scores.index({sort_inds});
    cate_labels=cate_labels.index({sort_inds});
    sum_masks = sum_masks.index({sort_inds});
    ///对mask进行双线性上采样,
    auto options=InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({feat_h*4,feat_w*4}));
    seg_preds = torch::nn::functional::interpolate(seg_preds.unsqueeze(0),options);
    ///对mask进行裁切、缩放，得到原始图片大小的mask
    seg_preds =seg_preds.index({"...",Slice(img_info.rect_y,img_info.rect_y+img_info.rect_h),
                                Slice(img_info.rect_x,img_info.rect_x+img_info.rect_w)});
    options=InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({img_info.origin_h, img_info.origin_w}));
    seg_preds = torch::nn::functional::interpolate(seg_preds,options);
    seg_preds=seg_preds.squeeze(0);
    ///阈值化
    seg_masks = seg_preds > para::kSoloMaskThr;
    auto merger_mask = seg_masks.sum(0).to(torch::kInt8) * 255;
    merger_mask = merger_mask.to(torch::kCPU);

    std::vector<cv::Mat> masks;
    cv::Mat merger_mask_img = cv::Mat(cv::Size(merger_mask.sizes()[1],
                                               merger_mask.sizes()[0]), CV_8UC1, merger_mask.data_ptr()).clone();
    masks.push_back(merger_mask_img);

    std::vector<InstInfo> insts;
    if(para::slam == SlamType::kDynamic){
        ///根据mask计算包围框
        for(int i=0;i<seg_masks.sizes()[0];++i){
            auto nz=seg_masks[i].nonzero();
            auto max_xy =std::get<0>( torch::max(nz,0) );
            auto min_xy =std::get<0>( torch::min(nz,0) );
            InstInfo inst;
            inst.id = i;
            inst.label_id =cate_labels[i].item().toInt();
            inst.max_pt.x = max_xy[1].item().toInt();
            inst.max_pt.y = max_xy[0].item().toInt();
            inst.min_pt.x = min_xy[1].item().toInt();
            inst.min_pt.y = min_xy[0].item().toInt();
            inst.prob = cate_scores[i].item().toFloat();
            insts.push_back(inst);
        }

        seg_masks = (seg_masks.to(torch::kInt8) *255).to(torch::kCPU);
        for(int i=0;i<seg_masks.sizes()[0];++i)
        {
            auto mask_t = seg_masks[i];
            cv::Mat mask_img = cv::Mat(cv::Size(mask_t.sizes()[1], mask_t.sizes()[0]),
                                       CV_8UC1, mask_t.data_ptr()).clone();
            masks.push_back(mask_img);
        }

    }
    return {masks,insts};*/
}


void Solov2::GetSegTensor(std::vector<torch::Tensor> &outputs, ImageInfo& img_info, torch::Tensor &mask_tensor,
                          std::vector<Box2D::Ptr> &insts){
    torch::Device device = outputs[0].device();

    constexpr int kBatchIndex=0;
    const int kNumStage= det2d_para::kSoloNumGrids.size();//FPN共输出5个层级

    auto kernel_tensor=outputs[0][kBatchIndex].view({det2d_para::kSoloTensorChannel, -1}).permute({1, 0});
    for(int i=1; i < kNumStage; ++i){
        auto kt=outputs[i][kBatchIndex].view({det2d_para::kSoloTensorChannel, -1}); //kt的维度是(128,h*w)
        kernel_tensor = torch::cat({kernel_tensor,kt.permute({1,0})},0);
    }

    constexpr int kCateChannel=80;
    auto cate_tensor=outputs[kNumStage][kBatchIndex].view({kCateChannel, -1}).permute({1, 0});
    for(int i= kNumStage + 1; i < 2 * kNumStage; ++i){
        auto ct=outputs[i][kBatchIndex].view({kCateChannel, -1}); //kt的维度是(h*w, 80)
        cate_tensor = torch::cat({cate_tensor,ct.permute({1,0})},0);
    }

    auto feat_tensor=outputs[2 * kNumStage][kBatchIndex];

    const int kFeatHeight=feat_tensor.sizes()[1];
    const int kFeatWidth=feat_tensor.sizes()[2];
    const int kPredNum=cate_tensor.sizes()[0];//所有的实例数量(3872)

    ///过滤掉低于0.1置信度的实例
    auto inds= cate_tensor > det2d_para::kSoloScoreThr;
    if(inds.sum(torch::IntArrayRef({0,1})).item().toInt() == 0){
        Warns("GetSegTensor | inds.sum(dims) == 0");
        return;
    }
    cate_tensor=cate_tensor.masked_select(inds);

    ///获得所有满足阈值的，得到的inds中的元素inds[i,j]表示第i个实例是属于j类
    inds=inds.nonzero();
    ///获得每个实例的类别
    auto cate_labels=inds.index({"...",1});
    ///获得满足阈值的kernel预测
    auto pred_index=inds.index({"...",0});
    auto kernel_preds=kernel_tensor.index({pred_index});

    ///计算每个实例的stride

    auto strides=torch::ones({kPredNum}, device);

    //计算各个层级上的实例的strides
    int index0=size_trans_[0].item().toInt();
    strides.index_put_({idx::Slice(idx::None,index0)}, det2d_para::kSoloStrides[0]);
    for(int i=1; i < kNumStage; ++i){
        int index_start=size_trans_[i - 1].item().toInt();
        int index_end=size_trans_[i].item().toInt();
        strides.index_put_({idx::Slice(index_start,index_end)}, det2d_para::kSoloStrides[i]);
    }
    //保留满足阈值的实例的strides
    strides=strides.index({pred_index});

    ///将mask_feat和kernel进行卷积
    auto seg_preds=feat_tensor.unsqueeze(0);
    //首先将kernel改变为1x1卷积核的形状
    kernel_preds=kernel_preds.view({kernel_preds.sizes()[0],kernel_preds.sizes()[1],1,1});
    //然后进行卷积
    seg_preds=torch::conv2d(seg_preds,kernel_preds,{},1);
    seg_preds=torch::squeeze(seg_preds,0).sigmoid();

    ///计算mask
    auto seg_masks= seg_preds > det2d_para::kSoloMaskThr;
    auto sum_masks=seg_masks.sum({1,2}).to(torch::kFloat);

    ///根据strides过滤掉像素点太少的实例
    auto keep=sum_masks > strides;
    if(keep.sum(0).item().toInt()==0){
        Warns("GetSegTensor | keep.sum(0) == 0");
        return ;
    }
    seg_masks = seg_masks.index({keep,"..."});
    seg_preds = seg_preds.index({keep,"..."});
    sum_masks = sum_masks.index({keep});
    cate_tensor = cate_tensor.index({keep});
    cate_labels = cate_labels.index({keep});

    ///根据mask预测设置实例的置信度
    auto seg_scores=(seg_preds * seg_masks.to(torch::kFloat)).sum({1,2}) / sum_masks;
    cate_tensor *= seg_scores;

    ///根据cate_score进行排序，用于NMS
    auto sort_inds = torch::argsort(cate_tensor,-1,true);
    if(sort_inds.sizes()[0] > det2d_para::kSoloNmsPre){
        sort_inds=sort_inds.index({idx::Slice(idx::None, det2d_para::kSoloNmsPre)});
    }
    seg_masks=seg_masks.index({sort_inds,"..."});
    seg_preds=seg_preds.index({sort_inds,"..."});
    sum_masks=sum_masks.index({sort_inds});
    cate_tensor=cate_tensor.index({sort_inds});
    cate_labels=cate_labels.index({sort_inds});

    Debugs("GetSegTensor | seg_masks.dims:{}", DimsToStr(seg_masks.sizes()));
    Debugs("GetSegTensor | cate_labels.dims:{}", DimsToStr(cate_labels.sizes()));

    ///执行Matrix NMS
    auto cate_scores = MatrixNMS(seg_masks,cate_labels,cate_tensor,sum_masks);

    ///根据新的置信度过滤结果
    keep = cate_scores >= det2d_para::kSoloUpdateThr;
    if(keep.sum(0).item().toInt() == 0){
        Warns("GetSegTensor | keep.sum(0) == 0");
        return ;
    }
    seg_preds = seg_preds.index({keep,"..."});
    cate_scores = cate_scores.index({keep});
    cate_labels = cate_labels.index({keep});
    sum_masks = sum_masks.index({keep});

    for(int i=0;i<cate_scores.sizes()[0];++i){
        Debugs("id:{},cls:{},prob:{}", i, coco::CocoLabel[cate_labels[i].item().toInt()],
               cate_scores[i].item().toFloat());
    }

    ///再次根据置信度进行排序
    sort_inds = torch::argsort(cate_scores,-1,true);
    if(sort_inds.sizes()[0] > det2d_para::kSoloMaxPerImg){
        sort_inds=sort_inds.index({idx::Slice(idx::None, det2d_para::kSoloMaxPerImg)});
    }
    seg_preds=seg_preds.index({sort_inds,"..."});
    cate_scores=cate_scores.index({sort_inds});
    cate_labels=cate_labels.index({sort_inds});
    sum_masks = sum_masks.index({sort_inds});

    Debugs("GetSegTensor | seg_preds.dims:{}", DimsToStr(seg_preds.sizes()));

    ///对mask进行双线性上采样,
    static auto options=InterpolateFuncOptions().mode(torch::kBilinear).align_corners(true);
    auto op1=options.size(std::vector<int64_t>({kFeatHeight * 4, kFeatWidth * 4}));
    seg_preds = torch::nn::functional::interpolate(seg_preds.unsqueeze(0),op1);

    ///对mask进行裁切、缩放，得到原始图片大小的mask
    seg_preds =seg_preds.index({"...",Slice(img_info.rect_y,img_info.rect_y+img_info.rect_h),
                                Slice(img_info.rect_x,img_info.rect_x+img_info.rect_w)});

    auto op2=options.size(std::vector<int64_t>({img_info.origin_h, img_info.origin_w}));
    seg_preds = torch::nn::functional::interpolate(seg_preds,op2);

    seg_preds=seg_preds.squeeze(0);

    ///阈值化
    mask_tensor = seg_preds > det2d_para::kSoloMaskThr;

    /*cout<<"cate_labels.sizes"<<cate_labels.sizes()<<endl;
    cout<<"cate_scores.sizes"<<cate_scores.sizes()<<endl;
    cout<<"sum_masks.sizes"<<sum_masks.sizes()<<endl;
    cout<<"seg_masks.sizes"<<mask_tensor.sizes()<<endl;*/

    ///根据mask计算包围框
    for(int i=0;i<mask_tensor.sizes()[0];++i){
        auto nz=mask_tensor[i].nonzero();
        auto max_xy =std::get<0>( torch::max(nz,0) );
        auto min_xy =std::get<0>( torch::min(nz,0) );

        Box2D::Ptr inst = std::make_shared<Box2D>();
        inst->id = i;

        int coco_id = cate_labels[i].item().toInt();
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

        inst->max_pt.x = max_xy[1].item().toInt();
        inst->max_pt.y = max_xy[0].item().toInt();
        inst->min_pt.x = min_xy[1].item().toInt();
        inst->min_pt.y = min_xy[0].item().toInt();
        inst->rect = cv::Rect2f(inst->min_pt,inst->max_pt);
        inst->score = cate_scores[i].item().toFloat();
        insts.push_back(inst);
    }
}


}

