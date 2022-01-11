/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "viode_utils.h"
#include "featureTracker/feature_utils.h"

namespace dynamic_vins{\

namespace VIODE{

/**
 * 根据VIODE的seg0设置背景掩码img.merge_mask、img.merge_mask_gpu 和 物体掩码img.inv_merge_mask_gpu、img.inv_merge_mask_gpu
 * @param img
 */
void SetViodeMaskSimple(SegImage &img)
{
    auto &semantic_img = img.seg0;
    cv::Mat merge_mask = cv::Mat(semantic_img.rows, semantic_img.cols, CV_8UC1, cv::Scalar(0));
    /*
     //低效的方法
     for (int i = 0; i < semantic_img.rows; i++) {
            uchar* row_ptr = semantic_img.data + i * semantic_img.step;
            uchar* semantic_ptr=semantic_mask.data+i*semantic_mask.step;
            for (int j = 0; j < semantic_img.cols; j++) {
                //将像素值转换为label_ID,
                unsigned int key= PixelToKey(row_ptr);
                int label_id=Config::ViodeKeyToIndex[key];//key的计算公式r*1000000+g*1000+b
                //判断该点是否是动态物体点
                if(Config::ViodeDynamicIndex.count(label_id)!=0){
                    semantic_ptr[0]=0;
                }
                row_ptr += 3;
                semantic_ptr+=1;
            }
        }
     */
    //同时遍历单通道图像mask和BGR图像semantic_img
    auto calBlock=[&](size_t row_start,size_t row_end){
        for (size_t i = row_start; i < row_end; ++i) {
            uchar* seg_ptr = semantic_img.data + i * semantic_img.step;
            uchar* merge_ptr=merge_mask.data +i*merge_mask.step;
            for (int j = 0; j < semantic_img.cols; ++j) {
                if(VIODE::IsDynamic(seg_ptr))
                    merge_ptr[0]=255;
                seg_ptr += 3;
                merge_ptr+=1;
            }
        }
    };
    ///两线程并行
    auto half_row=semantic_img.rows/2;
    std::thread block_thread(calBlock,0,half_row);
    calBlock(half_row,semantic_img.rows);
    block_thread.join();

    img.merge_mask = merge_mask;
    img.merge_mask_gpu.upload(merge_mask);
    cv::cuda::bitwise_not(img.merge_mask_gpu,img.inv_merge_mask_gpu);
    img.inv_merge_mask_gpu.download(img.inv_merge_mask);
}



/**
 * 根据VIODE数据集的seg图像，设置背景掩码img.merge_mask、img.merge_mask_gpu 和 物体掩码img.inv_merge_mask_gpu、img.inv_merge_mask_gpu
 * 以及对每个物体，设置其背景mask，并进行形态学滤波
 * @param img
 */
void SetViodeMask(SegImage &img)
{
    struct MiniInstance{
        MiniInstance()=default;
        MiniInstance(int row_start_,int row_end_,int col_start_,int col_end_):
        row_start(row_start_),row_end(row_end_),col_start(col_start_),col_end(col_end_){
            mask=cv::Mat(row_end-row_start,col_end-col_start,CV_8UC1,cv::Scalar(0));
        }
        cv::Mat mask;
        size_t num_pixel{0};
        int row_start{},row_end{},col_start{},col_end{};
    };

    static TicToc tt;

    int img_row=img.seg0.rows;
    int img_col=img.seg0.cols;

    cv::Mat merge_mask = cv::Mat(img_row, img_col, CV_8UC1, cv::Scalar(0));

    auto calBlock=[&merge_mask,&img](int row_start,int row_end,int col_start,int col_end,
            std::unordered_map<unsigned int,MiniInstance> *blockInsts){
        for (int i = row_start; i < row_end; ++i) {
            uchar* seg_ptr = img.seg0.data + i * img.seg0.step + col_start*3;
            uchar* merge_ptr=merge_mask.data + i * merge_mask.step + col_start*3;
            for (int j = col_start; j < col_end; ++j) {
                if(auto key= PixelToKey(seg_ptr);VIODE::IsDynamic(key)){
                    merge_ptr[0]=255;
                    if(blockInsts->count(key)==0){//创建实例
                        MiniInstance inst(row_start,row_end,col_start,col_end);
                        blockInsts->insert({key,inst});
                    }
                    //设置实例的mask
                    (*blockInsts)[key].mask.at<uchar>(i-row_start,j-col_start)=255;
                }
                seg_ptr += 3;
                merge_ptr+=1;
            }
        }
    };

    tt.Tic();

    auto *insts1=new std::unordered_map<unsigned int,MiniInstance>;
    auto *insts2=new std::unordered_map<unsigned int,MiniInstance>;
    auto *insts3=new std::unordered_map<unsigned int,MiniInstance>;
    auto *insts4=new std::unordered_map<unsigned int,MiniInstance>;

    ///4线程并行
    auto half_row=img_row/2, half_col=img_col/2;
    std::thread block_thread1(calBlock, 0,          half_row,      0,          half_col,insts1);
    std::thread block_thread2(calBlock, half_row,   img_row,       0,          half_col,insts2);
    std::thread block_thread3(calBlock, 0,          half_row,      half_col,   img_col, insts3);
    calBlock(                           half_row,   img_row,       half_col,   img_col, insts4);

    block_thread1.join();
    block_thread2.join();
    block_thread3.join();

    Debugs("setViodeMask calBlock :{} ms", tt.TocThenTic());

    ///线程结果合并
    std::unordered_multimap<unsigned int,MiniInstance> insts_all;
    insts_all.insert(insts1->begin(),insts1->end());
    insts_all.insert(insts2->begin(),insts2->end());
    insts_all.insert(insts3->begin(),insts3->end());
    insts_all.insert(insts4->begin(),insts4->end());

    std::unordered_map<unsigned int,MiniInstance> insts;
    for(auto &[key,m_inst]: insts_all){
        if(insts.count(key)==0){
            MiniInstance inst(0,img_row,0,img_col);
            insts.insert({key,inst});
        }
        auto block=insts[key].mask(cv::Range(m_inst.row_start, m_inst.row_end),
                                   cv::Range(m_inst.col_start, m_inst.col_end));
        m_inst.mask.copyTo(block);
    }

    Debugs("setViodeMask merge :{} ms", tt.TocThenTic());

    ///构建InstanceInfo
    for(auto &[key,inst] : insts){
        InstInfo info;
        info.id = key;
        info.track_id=key;
        Debugs("id:{}", key);
        Debugs("mask:{} {} {}", inst.mask.empty(), inst.mask.rows, inst.mask.cols);
        info.mask_gpu.upload(inst.mask);
        ErodeMaskGpu(info.mask_gpu, info.mask_gpu);
        info.mask_gpu.download(info.mask_cv);
        img.insts_info.emplace_back(info);
    }

    Debugs("erode_filter:{}", tt.TocThenTic());
    img.merge_mask = merge_mask;
    img.merge_mask_gpu.upload(merge_mask);
    cv::cuda::bitwise_not(img.merge_mask_gpu,img.inv_merge_mask_gpu);
    img.inv_merge_mask_gpu.download(img.inv_merge_mask);

    Debugs("setViodeMask set gpu :{} ms", tt.TocThenTic());

    delete insts1;
    delete insts2;
    delete insts3;
    delete insts4;
}


}
}