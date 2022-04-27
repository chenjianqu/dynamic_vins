/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "viode_utils.h"

#include "front_end/feature_utils.h"

namespace dynamic_vins{

/**
 * 根据VIODE的seg0设置背景掩码img.merge_mask、img.merge_mask_gpu 和 物体掩码img.inv_merge_mask_gpu、img.inv_merge_mask_gpu
 * @param img
 */
void VIODE::SetViodeMaskSimple(SegImage &img)
{
    img.exist_inst=true;

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
void VIODE::SetViodeMask(SegImage &img)
{
    struct InstanceSimple{
        InstanceSimple()=default;
        InstanceSimple(int row_start_, int row_end_, int col_start_, int col_end_):
        row_start(row_start_),row_end(row_end_),col_start(col_start_),col_end(col_end_){
            mask=cv::Mat(row_end-row_start,col_end-col_start,CV_8UC1,cv::Scalar(0));
        }
        cv::Mat mask;
        size_t num_pixel{0};
        int row_start{},row_end{},col_start{},col_end{};
        int row_min,row_max,col_min,col_max;
    };

    static TicToc tt;
    int img_row=img.seg0.rows;
    int img_col=img.seg0.cols;
    cv::Mat merge_mask = cv::Mat(img_row, img_col, CV_8UC1, cv::Scalar(0));

    tt.Tic();
    std::unordered_map<unsigned int,InstanceSimple> insts;


    auto calBlock=[&merge_mask,&img](int row_start,int row_end,int col_start,int col_end,
            std::unordered_map<unsigned int,InstanceSimple> *blockInsts){
        std::unordered_map<unsigned int, InstanceSimple>::iterator it;
        for (int i = row_start; i < row_end; ++i) {
            uchar* seg_ptr = img.seg0.data + i * img.seg0.step + col_start*3;
            uchar* merge_ptr=merge_mask.data + i * merge_mask.step + col_start*3;
            for (int j = col_start; j < col_end; ++j) {
                if(auto key= PixelToKey(seg_ptr);VIODE::IsDynamic(key)){
                    merge_ptr[0]=255;
                    it=blockInsts->find(key);
                    int r=i-row_start;
                    int c=j-col_start;
                    if(it==blockInsts->end()){//创建实例
                        InstanceSimple inst(row_start, row_end, col_start, col_end);
                        blockInsts->insert({key,inst});
                        it=blockInsts->find(key);
                        it->second.mask.at<uchar>(r,c)=255;
                        it->second.row_min = r;
                        it->second.row_max = r;
                        it->second.col_min = c;
                        it->second.col_max = c;
                    }
                    else{//设置实例的mask
                        it->second.mask.at<uchar>(r,c)=255;
                        it->second.row_min = std::min(it->second.row_min,r);
                        it->second.row_max = std::max(it->second.row_max,r);
                        it->second.col_min = std::min(it->second.col_min,c);
                        it->second.col_max = std::max(it->second.col_max,c);
                    }
                }
                seg_ptr += 3;
                merge_ptr+=1;
            }
        }
    };


    bool use_multi_thread = false;
    if(use_multi_thread){
        auto *insts1=new std::unordered_map<unsigned int,InstanceSimple>;
        auto *insts2=new std::unordered_map<unsigned int,InstanceSimple>;
        auto *insts3=new std::unordered_map<unsigned int,InstanceSimple>;
        auto *insts4=new std::unordered_map<unsigned int,InstanceSimple>;

        //4线程并行
        auto half_row=img_row/2, half_col=img_col/2;
        std::thread block_thread1(calBlock, 0,          half_row,      0,          half_col,insts1);
        std::thread block_thread2(calBlock, half_row,   img_row,       0,          half_col,insts2);
        std::thread block_thread3(calBlock, 0,          half_row,      half_col,   img_col, insts3);
        calBlock(                           half_row,   img_row,       half_col,   img_col, insts4);

        block_thread1.join();
        block_thread2.join();
        block_thread3.join();
        Debugs("SetViodeMask calBlock :{} ms", tt.TocThenTic());

        std::unordered_multimap<unsigned int,InstanceSimple> insts_all;//线程结果合并
        insts_all.insert(insts1->begin(),insts1->end());
        insts_all.insert(insts2->begin(),insts2->end());
        insts_all.insert(insts3->begin(),insts3->end());
        insts_all.insert(insts4->begin(),insts4->end());

        for(auto &[key,m_inst]: insts_all){
            if(insts.count(key)==0){
                InstanceSimple inst(0, img_row, 0, img_col);
                insts.insert({key,inst});
            }
            auto block=insts[key].mask(cv::Range(m_inst.row_start, m_inst.row_end),
                                       cv::Range(m_inst.col_start, m_inst.col_end));
            m_inst.mask.copyTo(block);
        }
        delete insts1;
        delete insts2;
        delete insts3;
        delete insts4;
    }
    else{
        calBlock(0, img_row, 0, img_col, & insts);
    }

    Debugs("SetViodeMask merge :{} ms", tt.TocThenTic());
    Debugs("SetViodeMask detect num:{}",insts.size());

    ///构建InstanceInfo
    for(auto &[key,inst] : insts){
        InstInfo info;
        info.id = key;
        info.track_id=key;
        Debugs("SetViodeMask id:{}", key);
        info.mask_gpu.upload(inst.mask);
        ErodeMaskGpu(info.mask_gpu, info.mask_gpu);
        info.mask_gpu.download(info.mask_cv);
        info.min_pt = cv::Point2f(inst.col_min,inst.row_min);
        info.max_pt = cv::Point2f(inst.col_max,inst.row_max);
        img.insts_info.push_back(info);
        Debugs("SetViodeMask max_pt:(c{},r{}), min_pt:(c{},r{})", inst.col_min,inst.row_min, inst.col_max,inst.row_max);
    }

    img.exist_inst = !img.insts_info.empty();

    Debugs("SetViodeMask erode time:{} ms", tt.TocThenTic());
    img.merge_mask = merge_mask;
    img.merge_mask_gpu.upload(merge_mask);
    cv::cuda::bitwise_not(img.merge_mask_gpu,img.inv_merge_mask_gpu);
    img.inv_merge_mask_gpu.download(img.inv_merge_mask);

    Debugs("SetViodeMask set gpu :{} ms", tt.TocThenTic());


}


/**
 * 读取VIODE数据集的rgb_ids.txt
 * @param rgb_to_label_file
 * @return
 */
std::unordered_map<unsigned int,int> VIODE::ReadViodeRgbIds(const string &rgb_to_label_file){
    vector<vector<int>> label_data;
    std::ifstream fp(rgb_to_label_file); //定义声明一个ifstream对象，指定文件路径
    if(!fp.is_open()){
        throw std::runtime_error(fmt::format("Can not open:{}", rgb_to_label_file));
    }
    string line;
    getline(fp,line); //跳过列名，第一行不做处理
    while (getline(fp,line)){ //循环读取每行数据
        vector<int> data_line;
        string number;
        std::istringstream read_str(line); //string数据流化
        for(int j = 0;j < 4;j++){ //可根据数据的实际情况取循环获取
            getline(read_str,number,','); //将一行数据按'，'分割
            data_line.push_back(atoi(number.c_str())); //字符串传int
        }
        label_data.push_back(data_line); //插入到vector中
    }
    fp.close();

    std::unordered_map<unsigned int,int> rgb_to_key;
    for(const auto& v : label_data){
        rgb_to_key.insert(std::make_pair(VIODE::PixelToKey(v[1], v[2], v[3]), v[0]));
    }
    return rgb_to_key;
}


void VIODE::SetParameters(const std::string &config_path)
{
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(std::string("ERROR: Wrong path to settings:" + config_path));
    }

    std::string kBasicDir;
    fs["basic_dir"] >> kBasicDir;

    ///读取VIODE动态物体对应的Label Index
    cv::FileNode labelIDNode=fs["dynamic_label_id"];
    for(auto && it : labelIDNode){
        ViodeDynamicIndex.insert((int)it);
    }
    ///设置VIODE的RGB2Label
    string rgb2label_file;
    fs["rgb_to_label_file"]>>rgb2label_file;
    rgb2label_file = kBasicDir + rgb2label_file;
    ViodeKeyToIndex = VIODE::ReadViodeRgbIds(rgb2label_file);

    fs.release();

}



}