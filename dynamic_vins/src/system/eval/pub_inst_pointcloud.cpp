/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <memory>
#include <thread>
#include <mutex>
#include <chrono>
#include <future>
#include <iostream>

#include <spdlog/spdlog.h>
#include <ros/ros.h>

#include "basic/def.h"
#include "utils/parameters.h"
#include "utils/io/visualization.h"
#include "utils/io/publisher_map.h"
#include "utils/io/io_parameters.h"
#include "utils/io/feature_serialization.h"
#include "utils/dataset/viode_utils.h"
#include "utils/dataset/coco_utils.h"
#include "utils/io/dataloader.h"
#include "utils/io/image_viewer.h"
#include "utils/io/publisher_map.h"
#include "basic/semantic_image.h"
#include "basic/semantic_image_queue.h"
#include "basic/feature_queue.h"
#include "estimator/estimator.h"
#include "front_end/background_tracker.h"
#include "front_end/front_end_parameters.h"
#include "image_process/image_process.h"
#include "utils/io/output.h"
#include "utils/convert_utils.h"
#include "utils/io/markers_utils.h"

namespace dynamic_vins{\

namespace fs=std::filesystem;
using namespace std;

Estimator::Ptr estimator;
ImageProcessor::Ptr processor;
Dataloader::Ptr dataloader;

SemanticImageQueue image_queue;


PointCloud::Ptr DetectExtraPoints(Box2D::Ptr box2d,const cv::Mat& disp){
    PointCloud::Ptr pc(new PointCloud);
    if(!(box2d->roi) || disp.empty()){
        return pc;
    }

    const int rows=box2d->roi->roi_gray.rows;
    const int cols=box2d->roi->roi_gray.cols;

    //设置采样步长
    constexpr float N_max = 1000.;
    const int step = std::max(std::sqrt(0.8 * rows * cols / N_max),2.);

    for(int i=0;i<rows;i+=step){
        for(int j=0;j<cols;j+=step){
            if(box2d->roi->mask_cv.at<uchar>(i,j)<=0.5){
                continue;
            }
            const int r=i + box2d->rect.tl().y;
            const int c=j + box2d->rect.tl().x;
            const float disparity = disp.at<float>(r,c);
            if(disparity<=0){
                continue;
            }
            if(disparity!=disparity){ //disparity is nan
                continue;
            }

            ///TODO 注意，这里未实现畸变矫正，因此需要输入的图像无畸变

            const float depth = cam_s.fx0 * cam_s.baseline / disparity;//根据视差计算深度

            if(depth<=0.1 || depth>100){
                continue;
            }

            const float x_3d = (c - cam_s.cx0)*depth/cam_s.fx0;
            const float y_3d = (r - cam_s.cy0)*depth/cam_s.fy0;

            if(i==j){
                Debugt("p3d:{} {} {} disp:{}",x_3d,y_3d,depth,disparity);
            }
            PointT p(255,255,255);
            p.x = x_3d;
            p.y = y_3d;
            p.z = depth;
            pc->push_back(p);
        }
    }
    return pc;
}




/**
 * 构建额外点，并进行处理
 * 里面执行点云滤波和分割
 */
void ProcessExtraPoints(SemanticImage &img){
    TicToc t_all;

    pcl::RadiusOutlierRemoval<PointT> radius_filter;
    radius_filter.setRadiusSearch(0.5);
    radius_filter.setMinNeighborsInRadius(10);//一米内至少有10个点

    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (1.); //设置近邻搜索的搜索半径为1.0m
    ec.setMinClusterSize (10);//设置一个聚类需要的最少点数目为100
    ec.setMaxClusterSize (25000); //设置一个聚类需要的最大点数目为25000

    PointCloud::Ptr all_pc(new PointCloud);

    for(auto &box2d:img.boxes2d){

        Debugt("ProcessExtraPoints() start t_filter");

        ///检测额外点,构建点云
        PointCloud::Ptr pc=DetectExtraPoints(box2d,img.disp);
        pc->width = pc->points.size();
        pc->height=1;

        ///半径滤波
        PointCloud::Ptr pc_filtered(new PointCloud);
        radius_filter.setInputCloud(pc);
        radius_filter.filter(*pc_filtered);

        if(pc_filtered->empty() || pc_filtered->points.size()<5){
            return;
        }

        pc_filtered->width = pc_filtered->points.size();
        pc_filtered->height=1;
        Debugt("ProcessExtraPoints() end t_filter");

        pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
        ec.setSearchMethod (tree);//设置点云的搜索机制
        std::vector<pcl::PointIndices> cluster_indices;
        ec.setInputCloud (pc_filtered);
        ec.extract (cluster_indices);//从点云中提取聚类，并将点云索引保存在cluster_indices中

        if(cluster_indices.empty()){
            return;
        }

        ///选择第一个簇作为分割结果
        PointCloud::Ptr segmented_pc(new PointCloud);
        auto &indices = cluster_indices[0].indices;
        segmented_pc->points.reserve(indices.size());
        for(auto &index:indices){
            segmented_pc->points.push_back(pc_filtered->points[index]);
        }
        segmented_pc->width = segmented_pc->points.size();
        segmented_pc->height=1;
        segmented_pc->is_dense = true;

        (*all_pc) += (*segmented_pc);
    }

    PublisherMap::PubPointCloud((*all_pc),"instance_point_cloud");

    Debugt("ProcessExtraPoints() used time:{} ms",t_all.Toc());
}







/**
 * 前端线程,包括同步图像流,实例分割,光流估计,3D目标检测等
 */
void ImageProcess()
{
    int cnt = 0;
    double time_sum=0;
    TicToc t_all;
    ImageViewer viewer;

    while(cfg::ok==true){

        if(image_queue.size() >= kImageQueueSize){
            cerr<<"ImageProcess image_queue.size() >= kImageQueueSize,blocked"<<endl;
            std::this_thread::sleep_for(50ms);
            continue;
        }
        Debugs("Start sync");

        ///同步获取图像
        SemanticImage img;
        img = dataloader->LoadStereo();

        std::cout<<"image seq_id:"<<img.seq<<std::endl;
        Warns("----------Time : {} ----------", std::to_string(img.time0));
        t_all.Tic();

        ///结束程序
        if(img.color0.empty()){
            cfg::ok = false;
            ros::shutdown();
            break;
        }

        if(img.color0.rows!=cfg::kInputHeight || img.color0.cols!=cfg::kInputWidth){
            cerr<<fmt::format("The input image sizes is:{}x{},but config size is:{}x{}",
                              img.color0.rows,img.color0.cols,cfg::kInputHeight,cfg::kInputWidth)<<endl;
            std::terminate();
        }

        ///入口程序
        processor->Run(img);


        double time_cost=t_all.Toc();
        time_sum+=time_cost;
        cnt++;

        Warns("ImageProcess, all:{} ms \n",time_cost);

        MarkerArray marker_array;

        if(io_para::is_show_input){
            cv::Mat img_show;
            cv::cvtColor(img.inv_merge_mask,img_show,CV_GRAY2BGR);
            cv::scaleAdd(img_show,0.3,img.color0,img_show);

            ///3D目标检测结果可视化
            int id=0;
            for(auto &box:img.boxes3d){
                box->VisCorners2d(img_show, cv::Scalar(255, 255, 255),cam_s.cam0);

                auto cube_marker = CubeMarker(box->corners,id++, BgrColor("green"));
                marker_array.markers.push_back(cube_marker);
            }
            viewer.ImageShow(img_show,io_para::kImageDatasetPeriod,1);
        }
        else{
            viewer.Delay(io_para::kImageDatasetPeriod);
        }

        ProcessExtraPoints(img);

        PublisherMap::PubMarkers(marker_array,"/pub_object3d/fcos3d_markers");
    }

    Infos("Image Process Avg cost:{} ms",time_sum/cnt);
    Warns("ImageProcess 线程退出");
}



int Run(int argc, char **argv){
    ros::init(argc, argv, "dynamic_vins");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    string file_name = argv[1];
    cout<<fmt::format("cfg_file:{}",file_name)<<endl;

    string seq_name = argv[2];
    cout<<fmt::format("seq_name:{}",seq_name)<<endl;

    try{
        cfg cfg(file_name,seq_name);
        ///初始化logger
        MyLogger::InitLogger(file_name);
        ///初始化相机模型
        cout<<"start init camera"<<endl;
        InitCamera(file_name,seq_name);
        ///初始化局部参数
        cout<<"start init dataset parameters"<<endl;
        coco::SetParameters(file_name);
        if(cfg::dataset == DatasetType::kViode){
            VIODE::SetParameters(file_name);
        }
        cout<<"start init io"<<endl;
        io_para::SetParameters(file_name,seq_name);

        cout<<"start init three threads"<<endl;

        estimator.reset(new Estimator(file_name));
        processor.reset(new ImageProcessor(file_name,seq_name));

    }
    catch(std::runtime_error &e){
        cerr<<e.what()<<endl;
        return -1;
    }
    estimator->SetParameter();

    cout<<"init completed"<<endl;
    dataloader = std::make_shared<Dataloader>(io_para::kImageDatasetLeft,io_para::kImageDatasetRight);

    Publisher::e = estimator;
    Publisher::RegisterPub(n);

    MyLogger::vio_logger->flush();
    cout<<"waiting for image and imu..."<<endl;

    std::thread sync_thread{ImageProcess};

    ros::spin();

    cfg::ok= false;
    cv::destroyAllWindows();

    sync_thread.join();
    spdlog::drop_all();

    cout<<"vins结束"<<endl;

    return 0;
}


}

int main(int argc, char **argv)
{
    if(argc != 3){
        std::cerr<<"please input: rosrun dynamic_vins pub_inst_pointcloud [cfg file] [seq_name]"<< std::endl;
        return 1;
    }

    return dynamic_vins::Run(argc,argv);
}



