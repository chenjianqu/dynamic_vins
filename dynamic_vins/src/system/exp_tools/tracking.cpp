/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "det2d/detector2d.h"
#include "mot/deep_sort.h"
#include "utils/def.h"


Infer::Ptr infer;
DeepSORT::Ptr tracker;


ros::Publisher pub_image_track;


void Img0Callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr= cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
    cv::Mat img= ptr->image.clone();

    TicToc ticToc;

    /*auto [masks,insts_info] = infer->forward(img);
    if(!masks.empty()){
        //cv::imshow("mask_img",masks[0]);
        cv::cvtColor(masks[0],masks[0],CV_GRAY2BGR);
        cv::add(img,masks[0],img);
    }*/

    torch::Tensor mask_tensor;
    std::vector<InstInfo> insts_info;
    infer->forward_tensor(img,mask_tensor,insts_info);

    ticToc.toc_print_tic("infer time");


    ticToc.tic();
    std::vector<cv::Rect_<float>> dets;
    auto trks = tracker->update(insts_info, img);
    fmt::print("tracker time:{} ms\n",ticToc.toc());


    cv::Mat img_raw = img.clone();
    //for(auto &inst: insts_info){
    //    cv::rectangle(img_raw,inst.min_pt,inst.max_pt,cv::Scalar(255,255,255),1);
    //}

    //for (auto &d:dets) {
    //    draw_bbox(img0, d);
    //}
    for (auto &t:trks) {
        draw_bbox(img_raw, t.rect, std::to_string(t.track_id), color_map(t.track_id));
        //draw_trajectories(img0, repo.get().at(t.id).trajectories, color_map(t.id));
    }


    /*if(!insts_info.empty()){
        auto merger_tensor = mask_tensor.sum(0).to(torch::kInt8) * 255;
            merger_tensor = merger_tensor.to(torch::kCPU);
        //merger_tensor =merger_tensor.clone();
        cout<<merger_tensor.sizes()<<endl;

        auto semantic_mask = cv::Mat(cv::Size(merger_tensor.sizes()[1],merger_tensor.sizes()[0]), CV_8UC1, merger_tensor.data_ptr()).clone();
        cv::cvtColor(semantic_mask,semantic_mask,CV_GRAY2BGR);
        cv::add(img,semantic_mask,img);
    }*/

    cv::imshow("img_raw",img_raw);

    switch (cv::waitKey(1) & 0xFF) {
        case 'q':
            ros::shutdown();
            break;
        default:
            break;
    }
    //sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(ptr->header, "bgr8", img).toImageMsg();
    //pub_image_track.publish(imgTrackMsg);
}


int main(int argc,char** argv)
{
    ros::init(argc, argv, "dynamic_vins_seg_test");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    if(argc != 2){
        printf("please intput: rosrun vins vins_node [config file] \n");
        return 1;
    }
    string config_file = argv[1];
    printf("config_file: %s\n", argv[1]);

    Config cfg(config_file);

    infer= std::make_shared<Infer>();

    std::array<int64_t, 2> orig_dim{int64_t(Config::ROW), int64_t(Config::COL)};
    tracker = std::make_unique<DeepSORT>(orig_dim);

    ros::Subscriber sub_img0 = n.subscribe(Config::IMAGE0_TOPIC, 100, img0_callback);
    pub_image_track = n.advertise<sensor_msgs::Image>("image_seg", 1000);


    cout<<"waiting images:"<<endl;
    ros::spin();

    return 0;

}


