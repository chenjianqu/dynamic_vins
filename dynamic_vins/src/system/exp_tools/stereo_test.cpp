/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <thread>
#include <iostream>
#include <optional>
#include <random>

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>

#include <opencv2/opencv.hpp>

#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/cloud_viewer.h>

#include "utils/def.h"
#include "utils/io_utils.h"

using namespace std;

namespace dynamic_vins{\

using PointT=pcl::PointXYZRGB;
using PointCloud=pcl::PointCloud<PointT>;

using namespace visualization_msgs;


ros::NodeHandle *nh;

std::unordered_map<string,ros::Publisher> pub_map;
std::shared_ptr<ros::Publisher> pub_instance_marker;



void Pub(PointCloud &cloud,const string &topic){
    if(pub_map.find(topic)==pub_map.end()){
        ros::Publisher pub = nh->advertise<sensor_msgs::PointCloud2>(topic,10);
        pub_map.insert({topic,pub});
    }

    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time::now();

    sensor_msgs::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(cloud,point_cloud_msg);
    point_cloud_msg.header = header;

    pub_map[topic].publish(point_cloud_msg);
}


class ImageViewer{
public:
    ImageViewer(){
        tt.Tic();
    }
    void ImageShow(cv::Mat &img,int period, int delay_frames=0);

    void Delay(int period);

    std::queue<cv::Mat> img_queue;

    TicToc tt;
};


void ImageViewer::Delay(int period){
    int delta_t =(int) tt.TocThenTic();
    int wait_time = period - delta_t;
    if(wait_time>0){
        std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
    }
}


void ImageViewer::ImageShow(cv::Mat &img,int period, int delay_frames){
    int delta_t =(int) tt.TocThenTic();
    int wait_time = period - delta_t;
    wait_time = std::max(wait_time,1);

    img_queue.push(img);//使用队列存储图片

    if(img_queue.size()>delay_frames){
        img = img_queue.front();
        img_queue.pop();

        bool pause=false;
        do{
            cv::imshow("ImageViewer",img);
            int key = cv::waitKey(wait_time);
            if(key ==' '){
                pause = !pause;
            }
            else if(key== 27){ //ESC
                ros::shutdown();
                pause=false;
            }
            wait_time=period;

        } while (pause);

    }
}


struct SemanticImage{
    SemanticImage()= default;

    cv::Mat color0,seg0,color1,seg1;
    double time0,seg0_time,time1,seg1_time;
    cv::Mat gray0,gray1;

    cv::Mat disp;//视差图

    unsigned int seq;
};


class Dataloader{
public:
    using Ptr = std::shared_ptr<Dataloader>;
    Dataloader(const string &kImageDatasetLeft,
               const string &kImageDatasetRight,
               const string &kImageDatasetStereo);

    //获取一帧图像
    SemanticImage LoadStereo();

private:
    vector<string> left_names;
    vector<string> right_names;
    vector<string> stereo_names;

    int index{0};
    double time{0};

};


Dataloader::Dataloader(const string &kImageDatasetLeft,
                       const string &kImageDatasetRight,
                       const string &kImageDatasetStereo){

    GetAllImageFiles(kImageDatasetLeft,left_names);
    GetAllImageFiles(kImageDatasetRight,right_names);
    GetAllImageFiles(kImageDatasetStereo,stereo_names);

    cout<<"left:"<<left_names.size()<<endl;
    cout<<"right:"<<right_names.size()<<endl;
    if(left_names.size() != right_names.size()){
        cerr<< "left and right image number is not equal!"<<endl;
        return;
    }

    std::sort(left_names.begin(),left_names.end());
    std::sort(right_names.begin(),right_names.end());
    std::sort(stereo_names.begin(),stereo_names.end());

    index=0;
    time=0.;
}


SemanticImage Dataloader::LoadStereo()
{
    SemanticImage img;

    if(index >= left_names.size()){
        ros::shutdown();
        return img;
    }
    cout<<left_names[index]<<endl;

    img.color0  = cv::imread(left_names[index],-1);
    img.color1 = cv::imread(right_names[index],-1);

    cv::Mat disp_raw = cv::imread(stereo_names[index],-1);
    disp_raw.convertTo(img.disp, CV_32F,1./256.);

    std::filesystem::path name(left_names[index]);
    std::string name_stem =  name.stem().string();//获得文件名(不含后缀)

    img.time0 = time;
    img.time1 = time;
    img.seq = std::stoi(name_stem);

    time+=0.05; // 时间戳
    index++;

    return img;
}



void ReadStereo()
{
    const string stereo_path="/home/chen/datasets/kitti/tracking/stereo/training/0004";
    const string left_image_path="/home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/0004";
    const string right_image_path="/home/chen/datasets/kitti/tracking/data_tracking_image_3/training/image_03/0004";

    Dataloader dataloader(left_image_path,right_image_path,stereo_path);

    ImageViewer imageViewer;

    const float fx = 721.538,fy=721.538 ,cx=609.559, cy=172.854;
    const float baseline = 0.532725;

    while(true){
        SemanticImage img = dataloader.LoadStereo();
        if(img.color0.empty()){
            cerr<<"img.color0.empty()==true, finished"<<endl;
        }

        PointCloud pc;
        int rows = img.color0.rows;
        int cols = img.color0.cols;
        for(int i=0;i<rows;++i){
            for(int j=0;j<cols;++j){
                float disparity = img.disp.at<float>(i,j);
                if(disparity<=0)
                    continue;
                float depth =fx * baseline / disparity;//根据视差计算深度
                float x_3d = (j-cx)*depth/fx;
                float y_3d = (i-cy)*depth/fy;
                auto pixel = img.color0.at<cv::Vec3b>(i,j);
                PointT p(pixel[2],pixel[1],pixel[0]);
                p.x = x_3d;
                p.y = y_3d;
                p.z = depth;
                pc.points.push_back(p);
            }
        }

        cout<<pc.points.size()<<endl;

        Pub(pc,"/dynamic_vins/instance_point_cloud");

        imageViewer.ImageShow(img.color0,100);
    }


}




int Run(int argc, char **argv){
    ros::init(argc, argv, "ransac_test");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    pub_instance_marker=std::make_shared<ros::Publisher>();
    *pub_instance_marker=n.advertise<MarkerArray>("/dynamic_vins/instance_marker", 1000);

    nh = &n;

    std::thread t(&ReadStereo);

    ros::spin();

    //pcl::visualization::CloudViewer viewer ("test");
    //viewer.showCloud(point_cloud_ptr);
    //viewer.showCloud(point_cloud_ptr);
    //while (!viewer.wasStopped()){ };

    return 0;
}





}


int main(int argc, char **argv)
{
    return dynamic_vins::Run(argc,argv);
}






