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
#include <pcl/visualization/cloud_viewer.h>

#include "basic/def.h"
#include "utils/file_utils.h"

using namespace std;

namespace dynamic_vins{\

using PointT=pcl::PointXYZRGB;
using PointCloud=pcl::PointCloud<PointT>;

using namespace visualization_msgs;


ros::NodeHandle *nh;

std::unordered_map<string,ros::Publisher> pub_map;
std::shared_ptr<ros::Publisher> pub_instance_marker;

Vec3d target(10,10,0);


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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr EigenToPCL(vector<Vec3d> &points){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    const double max_dist = 5.;
    for(int i=0;i<points.size();++i){
        pcl::PointXYZRGB point;
        point.x = points[i].x();
        point.y = points[i].y();
        point.z = points[i].z();

        double d = (points[i]-target).norm();
        d = std::min(max_dist,d);
        //cout<<d*255./10.5<<endl;
        point.r = uint8_t(d*255./max_dist);
        point.g = 255 - uint8_t(d*255./max_dist);
        point.b=255;
        point_cloud_ptr->points.push_back (point);
    }

    point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
    point_cloud_ptr->height = 1;

    return point_cloud_ptr;
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




int Run(int argc, char **argv){
    ros::init(argc, argv, "read_pub_pointcloud");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    pub_instance_marker=std::make_shared<ros::Publisher>();
    *pub_instance_marker=n.advertise<MarkerArray>("/dynamic_vins/instance_marker", 1000);

    nh = &n;

    const string inst_pc_path="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/point_cloud_temp/";
    vector<string> pcd_files;

    GetAllFiles(inst_pc_path, pcd_files,".pcd") ;
    std::sort(pcd_files.begin(),pcd_files.end());

    vector<string> left_names;
    GetAllImagePaths("/home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/0000/", left_names);
    std::sort(left_names.begin(),left_names.end());

    ImageViewer image_viewer;
    //pcl::visualization::CloudViewer viewer ("test");

    int pc_index=0;
    int img_index=0;
    ros::Rate rate(10);
    while(ros::ok()){
        if(img_index >= left_names.size()){
            break;
        }

        ///读取并显示图像
        cv::Mat img = cv::imread(left_names[img_index]);
        img_index++;

        ///判断该帧是否有点云
        fs::path pc_file_path(pcd_files[pc_index]);
        vector<string> tokens;
        split(pc_file_path.stem().string(),tokens,"_");
        fs::path img_file_path(left_names[img_index]);

        cout<<img_file_path.stem().string()<<"---"<<tokens[0]<<endl;
        if(img_file_path.stem().string() != tokens[0]){
            continue;
        }

        image_viewer.ImageShow(img,50,0);

        for(int i=0;i<3;++i){

            if(pc_index>130){
                std::this_thread::sleep_for(3000ms);
            }
            else{
                std::this_thread::sleep_for(100ms);
            }

            ///读取并显示点云
            //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
            cout<<pcd_files[pc_index]<<endl;
            if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_files[pc_index], *cloud) == -1) {
                PCL_ERROR("Couldn't read file rabbit.pcd\n");
                return -1;
                //return(-1);
            }
            pc_index++;

            PointCloud::Ptr cloud_rgb(new PointCloud);
            cloud_rgb->points.reserve(cloud->points.size());
            for(int i=0;i<cloud->points.size();++i){
                PointT p(255,128,0);
                p.x = cloud->points[i].x;
                p.y = cloud->points[i].y;
                p.z = cloud->points[i].z;
                cloud_rgb->points.push_back(p);
            }

            Pub(*cloud_rgb,"/dynamic_vins/instance_point_cloud");

            //viewer.showCloud(cloud_rgb);
            //while (!viewer.wasStopped()){ };

            ros::spinOnce();
            rate.sleep();
        }


    }

    //viewer.showCloud(point_cloud_ptr);
    //while (!viewer.wasStopped()){ };

    return 0;
}





}


int main(int argc, char **argv)
{
    return dynamic_vins::Run(argc,argv);
}






