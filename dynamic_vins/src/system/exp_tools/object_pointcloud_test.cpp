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
#include <pcl/registration/icp.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/visualization/cloud_viewer.h>

#include "utils/def.h"
#include "utils/file_utils.h"

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


/**
 * This function takes the reference of a 4x4 matrix and prints the rigid transformation (刚体变换)in an human readable (可读的) way.
 * %6.3f 是指：要输出的浮点数总位数（包括小数点）大于6位的话，按全宽输出，小于 6 位时，小数点后输出3位小数，右对齐，左边不足的位用空格填充
 */
void print4x4Matrix(const Eigen::Matrix4d &matrix) {
    printf("Rotation matrix :\n");
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
    printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
    printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
    printf("Translation vector :\n");
    printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}



int Run(int argc, char **argv){
    ros::init(argc, argv, "ransac_test");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    pub_instance_marker=std::make_shared<ros::Publisher>();
    *pub_instance_marker=n.advertise<MarkerArray>("/dynamic_vins/instance_marker", 1000);
    nh = &n;


    const string inst_pc_path="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/0000/point_cloud/";
    vector<string> pcd_files;

    GetAllFiles(inst_pc_path, pcd_files,".pcd") ;

    std::sort(pcd_files.begin(),pcd_files.end());

    ///ICP算法
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    // Set the max correspondence distance to 10cm (e.g., correspondences with higher distances will be ignored)
    //icp.setMaxCorrespondenceDistance (0.1);
    // Set the maximum number of iterations (criterion 1)
    //icp.setMaximumIterations (50);
    // Set the transformation epsilon (criterion 2)
    //icp.setTransformationEpsilon (1e-8);
    // Set the euclidean distance difference epsilon (criterion 3)
    //icp.setEuclideanFitnessEpsilon (1);

    ///聚类分割
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (1.); //设置近邻搜索的搜索半径为1.0m
    ec.setMinClusterSize (10);//设置一个聚类需要的最少点数目为100
    ec.setMaxClusterSize (25000); //设置一个聚类需要的最大点数目为25000

    int pc_index=0;

    PointCloud::Ptr last_cloud;
    PointCloud::Ptr inliners(new PointCloud);

    ros::Rate rate(10);
    while(ros::ok()){
        if(pc_index >= pcd_files.size()){
            break;
        }

        //pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        PointCloud::Ptr cloud(new PointCloud);
        if (pcl::io::loadPCDFile<PointT>(pcd_files[pc_index], *cloud) == -1) {
            PCL_ERROR("Couldn't read file rabbit.pcd\n");
            return -1;
            //return(-1);
        }
        cout<<pcd_files[pc_index]<<endl;
        pc_index++;

        ///聚类分割
        ec.setSearchMethod (tree);//设置点云的搜索机制
        ec.setInputCloud (cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        ec.extract (cluster_indices);//从点云中提取聚类，并将点云索引保存在cluster_indices中

        cout<<"分割簇的数量："<<cluster_indices.size()<<endl;
        for(int i=0;i<cluster_indices.size();++i){
            cout<<i<<" "<<cluster_indices[i].indices.size()<<endl;
        }

        if(cluster_indices.empty()){
            continue;
        }

        ///选择第一个簇作为分割结果
        PointCloud::Ptr segmented_pc(new PointCloud);
        auto &indices = cluster_indices[0].indices;
        segmented_pc->points.reserve(indices.size());
        for(auto &index:indices){
            segmented_pc->points.push_back(cloud->points[index]);
        }
        segmented_pc->width = segmented_pc->points.size();
        segmented_pc->height=1;
        segmented_pc->is_dense = true;


        if(last_cloud){

            icp.setInputSource (last_cloud);//点云A
            icp.setInputTarget (segmented_pc);//点云B
            icp.align(*inliners);

            cout<<"icp,source:"<<last_cloud->points.size()<<" target:"<<segmented_pc->points.size()<<endl;

            if (icp.hasConverged()) {
                printf("\033[11A");  // Go up 11 lines in terminal output.
                printf("\nICP has converged, score is %+.0e\n", icp.getFitnessScore());
                Eigen::Matrix4d T_ba = icp.getFinalTransformation().cast<double>();//得到的是将点从source变换到target的变换矩阵
                print4x4Matrix(T_ba);
            } else {
                PCL_ERROR ("\nICP has not converged.\n");
                continue;
            }

        }

        Pub(*segmented_pc,"/dynamic_vins/instance_point_cloud");

        ros::spinOnce();
        rate.sleep();

        last_cloud=segmented_pc;
    }


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






