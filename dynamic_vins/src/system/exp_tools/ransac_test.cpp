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

inline geometry_msgs::Point EigenToGeometryPoint(const Vec3d &vec_point){
    geometry_msgs::Point point;
    point.x = vec_point.x();
    point.y = vec_point.y();
    point.z = vec_point.z();
    return point;
}

inline std_msgs::ColorRGBA ScalarBgrToColorRGBA(const cv::Scalar &color){
    std_msgs::ColorRGBA color_rgba;
    color_rgba.r = color[2];
    color_rgba.g = color[1];
    color_rgba.b = color[0];
    color_rgba.a = color[3];
    if(color_rgba.a==0){
        color_rgba.a=1.;
    }
    return color_rgba;
}

Marker CubeMarker(Mat38d &corners, unsigned int id, const cv::Scalar &color, double scale,
                  Marker::_action_type action,const string &ns ,int offset){
    Marker msg;

    msg.header.frame_id="world";
    msg.header.stamp=ros::Time::now();
    msg.ns=ns;
    msg.action=action;

    const int kMarkerTypeNumber=1;

    msg.id=id * kMarkerTypeNumber + offset;
    msg.type=Marker::LINE_STRIP;//marker的类型
    if(action==Marker::DELETE){
        return msg;
    }

    msg.lifetime=ros::Duration();//持续时间3s，若为ros::Duration()表示一直持续

    msg.pose.orientation.w=1.0;

    msg.scale.x=scale;//线宽
    msg.color = ScalarBgrToColorRGBA(color);

    //设置立方体的八个顶点
    geometry_msgs::Point p[8];
    p[0] = EigenToGeometryPoint(corners.col(0));
    p[1] = EigenToGeometryPoint(corners.col(1));
    p[2] = EigenToGeometryPoint(corners.col(2));
    p[3] = EigenToGeometryPoint(corners.col(3));
    p[4] = EigenToGeometryPoint(corners.col(4));
    p[5] = EigenToGeometryPoint(corners.col(5));
    p[6] = EigenToGeometryPoint(corners.col(6));
    p[7] = EigenToGeometryPoint(corners.col(7));

    /**
             .. code-block:: none

                             front z
                                    /
                                   /
                   p1(x0, y0, z1) + -----------  + p5(x1, y0, z1)
                                 /|            / |
                                / |           /  |
                p0(x0, y0, z0) + ----------- +   + p6(x1, y1, z1)
                               |  /      .   |  /
                               | / origin    | /
                p3(x0, y1, z0) + ----------- + -------> x right
                               |             p7(x1, y1, z0)
                               |
                               v
                        down y
     输入的点序列:p0:0,0,0, p1: 0,0,1,  p2: 0,1,1,  p3: 0,1,0,  p4: 1,0,0,  p5: 1,0,1,  p6: 1,1,1,  p7: 1,1,0;

     */
    msg.points.push_back(p[0]);
    msg.points.push_back(p[1]);
    msg.points.push_back(p[5]);
    msg.points.push_back(p[4]);
    msg.points.push_back(p[0]);
    msg.points.push_back(p[3]);
    msg.points.push_back(p[7]);
    msg.points.push_back(p[4]);
    msg.points.push_back(p[7]);
    msg.points.push_back(p[6]);
    msg.points.push_back(p[5]);
    msg.points.push_back(p[6]);
    msg.points.push_back(p[2]);
    msg.points.push_back(p[1]);
    msg.points.push_back(p[2]);
    msg.points.push_back(p[3]);

    return msg;
}


void PubCube(const Vec3d &center,const Vec3d &dims){
    /**
         .. code-block:: none

                         front z
                                /
                               /
               p1(x0, y0, z1) + -----------  + p5(x1, y0, z1)
                             /|            / |
                            / |           /  |
            p0(x0, y0, z0) + ----------- +   + p6(x1, y1, z1)
                           |  /      .   |  /
                           | / origin    | /
            p3(x0, y1, z0) + ----------- + -------> x right
                           |             p7(x1, y1, z0)
                           |
                           v
                    down y
 输入的点序列:p0:0,0,0, p1: 0,0,1,  p2: 0,1,1,  p3: 0,1,0,  p4: 1,0,0,  p5: 1,0,1,  p6: 1,1,1,  p7: 1,1,0;
    */
    Vec3d d = dims/2.;
    Mat38d corners;
    corners.col(0) = Vec3d(-d.x(),-d.y(),-d.z()) + center;
    corners.col(1) = Vec3d(-d.x(),-d.y(),d.z()) + center;
    corners.col(2) = Vec3d(-d.x(),d.y(),d.z())+ center;
    corners.col(3) = Vec3d(-d.x(),d.y(),-d.z())+ center;
    corners.col(4) = Vec3d(d.x(),-d.y(),-d.z())+ center;
    corners.col(5) = Vec3d(d.x(),-d.y(),d.z())+ center;
    corners.col(6) = Vec3d(d.x(),d.y(),d.z())+ center;
    corners.col(7) = Vec3d(d.x(),d.y(),-d.z())+ center;

    Marker::_action_type action=Marker::ADD;
    Marker marker = CubeMarker(corners, 1, cv::Scalar(0,0,1.), 0.1, action,"cube_estimation",7);

    MarkerArray markers;
    markers.markers.push_back(marker);

    pub_instance_marker->publish(markers);
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



/**
* 使用RANSAC拟合包围框,根据距离的远近删除点
* @param points 相机坐标系下的3D点
* @param dims 包围框的长度
* @return
*/
std::optional<Vec3d> FitBox3DFromCameraFrame(vector<Vec3d> &points,const Vec3d& dims)
{
    if(points.empty()){
        return {};
    }
    std::list<Vec3d> points_rest;
    Vec3d center_pt = Vec3d::Zero();
    ///计算初始值
    for(auto &p:points){
        center_pt+=p;
        points_rest.push_back(p);
    }
    center_pt /= (double)points.size();
    bool is_find=false;

    double dims_norm = (dims/2.).norm();

    string log_text = "FitBox3DFromCameraFrame: \n";

    ///最多迭代10次
    for(int iter=0;iter<10;++iter){
        //计算每个点到中心的距离
        vector<tuple<double,Vec3d>> points_with_dist;
        for(auto &p : points_rest){
            points_with_dist.emplace_back((p-center_pt).norm(),p);
        }
        //根据距离排序,升序
        std::sort(points_with_dist.begin(),points_with_dist.end(),[](tuple<double,Vec3d> &a,tuple<double,Vec3d> &b){
            return std::get<0>(a) < std::get<0>(b);
        });

        //选择前80%的点重新计算中心
        center_pt.setZero();
        double len = points_with_dist.size();
        int len_used=len*0.8;
        for(int i=0;i<len_used;++i){
            center_pt += std::get<1>(points_with_dist[i]);
        }
        if(len_used<2){
            break;
        }
        center_pt /= len_used;

        log_text += fmt::format("iter:{} center_pt:{}\n",iter, VecToStr(center_pt));

        //如前80%的点位于包围框内,则退出
        if(std::get<0>(points_with_dist[len_used]) <= dims_norm){
            is_find=true;
            break;
        }

        ///只将距离相机中心最近的前50%点用于下一轮的计算
        vector<tuple<double,Vec3d>> points_cam_dist;
        for(auto &p : points_rest){
            points_cam_dist.emplace_back(p.norm(),p);
        }
        //根据距离排序,升序
        std::sort(points_cam_dist.begin(),points_cam_dist.end(),[](tuple<double,Vec3d> &a,tuple<double,Vec3d> &b){
            return std::get<0>(a) < std::get<0>(b);
        });
        points_rest.clear();
        for(int i=0;i<len*0.5;++i){
            points_rest.push_back(std::get<1>(points_cam_dist[i]));
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr = EigenToPCL(points);
        Pub(*point_cloud_ptr,"/dynamic_vins/instance_point_cloud");
        PubCube(center_pt,dims);
        std::this_thread::sleep_for(1000ms);

        cout<<center_pt.transpose()<<endl;


    }


    if(is_find){
        return center_pt;
    }
    else{
        return {};
    }
}





/**
* 使用RANSAC拟合包围框,根据距离的远近删除点
* @param points 相机坐标系下的3D点
* @param dims 包围框的长度
* @return
*/
std::optional<Vec3d> FitBox3DWithRANSAC(vector<Vec3d> &points,const Vec3d& dims)
{
    if(points.empty()){
        return {};
    }
    int size = points.size();

    vector<Vec3d> points_rest(size);
    Vec3d best_center = Vec3d::Zero();
    int best_inlines = 10;
    ///计算初始值
    for(int i=0;i<points.size();++i){
        best_center+= points[i];
        points_rest[i] = points[i].cwiseAbs();
    }
    best_center /= (double)size;

    Vec3d box = dims/2;

    string log_text = "FitBox3DWithRANSAC: \n";

    std::random_device rd;
    vector<int> random_indices(size);
    std::iota(random_indices.begin(),random_indices.end(),0);

    int batch_size = std::min(10,size);

    for(int iter=0;iter<20;++iter){
        ///选择其中的10个点计算中心
        std::shuffle(random_indices.begin(),random_indices.end(),rd);
        Vec3d center=Vec3d::Zero();
        for(int i=0;i<batch_size;++i){
            center += points_rest[random_indices[i]];
        }
        center /= (double)batch_size;

        int inliers=0;

        ///判断落在边界框内的点的数量
        for(int i=0;i<size;++i){
            if(points_rest[i].x() <= box.x() ||
            points_rest[i].y() <= box.y() ||
            points_rest[i].z() <= box.z()){
                inliers++;
            }
        }

        if(inliers > best_inlines){
            best_inlines = inliers;
            best_center = center;
        }

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr = EigenToPCL(points);
        Pub(*point_cloud_ptr,"/dynamic_vins/instance_point_cloud");
        PubCube(center,dims);
        std::this_thread::sleep_for(1000ms);

        cout<<center.transpose()<<endl;
    }


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr = EigenToPCL(points);
    Pub(*point_cloud_ptr,"/dynamic_vins/instance_point_cloud");
    PubCube(best_center,dims);

    return best_center;
}




int Run(int argc, char **argv){
    ros::init(argc, argv, "ransac_test");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    pub_instance_marker=std::make_shared<ros::Publisher>();
    *pub_instance_marker=n.advertise<MarkerArray>("/dynamic_vins/instance_marker", 1000);

    nh = &n;

    int points_size = 100;
    std::default_random_engine engine;
    std::normal_distribution<double> distribution(0,1);//均值为0,方差为1

    vector<Vec3d> points(points_size);
    for(int i=0;i<points_size;++i){
        Vec3d noise;
        if(i%4==0){
            noise<<distribution(engine),distribution(engine)+5,distribution(engine)/2.;
        }else{
            noise<<distribution(engine),distribution(engine),distribution(engine)/4.;
        }
        points[i]= target + noise;
        //cout<<points[i].x()<<" "<<points[i].y()<<" "<<points[i].z()<<endl;
    }


    std::thread t(&FitBox3DWithRANSAC,std::ref(points),Vec3d(2,2,1));

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






