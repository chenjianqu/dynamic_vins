#include <cstdio>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <mutex>
#include <list>
#include <thread>
#include <regex>
#include<filesystem>
#include <random>

#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "call_back.h"

using namespace std;

using Vec3d=Eigen::Vector3d;



class Rect2D{
public:
    using Ptr = std::shared_ptr<Rect2D>;
    Rect2D() =default;
    Rect2D(cv::Point2f &min_p, cv::Point2f &max_p): min_pt(min_p), max_pt(max_p){
        center_pt = (min_pt+max_pt)/2;
    }

    cv::Point2f min_pt,max_pt;//边界框的两个点
    cv::Point2f center_pt;
};


class Box3D{
public:
    using Ptr=std::shared_ptr<Box3D>;

    int id;
    ///每行的前3个数字是类别,属性,分数
    int class_id;
    string class_name;
    int attribution_id ;
    double score;

    Vec3d bottom_center{0,0,0};//单目3D目标检测算法预测的包围框底部中心(在相机坐标系下)
    Vec3d dims{0,0,0};//预测的大小
    double yaw{0};//预测的yaw角(沿着垂直向下的z轴)

    Eigen::Matrix<double,3,8> corners;//包围框的8个顶点在相机坐标系下的坐标
    Vec3d center{0,0,0};//包围框中心坐标

    Eigen::Matrix<double,2,8> corners_2d;////包围框的8个顶点在图像坐标系下的像素坐标
    Rect2D box2d;

    cv::Scalar color;
};

std::default_random_engine randomEngine;
std::uniform_int_distribution<unsigned int> color_rd{0,255};


void DrawText(cv::Mat &img, const std::string &str, const cv::Scalar &color, const cv::Point& pos,
              float scale, int thickness, bool reverse) {
    auto t_size = cv::getTextSize(str, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, nullptr);
    cv::Point bottom_left, upper_right;
    if (reverse) {
        upper_right = pos;
        bottom_left = cv::Point(upper_right.x - t_size.width, upper_right.y + t_size.height);
    } else {
        bottom_left = pos;
        upper_right = cv::Point(bottom_left.x + t_size.width, bottom_left.y - t_size.height);
    }

    cv::rectangle(img, bottom_left, upper_right, color, -1);
    cv::putText(img, str, bottom_left, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255),thickness);
}



inline void split(const std::string& source, std::vector<std::string>& tokens, const std::string& delimiters = " ") {
    std::regex re(delimiters);
    std::copy(std::sregex_token_iterator(source.begin(), source.end(),re,-1),
              std::sregex_token_iterator(),
              std::back_inserter(tokens));
}

std::unordered_map<int,std::vector<Box3D::Ptr>> ReadBox3dFromTxt(const std::string &txt_path,double score_threshold)
{
    std::unordered_map<int,std::vector<Box3D::Ptr>> all_box;

    static std::unordered_map<int,cv::Scalar> color_map;

    std::ifstream fp(txt_path);
    cout<<txt_path<<endl;
    string line;
    int index=0;
    while (getline(fp,line)){ //循环读取每行数据
        vector<string> tokens;
        split(line,tokens," ");
        cout<<line<<endl;

        int frame=std::stoi(tokens[0]);

        Box3D::Ptr box = std::make_shared<Box3D>();
        box->id = std::stoi(tokens[1]);
        box->class_name = tokens[2];
        box->box2d.min_pt.x = std::stof(tokens[6]);
        box->box2d.min_pt.y = std::stof(tokens[7]);
        box->box2d.max_pt.x = std::stof(tokens[8]);
        box->box2d.max_pt.y = std::stof(tokens[9]);
        box->box2d.center_pt=(box->box2d.min_pt+box->box2d.max_pt)/2.;
        box->dims.x() = std::stod(tokens[10]);
        box->dims.y() = std::stod(tokens[11]);
        box->dims.z() = std::stod(tokens[12]);
        box->bottom_center.x() = std::stod(tokens[13]);
        box->bottom_center.y() = std::stod(tokens[14]);
        box->bottom_center.z() = std::stod(tokens[15]);
        box->yaw = std::stod(tokens[16]);
        //box->score = std::stod(tokens[17]);

        if(color_map.find(box->id)==color_map.end()){
            color_map.insert(
                    {box->id,
                     cv::Scalar(color_rd(randomEngine),color_rd(randomEngine),color_rd(randomEngine))
                    });
        }
        box->color = color_map[box->id];

        if(all_box.find(frame)==all_box.end()){
            all_box.insert({frame,std::vector<Box3D::Ptr>()});
        }
        all_box[frame].push_back(box);
    }
    fp.close();

    return all_box;
}




int main(int argc, char** argv)
{
    if(argc!=3){
        cerr<<"usage:rosrun dynamic_vins_eval visualize_box ${tracking_result} ${image_dir}"<<endl;
        return -1;
    }

    setlocale(LC_ALL, "");//防止中文乱码
    ros::init(argc, argv, "visualize_3d_detection");
    ros::start();

    ros::NodeHandle nh;

    string data_path=argv[1];
    string image_path=argv[2];

    auto boxes = ReadBox3dFromTxt(data_path,0.0);

    Dataloader dataloader(image_path);

    while(ros::ok()){
        SemanticImage img = dataloader.LoadStereo();
        if(img.color0.empty()){
            break;
        }
        auto it=boxes.find(img.seq);
        if(it!=boxes.end()){
            for(auto &box : it->second){
                 cv::rectangle(img.color0,box->box2d.min_pt,box->box2d.max_pt,box->color,2);
                 cv::putText(img.color0,std::to_string(box->id),box->box2d.center_pt,cv::FONT_HERSHEY_PLAIN,
                             1.5,box->color,2);
            }
        }

        DrawText(img.color0, std::to_string(img.seq), cv::Scalar(255,0,0), cv::Point2f(10,50),
                1.5, 2, false);
        dataloader.ShowImage(img.color0,100);
    }


    return 0;
}

