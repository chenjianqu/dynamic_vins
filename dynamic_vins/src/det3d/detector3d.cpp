/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "detector3d.h"

#include <iostream>
#include <filesystem>
#include <spdlog/logger.h>

#include "det3d_parameter.h"
#include "utils/utils.h"
#include "utils/log_utils.h"

namespace dynamic_vins{\


Detector3D::Detector3D(const std::string& config_path){
    det3d_para::SetParameters(config_path);
}


std::vector<Box3D::Ptr> Detector3D::ReadBox3dFromTxt(const std::string &txt_path,double score_threshold)
{
    std::vector<Box3D::Ptr> boxes;

    Eigen::Matrix<double,8,3> corners_norm;
    corners_norm << 0,0,0,  0,0,1,  0,1,1,  0,1,0,  1,0,0,  1,0,1,  1,1,1,  1,1,0;
    Eigen::Vector3d offset(0.5,1,0.5);//预测结果所在的坐标系与相机坐标系之间的偏移
    corners_norm = corners_norm.array().rowwise() - offset.transpose().array();//将每个坐标减去偏移量


    std::ifstream fp(txt_path);
    string line;
    int index=0;
    while (getline(fp,line)){ //循环读取每行数据
        vector<string> tokens;
        split(line,tokens," ");

        ///每行的前3个数字是类别,属性,分数
        Box3D::Ptr box = std::make_shared<Box3D>(
                std::stoi(tokens[0]),std::stoi(tokens[1]),std::stod(tokens[2]));
        if(box->score<score_threshold){
            continue;
        }

        ///3-5个数字是物体包围框底部的中心
        box->bottom_center<<std::stod(tokens[3]),std::stod(tokens[4]),std::stod(tokens[5]);
        ///6-8数字是物体在x,y,z轴上的大小
        box->dims<<std::stod(tokens[6]),std::stod(tokens[7]),std::stod(tokens[8]);
        ///9个yaw角(绕着y轴,因为y轴是垂直向下的)
        double yaw=std::stod(tokens[9]);
        box->yaw = yaw;

        Eigen::Matrix<double,3,8> corners = corners_norm.transpose(); //得到矩阵 3x8
        corners = corners.array().colwise() * box->dims.array();//广播逐点乘法

        ///根据yaw角构造旋转矩阵
        Eigen::Matrix3d R;
        R<<cos(yaw),0, -sin(yaw),   0,1,0,   sin(yaw),0,cos(yaw);

        Eigen::Matrix<double,8,3> result =  corners.transpose() * R;//8x3
        //加上偏移量
        Eigen::Matrix<double,8,3> output= result.array().rowwise() + box->bottom_center.transpose().array();
        box->corners = output.transpose();
        ///计算3D box投影到图像平面
        for(int i=0;i<8;++i){
            Vec2d p;
            cam0->ProjectPoint(box->corners.col(i),p);
            box->corners_2d.col(i) = p;
        }
        Vec2d corner2d_min_pt = box->corners_2d.rowwise().minCoeff();
        Vec2d corner2d_max_pt = box->corners_2d.rowwise().maxCoeff();
        box->box2d.min_pt = cv::Point2f(corner2d_min_pt.x(),corner2d_min_pt.y());
        box->box2d.max_pt = cv::Point2f(corner2d_max_pt.x(),corner2d_max_pt.y());
        box->box2d.center_pt = (box->box2d.min_pt + box->box2d.max_pt) / 2;

        ///计算包围框中心坐标
        Eigen::Vector3d center = (output.row(0)+output.row(6)).transpose()  /2;
        box->center = center;

        boxes.push_back(box);

        index++;
    }
    fp.close();

    return boxes;
}




std::vector<Box3D::Ptr> Detector3D::ReadBox3D(unsigned int seq_id){
    ///补零
    int name_width=6;
    std::stringstream ss;
    ss<<std::setw(name_width)<<std::setfill('0')<<seq_id;
    string target_name;
    ss >> target_name;

    ///获取目录中所有的文件名
    static vector<fs::path> names;
    if(names.empty()){
        fs::path dir_path(det3d_para::kDet3dPreprocessPath);
        if(!fs::exists(dir_path))
            return {};
        fs::directory_iterator dir_iter(dir_path);
        for(auto &it : dir_iter)
            names.emplace_back(it.path().filename());
        std::sort(names.begin(),names.end());
    }

    vector<Box3D::Ptr> boxes;
    bool success_read= false;

    ///二分查找
    int low=0,high=names.size()-1;
    while(low<=high){
        int mid=(low+high)/2;
        string name_stem = names[mid].stem().string();
        if(name_stem == target_name){
            string n_path = (det3d_para::kDet3dPreprocessPath/names[mid]).string();
            boxes = ReadBox3dFromTxt(n_path,det3d_para::kDet3dScoreThreshold);
            success_read = true;
            break;
        }
        else if(name_stem > target_name){
            high = mid-1;
        }
        else{
            low = mid+1;
        }
    }

    if(!success_read){
        string msg=fmt::format("Can not find the target name:{} in dir:{}",target_name,det3d_para::kDet3dPreprocessPath);
        Errors(msg);
        std::cerr<<msg<<std::endl;
    }

    return boxes;
}


}
