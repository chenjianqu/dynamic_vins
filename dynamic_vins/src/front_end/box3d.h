//
// Created by chen on 2022/5/2.
//

#ifndef DYNAMIC_VINS_BOX3D_H
#define DYNAMIC_VINS_BOX3D_H

#include <vector>
#include <string>
#include <eigen3/Eigen/Core>

namespace dynamic_vins{\


class Box3D{
public:
    ///每行的前3个数字是类别,属性,分数
    int class_id;
    int attribution_id ;
    double score;

    Eigen::Vector3d bottom_center;//单目3D目标检测算法预测的包围框底部中心(在相机坐标系下)
    Eigen::Vector3d dims;//预测的大小
    double yaw;//预测的yaw角(沿着垂直向下的z轴)

    Eigen::Matrix<double,8,3> corners;//包围框的8个顶点在相机坐标系下的坐标
    Eigen::Vector3d center;//包围框中心坐标

    static std::vector<Box3D> ReadBox3dFromTxt(const std::string &txt_path,const double score_threshold);
};

std::vector<Box3D> ReadBox3D(unsigned int seq_id);


}

#endif //DYNAMIC_VINS_BOX3D_H
