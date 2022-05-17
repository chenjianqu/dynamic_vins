/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "kitti_utils.h"
#include <regex>
#include "utils/def.h"
#include "utils/utils.h"

namespace dynamic_vins::kitti{ \


/**
 * 从kitti tracking数据集的参数文件中读取相关的内参和外参
 * @param path
 * @return
 */
std::map<string,Eigen::MatrixXd> ReadCalibFile(const string &path){
    std::map<string,Eigen::MatrixXd> calib_map;

    std::ifstream fp(path);
    if(!fp.is_open()){
        cerr <<"ReadCalibFile() failed. Can not open calib file:"<<path<<endl;
        std::terminate();
    }

    string line;
    while (getline(fp,line)){ //循环读取每行数据
        vector<string> tokens;
        split(line,tokens," ");
        if(tokens[0]=="P0:"){
            vector<double> data(12);
            for(int i=0;i<12;++i)
                data[i] = std::stod(tokens[i+1]);
            calib_map["P0"] = Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>>(data.data(), 3, 4);
        }
        else if(tokens[0]=="P1:"){
            vector<double> data(12);
            for(int i=0;i<12;++i)
                data[i] = std::stod(tokens[i+1]);
            calib_map["P1"] = Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>>(data.data(), 3, 4);
        }
        else if(tokens[0]=="P2:"){
            vector<double> data(12);
            for(int i=0;i<12;++i)
                data[i] = std::stod(tokens[i+1]);
            calib_map["P2"] = Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>>(data.data(), 3, 4);
        }
        else if(tokens[0]=="P3:"){
            vector<double> data(12);
            for(int i=0;i<12;++i)
                data[i] = std::stod(tokens[i+1]);
            calib_map["P3"] = Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>>(data.data(), 3, 4);
        }
        else if(tokens[0]=="R_rect"){
            vector<double> data(9);
            for(int i=0;i<9;++i)
                data[i] = std::stod(tokens[i+1]);
            calib_map["R_rect"] = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(data.data(), 3, 3);
        }
        else if(tokens[0]=="Tr_velo_cam"){
            vector<double> data(12);
            for(int i=0;i<12;++i)
                data[i] = std::stod(tokens[i+1]);
            calib_map["Tr_velo_cam"] = Mat4d::Identity();
            calib_map["Tr_velo_cam"].topLeftCorner(3,4) = Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>>(data.data(), 3, 4);
        }
        else if(tokens[0]=="Tr_imu_velo"){
            vector<double> data(12);
            for(int i=0;i<12;++i)
                data[i] = std::stod(tokens[i+1]);
            calib_map["Tr_imu_velo"] = Mat4d::Identity();
            calib_map["Tr_imu_velo"].topLeftCorner(3,4) = Eigen::Map<Eigen::Matrix<double,3,4,Eigen::RowMajor>>(data.data(), 3, 4);
        }
    }
    fp.close();

    return calib_map;
}


/**
 * 保存到文件中每行的内容
 * 1    frame        Frame within the sequence where the object appearers
   1    track id     Unique tracking id of this object within this sequence
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Integer (0,1,2) indicating the level of truncation.
                     Note that this is in contrast to the object detection
                     benchmark where truncation is a float in [0,1].
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

alpha和rotation_y的区别：
 The coordinates in the camera coordinate system can be projected in the image
by using the 3x4 projection matrix in the calib folder, where for the left
color camera for which the images are provided, P2 must be used. The
difference between rotation_y and alpha is, that rotation_y is directly
given in camera coordinates, while alpha also considers the vector from the
camera center to the object center, to compute the relative orientation of
the object with respect to the camera. For example, a car which is facing
along the X-axis of the camera coordinate system corresponds to rotation_y=0,
no matter where it is located in the X/Z plane (bird's eye view), while
alpha is zero only, when this object is located along the Z-axis of the
camera. When moving the car away from the Z-axis, the observation angle
(\alpha) will change.
 */


void SaveInstanceTrajectory(unsigned int frame_id,unsigned int track_id,std::string &type,
                            int truncated,int occluded,double alpha,Vec4d &box,
                            Vec3d &dims,Vec3d &location,double rotation_y,double score){
    string save_path = cfg::kOutputFolder + "kitti_tracking/"+kDatasetSequence+".txt";

    //追加写入
    ofstream fout(save_path,std::ios::out | std::ios::app);

    fout<<frame_id<<" "<<track_id<<" "<<type<<" "<<truncated<<" "<<occluded<<" ";
    fout<<alpha<<" "<<VecToStr(bbox)<<" "<<VecToStr(dims)<<" "<<VecToStr(location)<<
        " "<<rotation_y<<" "<<score;
    fout<<endl;
    fout.close();
}


void ClearTrajectoryFile(){
    string save_path = cfg::kOutputFolder + "kitti_tracking/"+kDatasetSequence+".txt";
    Warnv("ClearTrajectoryFile | Clear File:{}",save_path);
    ofstream fout(save_path,std::ios::out);
    fout.close();
    return;
}



}



