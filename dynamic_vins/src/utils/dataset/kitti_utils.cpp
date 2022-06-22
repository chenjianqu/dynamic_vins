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





}



