//
// Created by chen on 2022/4/25.
//

#include "kitti_utils.h"

#include <regex>

namespace dynamic_vins::kitti{ \



void split(const std::string& source, std::vector<std::string>& tokens, const string& delimiters = " ") {
    std::regex re(delimiters);
    std::copy(std::sregex_token_iterator(source.begin(), source.end(),re,-1),
              std::sregex_token_iterator(),
              std::back_inserter(tokens));
}

std::map<string,Eigen::MatrixXd> ReadCalibFile(const string &path){
    std::map<string,Eigen::MatrixXd> calib_map;

    std::ifstream fp(path);
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



