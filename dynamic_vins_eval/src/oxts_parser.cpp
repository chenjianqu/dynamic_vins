//
// Created by chen on 2022/5/19.
//

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <mutex>
#include <list>
#include <thread>
#include <regex>
#include<filesystem>

#include <Eigen/Dense>

using namespace std;


vector<vector<double>> ReadOxtsData(const string &path){
    vector<vector<double>> data;
    ifstream fp(path); //定义声明一个ifstream对象，指定文件路径
    string line;
    vector<double> tokens;
    while (getline(fp,line)){ //循环读取每行数据
        stringstream text_stream(line);
        string item;
        tokens.clear();
        while (std::getline(text_stream, item, ' ')) {
            tokens.push_back(stod(item));
        }
        data.push_back(tokens);
    }
    fp.close();

    return data;
}

/**
 * converts lat/lon coordinates to mercator coordinates using mercator scale
 * @param lat
 * @param lon
 * @param scale
 * @return
 */
std::pair<double,double> LatLonToMercator(double lat,double lon,double scale){
    constexpr double er = 6378137;
    double mx = scale * lon * M_PI * er /180.;
    double my = scale * er * log(tan((90+lat)*M_PI/360));
    return {mx,my};
}

/**
 % converts a list of oxts measurements into metric poses,
% starting at (0,0,0) meters, OXTS coordinates are defined as
% x = forward, y = right, z = down (see OXTS RT3000 user manual)
% afterwards, pose{i} contains the transformation which takes a
% 3D point in the i'th frame and projects it into the oxts
% coordinates of the first frame.
 */
void ParseOxts(vector<Eigen::Matrix4d> &pose,const string &data_path){

    auto data = ReadOxtsData(data_path);
    if(data.empty()){
        cerr<<"data.empty()"<<endl;
        std::terminate();
    }

    double scale = cos(data[0][0] * M_PI / 180.0);

    Eigen::Matrix4d Tr_0_inv;
    bool is_init=false;

    for(auto &row:data){
        //for(double d:row) cout<<d<<" ";
        //cout<<endl;
        //cout<<row.size()<<endl;
        Eigen::Vector3d t;
        std::tie(t.x(),t.y()) = LatLonToMercator(row[0],row[1],scale);
        t.z() = row[2];

        //cout<<t.transpose()<<endl;

        double rx = row[3];
        double ry = row[4];
        double rz = row[5];

        Eigen::Matrix3d Rx,Ry,Rz;
        Rx << 1, 0, 0,
        0,cos(rx), -sin(rx),
        0, sin(rx), cos(rx); // base => nav  (level oxts => rotated oxts)
        Ry<<cos(ry), 0, sin(ry),
        0, 1, 0,
        -sin(ry), 0, cos(ry); // base => nav  (level oxts => rotated oxts)

        Rz<<cos(rz), -sin(rz), 0,
        sin(rz), cos(rz), 0,
        0, 0, 1; // base => nav  (level oxts => rotated oxts)
        Eigen::Matrix3d R  = Rz*Ry*Rx;

        if(!is_init){
            is_init=true;
            Eigen::Matrix4d T_inv = Eigen::Matrix4d::Identity();
            T_inv.topLeftCorner(3,3) = R;
            T_inv.block<3,1>(0,3) = t;
            cout<<T_inv<<endl;
            Tr_0_inv = T_inv.inverse();
        }

        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        T.topLeftCorner(3,3) = R;
        T.block<3,1>(0,3) = t;
        T = Tr_0_inv * T;
        pose.emplace_back(T);

        // cout<<T<<endl;

    }

}






