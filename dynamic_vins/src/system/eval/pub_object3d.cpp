/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <iostream>

#include <visualization_msgs/MarkerArray.h>

#include "utils/utils.h"
#include "utils/box3d.h"
#include "utils/camera_model.h"
#include "utils/parameters.h"
#include "utils/io/io_parameters.h"
#include "utils/io/io_utils.h"
#include "utils/io/visualization.h"
#include "utils/io/build_markers.h"
#include "det3d/det3d_parameter.h"

using namespace std;
using namespace visualization_msgs;


namespace dynamic_vins{\

/**
 * 可视化3D目标检测的结果
 * @param argc
 * @param argv
 * @return
 */
int PubObject3D(int argc, char **argv)
{
    const string object3d_root_path="/home/chen/ws/dynamic_ws/src/dynamic_vins/data/ground_truth/kitti_tracking_object/0003/";

    auto names = GetDirectoryFileNames(object3d_root_path);

    for(auto &name:names){
        string file_path = object3d_root_path+name.string();
        std::ifstream fp(file_path);
        string line;
        int index=0;
        while (getline(fp,line)){ //循环读取每行数据
            vector<string> tokens;
            split(line,tokens," ");
            if(tokens[0]=="DontCare"){
                continue;
            }

            Box3D::Ptr box = std::make_shared<Box3D>(tokens);


            index++;
        }
        fp.close();

    }

}



int PubFCOS3D(int argc, char **argv,ros::NodeHandle &nh)
{
    ros::Publisher pub_instance_marker=nh.advertise<MarkerArray>("fcos3d_markers", 1000);

    const string object3d_root_path="/home/chen/datasets/kitti/tracking/det3d_02/training/image_02/0003/";
    const string img_root_path="/home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/0003/";

    int index=0;

    while(ros::ok()){
        string name = PadNumber(index,6);
        index++;
        string img_path = img_root_path+name+".png";
        cv::Mat img = cv::imread(img_path,-1);
        cout<<img_path<<endl;
        if(img.empty()){
            break;
        }

        vector<Box3D::Ptr> boxes;

        string file_path = object3d_root_path+name+".txt";
        std::ifstream fp(file_path);
        if(!fp.is_open()){
            break;
        }
        string line;
        while (getline(fp,line)){ //循环读取每行数据
            vector<string> tokens;
            split(line,tokens," ");
            if(std::stod(tokens[2]) < det3d_para::kDet3dScoreThreshold)
                continue;
            Box3D::Ptr box = std::make_shared<Box3D>(tokens);
            boxes.push_back(box);
        }
        fp.close();

        MarkerArray marker_array;

        int cnt=0;
        for(auto &box : boxes){
            cnt++;
            box->VisCorners2d(img,cv::Scalar(255,255,255),*cam0);
            auto cube_marker = BuildCubeMarker(box->corners,cnt,GenerateNormBgrColor("blue"));
            marker_array.markers.push_back(cube_marker);

            Mat34d axis_matrix = box->GetCoordinateVectorInCamera(4);
            auto axis_markers = BuildAxisMarker(axis_matrix,cnt);
            marker_array.markers.push_back(std::get<0>(axis_markers));
            marker_array.markers.push_back(std::get<1>(axis_markers));
            marker_array.markers.push_back(std::get<2>(axis_markers));
        }

        pub_instance_marker.publish(marker_array);

        ///可视化
        bool pause=false;
        do{
            cv::imshow("PubFCOS3D",img);
            int key = cv::waitKey(100);
            if(key ==' '){
                pause = !pause;
            }
            else if(key== 27){ //ESC
                cfg::ok=false;
                ros::shutdown();
                pause=false;
            }
            else if(key == 'r' || key == 'R'){
                pause=false;
            }
        } while (pause);

    }

    cout<<"exit index:"<<index<<endl;


    return 0;

}



void InitGlobalParameters(const string &file_name){
    cfg cfg(file_name);
    ///初始化logger
    MyLogger::InitLogger(file_name);
    ///初始化相机模型
    InitCamera(file_name);
    ///初始化局部参数
    io_para::SetParameters(file_name);

    det3d_para::SetParameters(file_name);

}


}

int main(int argc, char **argv)
{
    if(argc != 2){
        std::cerr<<"please input: rosrun vins vins_node [cfg file]"<< std::endl;
        return 1;
    }

    ros::init(argc, argv, "pub_object3d");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);


    dynamic_vins::InitGlobalParameters(argv[1]);

    std::cout<<"InitGlobalParameters finished"<<std::endl;

    //return dynamic_vins::PubObject3D(argc,argv);
    return dynamic_vins::PubFCOS3D(argc, argv,n);
}



