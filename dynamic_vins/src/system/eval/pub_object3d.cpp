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

class PubDemo{
public:

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

                Box3D::Ptr box = Box3D::Box3dFromFCOS3D(tokens);


                index++;
            }
            fp.close();

        }

    }




    //const string object3d_root_path="/home/chen/datasets/VIODE/det3d_cam0/day_03_high/";
    //const string img_root_path="/home/chen/datasets/VIODE/cam0/day_03_high/";


    vector<Box3D::Ptr> ReadPredictBox(const string &name){
        string file_path = object3d_root_path+name+".txt";
        std::ifstream fp(file_path);
        if(!fp.is_open()){
            {};
        }

        vector<Box3D::Ptr> boxes;
        string line;
        while (getline(fp,line)){ //循环读取每行数据
            vector<string> tokens;
            split(line,tokens," ");
            if(std::stod(tokens[2]) < det3d_para::kDet3dScoreThreshold)
                continue;
            Box3D::Ptr box = Box3D::Box3dFromFCOS3D(tokens);
            boxes.push_back(box);
        }
        fp.close();

        return boxes;
    }

    int PubViodeFCOS3D(int argc, char **argv,ros::NodeHandle &nh)
    {
        ros::Publisher pub_instance_marker=nh.advertise<MarkerArray>("fcos3d_markers", 1000);

        ///获取目录中所有的文件名
        vector<fs::path> names = GetDirectoryFileNames(object3d_root_path);
        int index=0;

        while(ros::ok()){
            if(index>=names.size()){
                break;
            }

            string file_path = object3d_root_path + names[index].string();
            cout<<file_path<<endl;
            string time_str = names[index].stem().string();

            std::ifstream fp(file_path);
            if(!fp.is_open()){
                {};
            }

            vector<Box3D::Ptr> boxes;
            string line;
            while (getline(fp,line)){ //循环读取每行数据
                vector<string> tokens;
                split(line,tokens," ");
                //if(std::stod(tokens[2]) < det3d_para::kDet3dScoreThreshold)
                //    continue;
                Box3D::Ptr box = Box3D::Box3dFromFCOS3D(tokens);
                boxes.push_back(box);
                //cout<<line<<endl;
            }
            fp.close();

            cout<<boxes.size()<<endl;

            string img_path = img_root_path + time_str +".png";
            cv::Mat img = cv::imread(img_path,-1);




            MarkerArray marker_array;


            ///可视化预测的box
            int cnt=0;
            for(auto &box : boxes){
                cnt++;
                box->VisCorners2d(img,cv::Scalar(255,255,255),*cam0);
                auto cube_marker = CubeMarker(box->corners,cnt,BgrColor("blue"));
                marker_array.markers.push_back(cube_marker);

                Mat34d axis_matrix = box->GetCoordinateVectorInCamera(4);
                auto axis_markers = AxisMarker(axis_matrix,cnt);
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

            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            index++;
        }

        return 0;
    }


    vector<Box3D::Ptr> ReadPredictBox(int frame){
        string name = PadNumber(frame,6);
        string file_path = object3d_root_path+name+".txt";
        std::ifstream fp(file_path);
        if(!fp.is_open()){
            {};
        }

        vector<Box3D::Ptr> boxes;

        string line;
        while (getline(fp,line)){ //循环读取每行数据
            vector<string> tokens;
            split(line,tokens," ");
            if(std::stod(tokens[2]) < det3d_para::kDet3dScoreThreshold)
                continue;
            Box3D::Ptr box = Box3D::Box3dFromFCOS3D(tokens);
            boxes.push_back(box);
        }
        fp.close();

        return boxes;
    }


    vector<Box3D::Ptr> ReadGroundtruthBox(int frame){
        static unordered_map<int,vector<Box3D::Ptr>> boxes_gt;
        static bool is_first_run=true;
        if(is_first_run){
            is_first_run=false;
            std::ifstream fp_gt(tracking_gt_path);
            if(!fp_gt.is_open()){
                {};
            }

            string line_gt;
            while (getline(fp_gt,line_gt)){ //循环读取每行数据
                vector<string> tokens;
                split(line_gt,tokens," ");
                Box3D::Ptr box = Box3D::Box3dFromKittiTracking(tokens);
                int curr_frame = std::stoi(tokens[0]);
                boxes_gt[curr_frame].push_back(box);
            }
            fp_gt.close();
        }

        if(boxes_gt.count(frame)==0){
            return {};
        }
        else{
            return boxes_gt[frame];
        }
    }


    cv::Mat ReadImage(int frame) const{
        string name = PadNumber(frame,6);
        string img_path = img_root_path+name+".png";
        cout<<img_path<<endl;
        return cv::imread(img_path,-1);
    }


    int PubFCOS3D(int argc, char **argv,ros::NodeHandle &nh)
    {
        ros::Publisher pub_instance_marker=nh.advertise<MarkerArray>("fcos3d_markers", 1000);

        int index=0;

        while(ros::ok()){
            cv::Mat img= ReadImage(index);
            if(img.empty()){
                break;
            }

            MarkerArray marker_array;


            ///可视化预测的box

            int cnt=0;
            /*
            vector<Box3D::Ptr> boxes = ReadPredictBox(index);
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
            }*/

            ///可视化gt框
            vector<Box3D::Ptr> boxes_gt = ReadGroundtruthBox(index);
            for(auto &box : boxes_gt){
                cnt++;
                auto cube_marker = CubeMarker(box->corners, cnt, BgrColor("red"));
                marker_array.markers.push_back(cube_marker);

                auto textMarker = TextMarker(box->bottom_center, cnt, to_string(box->id), BgrColor("blue"), 1.2);
                marker_array.markers.push_back(textMarker);
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

            index++;
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


        object3d_root_path="/home/chen/datasets/kitti/tracking/det3d_02/training/image_02/"+cfg::kDatasetSequence+"/";
        tracking_gt_path="/home/chen/datasets/kitti/tracking/data_tracking_label_2/training/label_02/"+cfg::kDatasetSequence+".txt";
        img_root_path="/home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/"+cfg::kDatasetSequence+"/";

    }

    string object3d_root_path;
    string tracking_gt_path;
    string img_root_path;
};


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

    dynamic_vins::PubDemo pub_demo;

    pub_demo.InitGlobalParameters(argv[1]);

    std::cout<<"InitGlobalParameters finished"<<std::endl;

    //return dynamic_vins::PubObject3D(argc,argv);
    return pub_demo.PubFCOS3D(argc, argv,n);
    //return dynamic_vins::PubViodeFCOS3D(argc, argv,n);
}



