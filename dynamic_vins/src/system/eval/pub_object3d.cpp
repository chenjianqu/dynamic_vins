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

#include "utils/box3d.h"
#include "utils/camera_model.h"
#include "utils/parameters.h"
#include "utils/io/io_parameters.h"
#include "utils/io_utils.h"
#include "utils/io/visualization.h"
#include "utils/io/build_markers.h"
#include "utils/io/dataloader.h"
#include "det3d/det3d_parameter.h"

using namespace std;
using namespace visualization_msgs;


namespace dynamic_vins{\


std::unordered_map<string,vector<string>> ReadCameraPose(const string &pose_file)
{
    std::ifstream fp(pose_file);
    if(!fp.is_open()){
        cerr<<"Can not open:"<<pose_file<<endl;
        return {};
    }
    cout<<"pose_file:"<<pose_file<<endl;

    std::unordered_map<string,vector<string>> cam_pose;

    string line;
    while (getline(fp,line)){ //循环读取每行数据
        vector<string> tokens;
        split(line,tokens," ");
        //cout<<line<<endl;
        double time=std::stod(tokens[0]);
        cam_pose.insert({std::to_string(time),tokens});
    }

    return cam_pose;
}



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

                Box3D::Ptr box = Box3D::Box3dFromFCOS3D(tokens,cam_s.cam0);


                index++;
            }
            fp.close();

        }

    }


    /**
     * 获取对应时刻的位姿
     * @param frame
     * @return (是否成功，旋转矩阵，平移向量)
     */
    [[nodiscard]] std::tuple<bool,Eigen::Matrix3d,Eigen::Vector3d> GetBodyPose(int frame) const{
        double time = frame*0.05;

        static std::unordered_map<string,vector<string>> cam_pose;

        static bool is_first=true;
        if(is_first){
            is_first=false;
            cam_pose = ReadCameraPose(camera_pose_file);
        }

        ///获得位姿Two
        auto it_find = cam_pose.find(std::to_string(time));
        if(it_find == cam_pose.end()){
            cerr<<"Can not find cam pose at time:"<<std::to_string(time)<<endl;
            return {false,Eigen::Matrix3d::Identity(),Eigen::Vector3d::Zero()};
        }
        auto &tokens_cam = it_find->second;
        Eigen::Vector3d t_wc(std::stod(tokens_cam[1]),std::stod(tokens_cam[2]),std::stod(tokens_cam[3]));
        Eigen::Quaterniond q_wc(std::stod(tokens_cam[7]),
                                std::stod(tokens_cam[4]),
                                std::stod(tokens_cam[5]),
                                std::stod(tokens_cam[6]));
        q_wc.normalize();
        Eigen::Matrix3d R_wc = q_wc.toRotationMatrix();

        return {true,R_wc,t_wc};
    }


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
            if(std::stod(tokens[1]) < det3d_para::kDet3dScoreThreshold)
                continue;
            Box3D::Ptr box = Box3D::Box3dFromFCOS3D(tokens,cam_s.cam0);
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
                //if(std::stod(tokens[1]) < det3d_para::kDet3dScoreThreshold)
                //    continue;
                Box3D::Ptr box = Box3D::Box3dFromFCOS3D(tokens,cam_s.cam0);
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
                box->VisCorners2d(img,cv::Scalar(255,255,255),cam_s.cam0);
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


    [[nodiscard]] vector<Box3D::Ptr> ReadPredictBox(int frame) const{
        string name = PadNumber(frame,6);
        string file_path = object3d_root_path+name+".txt";
        cout<<"predict boxes path:"<<file_path<<endl;
        std::ifstream fp(file_path);
        if(!fp.is_open()){
            {};
        }

        vector<Box3D::Ptr> boxes;

        string line;
        while (getline(fp,line)){ //循环读取每行数据
            vector<string> tokens;
            split(line,tokens," ");
            if(std::stod(tokens[1]) < det3d_para::kDet3dScoreThreshold)
                continue;
            Box3D::Ptr box = Box3D::Box3dFromFCOS3D(tokens,cam_s.cam0);
            boxes.push_back(box);
        }
        fp.close();

        return boxes;
    }


    [[nodiscard]] vector<Box3D::Ptr> ReadGroundtruthBox(int frame) const{
        static unordered_map<int,vector<Box3D::Ptr>> boxes_gt;
        static bool is_first_run=true;
        if(is_first_run){
            is_first_run=false;
            std::ifstream fp_gt(tracking_gt_path);
            cout<<"tracking_gt_path:"<<tracking_gt_path<<endl;
            if(!fp_gt.is_open()){
                return {};
            }

            string line_gt;
            while (getline(fp_gt,line_gt)){ //循环读取每行数据
                vector<string> tokens;
                split(line_gt,tokens," ");
                Box3D::Ptr box = Box3D::Box3dFromKittiTracking(tokens,cam_s.cam0);
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


    [[nodiscard]] vector<Box3D::Ptr> ReadEstimateBox(int frame) const{
        static unordered_map<int,vector<Box3D::Ptr>> boxes_gt;
        static bool is_first_run=true;
        if(is_first_run){
            is_first_run=false;
            std::ifstream fp_gt(tracking_estimation_path);
            cout<<"tracking_estimation_path:"<<tracking_estimation_path<<endl;

            if(!fp_gt.is_open()){
                return {};
            }

            string line_gt;
            while (getline(fp_gt,line_gt)){ //循环读取每行数据
                vector<string> tokens;
                split(line_gt,tokens," ");
                Box3D::Ptr box = Box3D::Box3dFromKittiTracking(tokens,cam_s.cam0);
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


    [[nodiscard]] cv::Mat ReadImage(int frame) const{
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

            auto [success,R_wc,t_wc] = GetBodyPose(index);

            Publisher::PubTransform(R_wc,t_wc,transform_broadcaster,ros::Time::now(),"world","body");

            int cnt=0;
            if(mode.find("prediction")!=string::npos){
                ///可视化深度学习预测的box
                vector<Box3D::Ptr> boxes = ReadPredictBox(index);
                for(auto &box : boxes){
                    cnt++;
                    box->VisCorners2d(img,cv::Scalar(255,255,255),cam_s.cam0);
                    auto cube_marker = CubeMarker(box->corners,cnt, BgrColor("green"));
                    marker_array.markers.push_back(cube_marker);

                    Mat34d axis_matrix = box->GetCoordinateVectorInCamera(4);
                    auto axis_markers = AxisMarker(axis_matrix,cnt);
                    marker_array.markers.push_back(std::get<0>(axis_markers));
                    marker_array.markers.push_back(std::get<1>(axis_markers));
                    marker_array.markers.push_back(std::get<2>(axis_markers));

                }
                cout<<"prediction_boxes:"<<boxes.size()<<endl;
            }

            if(mode.find("estimation")!=string::npos && success){
                ///可视化估计的框
                vector<Box3D::Ptr> boxes_est = ReadEstimateBox(index);
                for(auto &box : boxes_est){
                    cnt++;
                    Vector3d center_pt=Vector3d::Zero();
                    Mat38d corners_w = box->corners;
                    for(int i=0;i<8;++i){
                        corners_w.col(i) = R_wc * corners_w.col(i) + t_wc;
                        center_pt+=corners_w.col(i);
                    }
                    center_pt/=8.;
                    auto cube_marker = CubeMarker(corners_w, cnt, BgrColor("blue"));
                    marker_array.markers.push_back(cube_marker);

                    auto textMarker = TextMarker(center_pt, cnt, to_string(box->id), BgrColor("blue"), 1.2);
                    marker_array.markers.push_back(textMarker);

                    box->VisCorners2d(img, BgrColor("white",false),cam_s.cam0);
                }
                cout<<"estimation_boxes:"<<boxes_est.size()<<endl;
            }


            if(mode.find("gt")!=string::npos && success){
                ///可视化gt框
                vector<Box3D::Ptr> boxes_gt = ReadGroundtruthBox(index);
                for(auto &box : boxes_gt){
                    cnt++;
                    Vector3d center_pt=Vector3d::Zero();
                    Mat38d corners_w = box->corners;
                    for(int i=0;i<8;++i){
                        corners_w.col(i) = R_wc * corners_w.col(i) + t_wc;
                        center_pt+=corners_w.col(i);
                    }
                    center_pt/=8.;
                    auto cube_marker = CubeMarker(corners_w, cnt, BgrColor("red"));
                    marker_array.markers.push_back(cube_marker);

                    auto textMarker = TextMarker(center_pt, cnt, to_string(box->id), BgrColor("red"), 1.2);
                    marker_array.markers.push_back(textMarker);
                }
                cout<<"gt_boxes:"<<boxes_gt.size()<<endl;
            }

            pub_instance_marker.publish(marker_array);

            ///可视化
            viewer.ImageShow(img,io_para::kImageDatasetPeriod);

            index++;
        }

        cout<<"exit index:"<<index<<endl;


        return 0;

    }




    void InitGlobalParameters(const string &file_name,const string &seq_name){
        cfg cfg(file_name,seq_name);
        ///初始化logger
        MyLogger::InitLogger(file_name);
        ///初始化相机模型
        InitCamera(file_name,seq_name);
        ///初始化局部参数
        io_para::SetParameters(file_name,seq_name);

        det3d_para::SetParameters(file_name,seq_name);


        object3d_root_path=det3d_para::kDet3dPreprocessPath;
        tracking_gt_path=det3d_para::kGroundTruthPath;
        tracking_estimation_path= io_para::kOutputFolder + seq_name+".txt";
        camera_pose_file = io_para::kVinsResultPath;
        img_root_path=io_para::kImageDatasetLeft;
    }

    string object3d_root_path;
    string tracking_gt_path;
    string tracking_estimation_path;
    string camera_pose_file;

    string img_root_path;

    string mode;

    ImageViewer viewer;

    tf::TransformBroadcaster transform_broadcaster;

};


}

int main(int argc, char **argv)
{
    if(argc != 4){
        std::cerr<<"please input: rosrun dynamic_vins pub_object3d ${cfg_file} ${seq_name} mode"<< std::endl;
        return 1;
    }

    ros::init(argc, argv, "pub_object3d");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    dynamic_vins::PubDemo pub_demo;

    pub_demo.InitGlobalParameters(argv[1],argv[2]);

    string mode=argv[3];
    if(mode.find("gt")==string::npos && mode.find("estimation")==string::npos && mode.find("prediction")==string::npos){
        cerr<<"mode must be gt|estimation|prediction"<<endl;
        return -1;
    }
    pub_demo.mode = mode;

    std::cout<<"InitGlobalParameters finished"<<std::endl;


    //return dynamic_vins::PubObject3D(argc,argv);
    return pub_demo.PubFCOS3D(argc, argv,n);
    //return dynamic_vins::PubViodeFCOS3D(argc, argv,n);

    //return 0;
}



