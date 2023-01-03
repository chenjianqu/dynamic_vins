/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "camera_model.h"

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "basic/def.h"
#include "parameters.h"
#include "utils/dataset/kitti_utils.h"
#include "utils/convert_utils.h"

namespace dynamic_vins{\

std::vector<Eigen::Matrix3d> R_IC;
std::vector<Eigen::Vector3d> T_IC;

CameraInfo cam_s;//用于segmentation线程的相机
CameraInfo cam_t;//用于tracking线程的相机
CameraInfo cam_v;//用于VIO线程的相机


/*
void InitCamera(const std::string& config_path){

cv::FileStorage fs(config_path, cv::FileStorage::READ);
if(!fs.isOpened()){
    throw std::runtime_error("ERROR: Wrong path to settings:" + config_path);
}

///直接从kitti的参数文件中读取
if(cfg::dataset == DatasetType::kKitti){
    if(fs["kitti_calib_path"].isNone()){
        cerr<<"Use the kitti dataset,but not set kitti_calib_path"<<endl;
        fs.release();
        std::terminate();
    }
    string kitti_calib_path;
    fs["kitti_calib_path"] >> kitti_calib_path;
    auto calib_map = kitti::ReadCalibFile(kitti_calib_path);

    cam0 = std::make_shared<PinHoleCamera>();
    cam0->image_width = cfg::kInputWidth;
    cam0->image_height = cfg::kInputHeight;
    cam0->fx = calib_map["P2"](0,0);
    cam0->fy = calib_map["P2"](1,1);
    cam0->cx = calib_map["P2"](0,2);
    cam0->cy = calib_map["P2"](1,2);
    cam0->baseline = 0;
    double baseline_2 = calib_map["P2"](0,3) / (- cam0->fx);
    cout<<"baseline_2:"<<baseline_2<<endl;

    cam1 = std::make_shared<PinHoleCamera>();
    cam1->image_width = cfg::kInputWidth;
    cam1->image_height = cfg::kInputHeight;
    cam1->fx = calib_map["P3"](0,0);
    cam1->fy = calib_map["P3"](1,1);
    cam1->cx = calib_map["P3"](0,2);
    cam1->cy = calib_map["P3"](1,2);
    double baseline_3 = calib_map["P3"](0,3) / (- cam1->fx);
    cout<<"baseline_3:"<<baseline_3<<endl;

    cam1->baseline = baseline_3 - baseline_2;

    cout<<"P2:\n"<<calib_map["P2"]<<endl;
    cout<<"P3:\n"<<calib_map["P3"]<<endl;
}
///从文件中读取相机内参
else if(cfg::dataset == DatasetType::kViode || cfg::dataset==DatasetType::kEuRoc){
    vector<string> cam_path = GetCameraPath(config_path);

    cam0 = std::make_shared<PinHoleCamera>();
    cam0->ReadFromYamlFile(cam_path[0]);

    if(cfg::kCamNum>1){
        cam1 = std::make_shared<PinHoleCamera>();
        cam1->ReadFromYamlFile(cam_path[1]);
    }
}
else{
    std::cerr<<"InitCamera() not is implemented, as dataset is "<<cfg::dataset_name<<endl;
}

fs.release();

cam0->inv_k11 = 1.f / cam0->fx;
cam0->inv_k22 = 1.f / cam0->fy;
cam0->inv_k13 = -cam0->cx / cam0->fx;
cam0->inv_k23 = -cam0->cy / cam0->fy;

cam1->inv_k11 = 1.f / cam1->fx;
cam1->inv_k22 = 1.f / cam1->fy;
cam1->inv_k13 = -cam1->cx / cam1->fx;
cam1->inv_k23 = -cam1->cy / cam1->fy;

fmt::print("Camera Intrinsic:\n");
fmt::print("cam0 - fx:{},fy:{},cx:{},cy:{},baseline:{}\n",cam0->fx,cam0->fy,cam0->cx,cam0->cy,cam0->baseline);
if(cfg::is_stereo){
    fmt::print("cam1 - fx:{},fy:{},cx:{},cy:{},baseline:{}\n",cam1->fx,cam1->fy,cam1->cx,cam1->cy,cam1->baseline);
}
cout<<"Read Camera Intrinsic Finished"<<endl;

}
 */


/*

bool PinHoleCamera::ReadFromYamlFile(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened()){
        return false;
    }

    if (!fs["model_type"].isNone()){
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (sModelType != "PINHOLE"){
            return false;
        }
    }

    fs["camera_name"] >> camera_name;
    fs["image_width"] >>image_width;
    fs["image_height"] >> image_height;

    cv::FileNode n = fs["distortion_parameters"];
    k1 = static_cast<float>(n["k1"]);
    k2 = static_cast<float>(n["k2"]);
    p1 = static_cast<float>(n["p1"]);
    p2 = static_cast<float>(n["p2"]);

    n = fs["projection_parameters"];
    fx = static_cast<float>(n["fx"]);
    fy = static_cast<float>(n["fy"]);
    cx = static_cast<float>(n["cx"]);
    cy = static_cast<float>(n["cy"]);

    return true;
}*/


/**
 * 将特征点从图像平面反投影到归一化平面,并去畸变
 * @param p
 * @param P
 */
/*void PinHoleCamera::LiftProjective(const Vec2d& p, Vec3d& P) const
{
    double mx_d, my_d,mx2_d, mxy_d, my2_d, mx_u, my_u;

    mx_d = inv_k11 * p(0) + inv_k13;
    my_d = inv_k22 * p(1) + inv_k23;

    ///TODO
    if (1){
        mx_u = mx_d;
        my_u = my_d;
    }
    else{
        double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;

        // Apply inverse distortion model
        // proposed by Heikkila
        mx2_d = mx_d*mx_d;
        my2_d = my_d*my_d;
        mxy_d = mx_d*my_d;
        rho2_d = mx2_d+my2_d;
        rho4_d = rho2_d*rho2_d;
        radDist_d = k1*rho2_d+k2*rho4_d;
        Dx_d = mx_d*radDist_d + p2*(rho2_d+2*mx2_d) + 2*p1*mxy_d;
        Dy_d = my_d*radDist_d + p1*(rho2_d+2*my2_d) + 2*p2*mxy_d;
        inv_denom_d = 1/(1+4*k1*rho2_d+6*k2*rho4_d+8*p1*my_d+8*p2*mx_d);

        mx_u = mx_d - inv_denom_d*Dx_d;
        my_u = my_d - inv_denom_d*Dy_d;
    }

    // Obtain a projective ray
    P << mx_u, my_u, 1.0;
}*/



/**
 * 读取相机到IMU的外参
 * @param config_path
 * @param fx 相机的焦距
 */
void ReadExtrinsicParameters(const std::string& config_path,const std::string& seq_name,CameraInfo &cam){
    ///设置为单位矩阵
    if (cfg::is_estimate_ex == 2){
        R_IC.emplace_back(Eigen::Matrix3d::Identity());
        T_IC.emplace_back(Eigen::Vector3d::Zero());
        if(cfg::kCamNum==2){
            R_IC.emplace_back(Eigen::Matrix3d::Identity());
            T_IC.emplace_back(Eigen::Vector3d::Zero());
        }
        return;
    }

    ///从文件中读取外参
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if(!fs.isOpened()){
        throw std::runtime_error(std::string("ERROR: Wrong path to settings:"+config_path));
    }

    ///直接从kitti的参数文件中读取
    if(cfg::dataset == DatasetType::kKitti){
        if(fs["kitti_calib_path"].isNone()){
            cerr<<"Use the kitti dataset,but not set kitti_calib_path"<<endl;
            fs.release();
            std::terminate();
        }
        string kitti_calib_path;
        fs["kitti_calib_path"] >> kitti_calib_path;
        kitti_calib_path += seq_name+".txt";
        auto calib_map = kitti::ReadCalibFile(kitti_calib_path);

        //将点从IMU坐标系变换到相机坐标系0的变换矩阵
        Mat4d T_imu_c0 =calib_map["Tr_imu_velo"] * calib_map["Tr_velo_cam"];

        //如 baseline_2=4.485728000000e+01 / (-7.215377000000e+02) =−0.062169004
        double baseline_2 = calib_map["P2"](0,3) / calib_map["P2"](0,0);

        Mat4d T_c0_c2 = Mat4d::Identity();
        T_c0_c2(0,3) = -baseline_2;
        Mat4d T_imu_c2 = T_imu_c0*T_c0_c2;
        Mat4d T_ic0 = T_imu_c2.inverse();

        /*R_IC.emplace_back(T_ic0.block<3, 3>(0, 0));
        T_IC.emplace_back(T_ic0.block<3, 1>(0, 3));

        if(cfg::kCamNum == 2){
            double baseline_3 = calib_map["P3"](0,3) / (- cam1->fx);
            Mat4d T_c0_c3 = Mat4d::Identity();
            T_c0_c2(0,3) = - baseline_3;
            Mat4d T_imu_c3 = T_imu_c0*T_c0_c3;
            Mat4d T_ic1 = T_imu_c3.inverse();
            R_IC.emplace_back(T_ic1.block<3, 3>(0, 0));
            T_IC.emplace_back(T_ic1.block<3, 1>(0, 3));
        }*/

        ///简单的设置IMU坐标系在cam2上
        R_IC.emplace_back(Mat3d::Identity());
        T_IC.emplace_back(Vec3d::Zero());

        if(cfg::kCamNum == 2){
            //如 baseline_2=-3.395242000000e+02 / (-7.215377000000e+02) = 0.470556424
            double baseline_3 = calib_map["P3"](0,3) / calib_map["P3"](0,0);
            double baseline = baseline_3-baseline_2;//0.5
            cam.baseline = std::abs(baseline);
            R_IC.emplace_back(Mat3d::Identity());
            T_IC.emplace_back(Vec3d(cam.baseline,0,0));
        }

    }
    ///从文件中读取相机内参
    else if(cfg::dataset == DatasetType::kViode || cfg::dataset== DatasetType::kEuRoc || cfg::dataset==DatasetType::kCustom){
        cv::Mat cv_T;
        fs["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d body_T_cam0;
        cv::cv2eigen(cv_T, body_T_cam0);
        R_IC.emplace_back(body_T_cam0.block<3, 3>(0, 0));
        T_IC.emplace_back(body_T_cam0.block<3, 1>(0, 3));

        if(cfg::kCamNum == 2){
            Eigen::Matrix4d body_T_cam1;
            fs["body_T_cam1"] >> cv_T;
            cv::cv2eigen(cv_T, body_T_cam1);
            R_IC.emplace_back(body_T_cam1.block<3, 3>(0, 0));
            T_IC.emplace_back(body_T_cam1.block<3, 1>(0, 3));

            ///计算baseline
            Eigen::Matrix4d cam0_T_cam1 = body_T_cam0.inverse() * body_T_cam1;
            cam.baseline = std::abs(cam0_T_cam1(0,3));
            Debugv("ReadExtrinsicParameters() cam0_T_cam1:{}", EigenToStr(cam0_T_cam1));
        }

    }
    else{
        std::cerr<<"ReadExtrinsicParameters() not is implemented, as dataset is "<<cfg::dataset_name<<endl;
        std::terminate();
    }

    Debugv("ReadExtrinsicParameters() baseline:{}", cam.baseline);


    std::string kBasicDir;
    fs["basic_dir"] >> kBasicDir;
    string kOutputFolder = kBasicDir+"data/output/";
    cfg::kExCalibResultPath = kOutputFolder + "extrinsic_parameter.txt";

    fs.release();
}


/**
 * 获得相机模型的路径
 * @param config_path
 * @return
 */
vector<string> GetCameraPath(const string &config_path){
    auto pn = config_path.find_last_of('/');
    std::string config_dir = config_path.substr(0, pn);
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened()){
        return {};
    }
    vector<string> ans;

    if(!fs["cam0_calib"].isNone()){
        std::string cam0_calib;
        fs["cam0_calib"] >> cam0_calib;
        std::string cam0Path = config_dir + "/" + cam0_calib;
        ans.push_back(cam0Path);

        if(!fs["cam1_calib"].isNone()){
            std::string cam1_calib;
            fs["cam1_calib"] >> cam1_calib;
            std::string cam1Path = config_dir + "/" + cam1_calib;
            ans.push_back(cam1Path);
        }
    }

    int num_of_cam = fs["num_of_cam"];
    if(num_of_cam==2 && ans.size()==1){
        throw std::runtime_error("GetCameraPath() num_of_cam==2, but cam path==1");
    }

    fs.release();
    return ans;
}


void SetCameraIntrinsicByK(CameraInfo &cam){
    cam.fx0 = cam.K0.at<float>(0,0);
    cam.fy0 = cam.K0.at<float>(1,1);
    cam.cx0 = cam.K0.at<float>(0,2);
    cam.cy0 = cam.K0.at<float>(1,2);
    if(cfg::is_stereo){
        cam.fx1 = cam.K1.at<float>(0,0);
        cam.fy1 = cam.K1.at<float>(1,1);
        cam.cx1 = cam.K1.at<float>(0,2);
        cam.cy1 = cam.K1.at<float>(1,2);
    }
}



void InitCameraByConfig(const std::string& config_path,const std::string& seq_name,CameraInfo &cam){
    vector<string> cam_paths = GetCameraPath(config_path);

    if(!cam_paths.empty()){
        cam.cam0 = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam_paths[0]);

        if(cfg::is_stereo){
            if(cam_paths.size()==1){
                cerr<<"InitCameraByConfig() not found right camera config:"<<config_path<<endl;
                std::terminate();
            }
            cam.cam1 = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(cam_paths[1]);
        }
    }
    else{
        if(cfg::dataset == DatasetType::kKitti){
            cv::FileStorage fs(config_path, cv::FileStorage::READ);
            string kitti_calib_path;
            if(fs["kitti_calib_path"].isNone()){
                cerr<<"InitCameraByConfig() not found <kitti_calib_path> in "<<config_path<<endl;
                std::terminate();
            }
            fs["kitti_calib_path"] >> kitti_calib_path;
            kitti_calib_path += seq_name+".txt";

            int image_width,image_height;
            fs["image_width"] >> image_width;
            fs["image_height"] >> image_height;

            fs.release();

            cv::Size size(image_width,image_height);
            cam.cam0 = camodocal::CameraFactory::instance()->generateCamera(
                    CamModelType::PINHOLE,"camera",size);
            cam.cam1 = camodocal::CameraFactory::instance()->generateCamera(
                    CamModelType::PINHOLE,"camera",size);

            auto calib_map = kitti::ReadCalibFile(kitti_calib_path);

            //0 k1();1 k2();2 p1();3 p2();4 fx();5 fy();6 cx();7 cy();
            vector<double> left_cam_para =
                    {0,0,0,0,
                     calib_map["P2"](0,0),calib_map["P2"](1,1),calib_map["P2"](0,2),
                     calib_map["P2"](1,2)};
            cam.cam0->readParameters(left_cam_para);

            vector<double> right_cam_para =
                    {0,0,0,0,
                     calib_map["P3"](0,0),calib_map["P3"](1,1),calib_map["P3"](0,2),
                     calib_map["P3"](1,2)};
            cam.cam1->readParameters(right_cam_para);
        }
        else{
            cerr<<"InitCameraByConfig() not found camera config:"<<config_path<<endl;
            std::terminate();
        }


    }

}


/**
 * 初始化相机内参
 * @param config_path
 * @param cam
 */
void InitOneCamera(const std::string& config_path,const std::string& seq_name,CameraInfo &cam){

    InitCameraByConfig(config_path,seq_name,cam);

    cam.model_type = cam.cam0->modelType();

    ///获取内参
    if(cam.model_type==CamModelType::PINHOLE){
        //0 k1();1 k2();2 p1();3 p2();4 fx();5 fy();6 cx();7 cy();
        vector<double> left_cam_para;
        cam.cam0->writeParameters(left_cam_para);
        cam.K0 = ( cv::Mat_<float> ( 3,3 ) <<
                left_cam_para[4], 0.0, left_cam_para[6],
                0.0, left_cam_para[5], left_cam_para[7],
                0.0, 0.0, 1.0 );
        cam.D0 = ( cv::Mat_<float> ( 4,1 ) <<
                left_cam_para[0], left_cam_para[1], left_cam_para[2], left_cam_para[3]);//畸变系数 k1 k2 p1 p2

        if(cfg::is_stereo){
            vector<double> right_cam_para;
            cam.cam1->writeParameters(right_cam_para);
            cam.K1 = ( cv::Mat_<float> ( 3,3 ) <<
                    right_cam_para[4], 0.0, right_cam_para[6], 0.0, right_cam_para[5], right_cam_para[7], 0.0, 0.0, 1.0 );
            cam.D1 = ( cv::Mat_<float> ( 4,1 ) <<
                    right_cam_para[0], right_cam_para[1], right_cam_para[2], right_cam_para[3]);//畸变系数 k1 k2 p1 p2
        }

        SetCameraIntrinsicByK(cam);
    }
    else if(cam.cam0->modelType()==CamModelType::MEI){
        //throw std::runtime_error("InitOneCamera not implement");
        //std::terminate();
    }
    else{
        throw std::runtime_error("InitOneCamera() not implement");
    }

    ///设置是否对输入图像进行去畸变处理
    if(cfg::use_line && cfg::dataset==DatasetType::kEuRoc && cfg::is_undistort_input==false){
        throw std::runtime_error("InitOneCamera() cfg::use_line==true && "
                                 "cfg::dataset==DatasetType::kEuRoc && cfg::is_undistort_input==false");
    }

    ///获取去畸变的映射矩阵
    if(cfg::is_undistort_input){

        cam.K0 = cam.cam0->initUndistortRectifyMap(cam.left_undist_map1,cam.left_undist_map2);
        cam.D0 = ( cv::Mat_<float> ( 4,1 ) << 0, 0, 0, 0);//畸变系数 k1 k2 p1 p2

        Debugv("InitOneCamera() initUndistortRectifyMap K0:\n{}", CvMatToStr<float>(cam.K0));

        //cv::Size image_size(left_cam_dl->imageWidth(),left_cam_dl->imageHeight());
        //cv::initUndistortRectifyMap(left_K,left_D,cv::Mat(),left_K,image_size,
        //                            CV_16SC2,left_undist_map1,left_undist_map2);
        //dynamic_cast<camodocal::CataCamera*>(left_cam_dl.get())->initUndistortMap(left_undist_map1,left_undist_map2);

        if(cfg::is_stereo){
            cam.K1 = cam.cam1->initUndistortRectifyMap(cam.right_undist_map1,cam.right_undist_map2);
            cam.D1 = ( cv::Mat_<float> ( 4,1 ) << 0, 0, 0, 0);//畸变系数 k1 k2 p1 p2
            Debugv("InitOneCamera() initUndistortRectifyMap K1:\n{}", CvMatToStr<float>(cam.K1));
            //cv::Size image_size_2(right_cam_dl->imageWidth(),right_cam_dl->imageHeight());
            //cv::initUndistortRectifyMap(right_K,right_D,cv::Mat(),right_K,image_size_2,
            //                           CV_16SC2,right_undist_map1,right_undist_map2);

            //dynamic_cast<camodocal::CataCamera*>(right_cam_dl.get())->initUndistortMap(right_undist_map1,right_undist_map2);
        }

        ///由于会对整张图像进行去畸变，因此重新设置相机的内参
        if(cam.cam0->modelType()==camodocal::Camera::ModelType::PINHOLE){
            SetCameraIntrinsicByK(cam);
            //0 k1();1 k2();2 p1();3 p2();4 fx();5 fy();6 cx();7 cy();
            vector<double> left_cam_para = {0,0,0,0,cam.fx0,cam.fy0,cam.cx0,cam.cy0};
            cam.cam0->readParameters(left_cam_para);
            if(cfg::is_stereo){
                vector<double> right_cam_para = {0,0,0,0,cam.fx1,cam.fy1,cam.cx1,cam.cy1};
                cam.cam1->readParameters(right_cam_para);
            }
        }
        else if(cam.cam0->modelType()==camodocal::Camera::ModelType::MEI){
            cerr<<"InitOneCamera not implement"<<endl;
            std::terminate();
        }
        else{
            cerr<<"InitOneCamera not implement"<<endl;
            std::terminate();
        }
    }

    Debugv("InitOneCamera() left cam intrinsic fx:{} fy:{} cx:{} cy:{}",cam.fx0,cam.fy0,cam.cx0,cam.cy0);
    if(cfg::is_stereo){
        Debugv("InitOneCamera() right cam intrinsic fx:{} fy:{} cx:{} cy:{}",cam.fx1,cam.fy1,cam.cx1,cam.cy1);
    }

}


void InitCamera(const std::string& config_path,const std::string& seq_name){
    ///初始化内参
    InitOneCamera(config_path,seq_name,cam_s);
    //InitOneCamera(config_path,cam_t);
    //InitOneCamera(config_path,cam_v);

    ///初始化外参
    ReadExtrinsicParameters(config_path,seq_name,cam_s);

    cam_t = cam_s;
    cam_v = cam_s;
}




}