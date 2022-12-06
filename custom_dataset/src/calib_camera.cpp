/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include<iostream>
#include <string>

#include <opencv2/imgproc/types_c.h>
#include<opencv2/opencv.hpp>

using namespace std;


int main()
{
    string image_path = "/home/chen/datasets/MyData/ZED/calib/cam0/*.png";//待处理图路径


    cv::Mat image, img_gray;
    int BOARDSIZE[2]{ 6,9 };//棋盘格每行每列角点个数(角点即交叉点)

    vector<cv::String> images_path;//创建容器存放读取图像路径
    cv::glob(image_path, images_path);//读取指定文件夹下图像

    vector<cv::Point3f> obj_world_pts;//三维世界坐标
    //转世界坐标系
    for (int i = 0; i < BOARDSIZE[1]; i++){
        for (int j = 0; j < BOARDSIZE[0]; j++){
            obj_world_pts.emplace_back(j, i, 0);
        }
    }

    vector<vector<cv::Point3f>> objpoints_img;//保存棋盘格上角点的三维坐标
    vector<vector<cv::Point2f>> images_points;//保存所有角点
    vector<cv::Point2f> img_corner_points;//保存每张图检测到的角点

    for (int i = 0; i < images_path.size(); i++){
        image = imread(images_path[i]);
        cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);
        //检测角点
        bool found_success = findChessboardCorners(img_gray, cv::Size(BOARDSIZE[0], BOARDSIZE[1]),
                                                   img_corner_points,
                                                   cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
        //显示角点
        if (found_success){
            //迭代终止条件
            cv::TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
            //进一步提取亚像素角点
            cornerSubPix(img_gray, img_corner_points, cv::Size(11, 11),cv::Size(-1, -1), criteria);
            //绘制角点
            drawChessboardCorners(image, cv::Size(BOARDSIZE[0], BOARDSIZE[1]), img_corner_points,
                                  found_success);
 
            objpoints_img.push_back(obj_world_pts);//从世界坐标系到相机坐标系
            images_points.push_back(img_corner_points);
        }
        //char *output = "image";
        cv::imshow("image", image);
        cv::waitKey(30);
    }

    /*
	计算内参和畸变系数等
	*/
 
    cv::Mat cameraMatrix, distCoeffs, R, T;//内参矩阵，畸变系数，旋转量，偏移量
    calibrateCamera(objpoints_img, images_points, img_gray.size(),
                    cameraMatrix, distCoeffs, R, T);
 
    cout << "cameraMatrix:" << endl;
    cout << cameraMatrix << endl;
 
    cout << "*****************************" << endl;
    cout << "distCoeffs:" << endl;
    cout << distCoeffs << endl;
    cout << "*****************************" << endl;
 
    cout << "Rotation vector:" << endl;
    cout << R << endl;
 
    cout << "*****************************" << endl;
    cout << "Translation vector:" << endl;
    cout << T << endl;
 

    cv::Mat src, dst;
    src = cv::imread(images_path[0]);  //读取校正前图像
    cv::undistort(src, dst, cameraMatrix, distCoeffs);
    cv::imshow("dst", dst);
    cv::imshow("image", src);
    cv::waitKey(0);
 
    return 0;
}