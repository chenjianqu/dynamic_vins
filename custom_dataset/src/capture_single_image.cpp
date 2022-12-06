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
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "io_utils.h"

using namespace std;
using namespace cv;



const string kBaseDIR="/home/chen/datasets/MyData/ZED";

const int kWidth = 1280;
const int kHeight = 720;

string seq_name;
string dataset_path,left_image_path,right_image_path;

int save_counter=0;

cv::Rect rect0,rect1;

int ShowAndWrite(int argc, char** argv)
{

    cv:: VideoCapture cap = cv::VideoCapture(0);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, kWidth*2);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, kHeight);
    cv::Mat frame, img;
    cap.read(frame);
    cout<<frame.size<<endl;

    int count=0;

    while(cap.isOpened()){
        auto start = std::chrono::steady_clock::now();

        cap.read(frame);

        if(frame.empty()){
            std::cout << "Read frame failed!" << std::endl;
            break;
        }

        cv::imshow("Stereo", frame);

        char ch = cv::waitKey(30);
        if(ch== ' '){
            cv::Mat img0=frame(rect0);
            cv::Mat img1=frame(rect1);
            ///序号
            string pad_number = PadNumber(save_counter,6);

            ///保存图像
            cv::imwrite(left_image_path+pad_number+".png",img0);
            cv::imwrite(right_image_path+pad_number+".png",img1);

            save_counter++;

            cout<<"Save "<<save_counter<<endl;
            cout<<left_image_path+pad_number+".png"<<endl;
            cout<<right_image_path+pad_number+".png"<<endl;
        }
        else if(ch=='q'){
            break;
        }

        count++;
    }
}


int main(int argc, char** argv) {

    seq_name = "calib";

    dataset_path = kBaseDIR + "/" + seq_name + "/";
    left_image_path = dataset_path+"cam0/";
    right_image_path = dataset_path+"cam1/";

    rect0 = cv::Rect(0,0,kWidth,kHeight);
    rect1 = cv::Rect(kWidth,0,kWidth,kHeight);

    ///创建文件夹
    ClearDirectory(dataset_path);
    ClearDirectory(left_image_path);
    ClearDirectory(right_image_path);


    return ShowAndWrite(argc,argv);
}
