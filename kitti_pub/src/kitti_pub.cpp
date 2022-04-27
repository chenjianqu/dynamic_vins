#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <mutex>
#include <list>
#include <thread>
#include <regex>
#include<filesystem>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

using namespace std;

void split(const std::string& source, std::vector<std::string>& tokens, const string& delimiters = " ") {
    std::regex re(delimiters);
    std::copy(std::sregex_token_iterator(source.begin(), source.end(),re,-1),
              std::sregex_token_iterator(),
              std::back_inserter(tokens));
}


class TicToc{
public:
    TicToc(){
        Tic();
    }
    void Tic(){
        start_ = std::chrono::system_clock::now();
    }
    double Toc(){
        end_ = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_ - start_;
        return elapsed_seconds.count() * 1000;
    }
    double TocThenTic(){
        auto t= Toc();
        Tic();
        return t;
    }
    void TocPrintTic(const char* str){
        cout << str << ":" << Toc() << " ms" << endl;
        Tic();
    }
private:
    std::chrono::time_point<std::chrono::system_clock> start_, end_;
};



// 检查一个路径是否是目录
bool checkIsDir(const string &dir) {
    if (! std::filesystem::exists(dir)) {
        cout<<dir<<" not exists. Please check."<<endl;
        return false;
    }
    std::filesystem::directory_entry entry(dir);
    if (entry.is_directory())
        return true;
    return false;
}

// 搜索一个目录下所有的图像文件，以 jpg,jpeg,png 结尾的文件
void getAllImageFiles(const string& dir, vector<string> &files) {
    // 首先检查目录是否为空，以及是否是目录
    if (!checkIsDir(dir))
        return;

    // 递归遍历所有的文件
    std::filesystem::directory_iterator iters(dir);
    for(auto &iter: iters) {
        string file_path(dir);
        file_path += "/";
        file_path += iter.path().filename();

        // 查看是否是目录，如果是目录则循环递归
        if (checkIsDir(file_path)) {
            getAllImageFiles(file_path, files);
        }
        //不是目录则检查后缀是否是图像
        else {
            string extension = iter.path().extension(); // 获取文件的后缀名
            if (extension == ".jpg" || extension == ".png" || extension == ".jpeg") {
                files.push_back(file_path);
            }
        }
    }
}





int main(int argc, char** argv)
{
    setlocale(LC_ALL, "");//防止中文乱码
    ros::init(argc, argv, "kitti_pub");
    ros::start();

    ros::NodeHandle nh;

    string left_images_base = argv[1];
    string right_images_base = argv[2];

    vector<string> left_names;
    vector<string> right_names;
    getAllImageFiles(left_images_base,left_names);
    getAllImageFiles(right_images_base,right_names);

    cout<<"left:"<<left_names.size()<<endl;
    cout<<"right:"<<right_names.size()<<endl;
    if(left_names.size() != right_names.size()){
        cerr<< "left and right image number is not equal!"<<endl;
        return -1;
    }

    std::sort(left_names.begin(),left_names.end());
    std::sort(right_names.begin(),right_names.end());

    image_transport::ImageTransport it(nh);
    image_transport::Publisher pub_left = it.advertise("/kitti_pub/left",1);
    image_transport::Publisher pub_right = it.advertise("/kitti_pub/right",1);

    int index=0;
    double time=0.;

    TicToc tt;

    while(ros::ok()){
        tt.Tic();

        if(index >= left_names.size()){
            break;
        }
        cout<<left_names[index]<<endl;

        cv::Mat left_img = cv::imread(left_names[index],-1);
        cv::Mat right_img = cv::imread(right_names[index],-1);

        std::filesystem::path name(left_names[index]);
        std::string name_stem =  name.stem().string();//获得文件名(不含后缀)

        std_msgs::Header header;
        header.stamp=ros::Time(time);
        header.seq= std::stoi(name_stem);

        sensor_msgs::ImagePtr msg_left = cv_bridge::CvImage(header, "bgr8", left_img).toImageMsg();
        sensor_msgs::ImagePtr msg_right = cv_bridge::CvImage(header, "bgr8", right_img).toImageMsg();
        pub_left.publish(msg_left);
        pub_right.publish(msg_right);

        ros::spinOnce();

        int delta_t =(int) tt.Toc();
        int wait_time = 100 - delta_t;
        if(wait_time>0)
            std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));

        time+=0.05; // 时间戳
        index++;
    }

    return 0;
}