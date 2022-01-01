//
// Created by chen on 2021/9/18.
//

#include <cstdio>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <chrono>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "parameters.h"
#include "estimator/dynamic.h"
#include "featureTracker/feature_tracker.h"

using namespace std::literals;


constexpr int QUEUE_SIZE=20;
constexpr double DELAY=0.005;

std::shared_ptr<FeatureTracker> featureTracker;
ros::Publisher pub_image_track;


queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> seg0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
queue<sensor_msgs::ImageConstPtr> seg1_buf;
std::mutex m_buf;

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    if(img0_buf.size()<QUEUE_SIZE){
        img0_buf.push(img_msg);
    }
    m_buf.unlock();
}

void seg0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    if(seg0_buf.size()<QUEUE_SIZE)
        seg0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    if(img1_buf.size()<QUEUE_SIZE)
        img1_buf.push(img_msg);
    m_buf.unlock();
}

void seg1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    if(seg1_buf.size()<QUEUE_SIZE)
        seg1_buf.push(img_msg);
    m_buf.unlock();
}


cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr= cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
    return ptr->image.clone();
}



[[noreturn]] void sync_process()
{
    cout<<std::fixed;
    cout.precision(10);
    while(true)
    {
        m_buf.lock();
        //等待图片
        if( ( Config::isInputSeg && (img0_buf.empty() || img1_buf.empty() || seg0_buf.empty() || seg1_buf.empty())) ||
        (!Config::isInputSeg && (img0_buf.empty() || img1_buf.empty()))){
            m_buf.unlock();
            std::this_thread::sleep_for(2ms);
            continue;
        }

        ///下面以img0的时间戳为基准，找到与img0相近的图片
        SegImage img;
        img.color0= getImageFromMsg(img0_buf.front());
        img.time0=img0_buf.front()->header.stamp.toSec();
        img0_buf.pop();

        img.time1=img1_buf.front()->header.stamp.toSec();
        if(img.time0 + DELAY < img.time1){ //img0太早了
            m_buf.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        else if(img.time1 + DELAY < img.time0){ //img1太早了
            while(std::abs(img.time0 - img.time1) > DELAY){
                img1_buf.pop();
                img.time1=img1_buf.front()->header.stamp.toSec();
            }
        }
        img.color1= getImageFromMsg(img1_buf.front());
        img1_buf.pop();

        if(Config::isInputSeg)
        {
            img.seg0_time=seg0_buf.front()->header.stamp.toSec();
            if(img.time0 + DELAY < img.seg0_time){ //img0太早了
                m_buf.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            else if(img.seg0_time+DELAY < img.time0){ //seg0太早了
                while(std::abs(img.time0 - img.seg0_time) > DELAY){
                    seg0_buf.pop();
                    img.seg0_time=seg0_buf.front()->header.stamp.toSec();
                }
            }
            img.seg0= getImageFromMsg(seg0_buf.front());
            seg0_buf.pop();


            img.seg1_time=seg1_buf.front()->header.stamp.toSec();
            if(img.time0 + DELAY < img.seg1_time){ //img0太早了
                m_buf.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            else if(img.seg1_time+DELAY < img.time0){ //seg1太早了
                while(std::abs(img.time0 - img.seg1_time) > DELAY){
                    seg1_buf.pop();
                    img.seg1_time=seg1_buf.front()->header.stamp.toSec();
                }
            }
            img.seg1= getImageFromMsg(seg1_buf.front());
            seg1_buf.pop();
        }

        m_buf.unlock();


        ///rgb to gray
        if(img.gray0.empty())
            img.setGrayImage();
        else
            img.setColorImage();

        /// 前端跟踪

        if(Config::SLAM==SlamType::DYNAMIC){
            featureTracker->trackSemanticImage(img);
        }
        else if(Config::SLAM==SlamType::NAIVE){
            featureTracker->trackImageNaive(img);
        }
        else{
            featureTracker->trackImage(img);
        }
        std_msgs::Header header;
        header.stamp = ros::Time(img.time0);
        header.frame_id="world";
        sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", featureTracker->imTrack).toImageMsg();
        pub_image_track.publish(imgTrackMsg);

    }
}



int main(int argc, char **argv)
{
    cout<<"1"<<endl;
    ros::init(argc, argv, "front_end_node");
    cout<<"2"<<endl;

    ros::NodeHandle n("~");
    cout<<"3"<<endl;

    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    cout<<"4"<<endl;

    if(argc != 2){
        printf("please intput: rosrun vins vins_node [config file] \n");
        return 1;
    }
    string config_file = argv[1];
    cout<<"config_file:"<<argv[1]<<endl;

    Config cfg(config_file);

    featureTracker=std::make_shared<FeatureTracker>();
    featureTracker->readIntrinsicParameter(Config::CAM_NAMES);


#ifdef EIGEN_DONT_PARALLELIZE
ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    ROS_WARN("waiting for image and imu...");

    ros::Subscriber sub_img0 = n.subscribe(Config::IMAGE0_TOPIC, 100, img0_callback);
    ros::Subscriber sub_img1 = n.subscribe(Config::IMAGE1_TOPIC, 100, img1_callback);
    if(Config::isInputSeg){
        ros::Subscriber sub_seg0 = n.subscribe(Config::IMAGE0_SEGMENTATION_TOPIC, 100, seg0_callback);
        ros::Subscriber sub_seg1 = n.subscribe(Config::IMAGE1_SEGMENTATION_TOPIC, 100, seg1_callback);
    }
    pub_image_track = n.advertise<sensor_msgs::Image>("front_end_track", 1000);



    std::thread sync_thread{sync_process};
    ros::spin();

return 0;
}



