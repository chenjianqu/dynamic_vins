/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/
//
// Created by chen on 2021/9/18.
//

#include <cstdio>
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <spdlog/spdlog.h>

#include "parameters.h"
#include "estimator/estimator.h"
#include "utility/visualization.h"
#include "estimator/dynamic.h"
#include "featureTracker/SegmentImage.h"
#include "utility/ViodeUtils.h"
#include "featureTracker/feature_tracker.h"


constexpr int QUEUE_SIZE=200;
constexpr double DELAY=0.005;

Estimator::Ptr estimator;
Infer::Ptr infer;
FeatureTracker::Ptr feature_tracker;


queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
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


inline cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr= cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
    return ptr->image.clone();
}




void sync_process()
{
    int cnt = 0;
    while(Config::ok.load(std::memory_order_seq_cst))
    {
        if(infer->get_queue_size() >= INFER_IMAGE_LIST_SIZE){
            std::this_thread::sleep_for(50ms);
            continue;
        }

        m_buf.lock();
        //等待图片
        if( ( Config::isInputSeg && (img0_buf.empty() || img1_buf.empty() || seg0_buf.empty() || seg1_buf.empty())) ||
            (!Config::isInputSeg && (img0_buf.empty() || img1_buf.empty()))) {
            m_buf.unlock();
            std::this_thread::sleep_for(2ms);
            continue;
        }

        debug_s("sync_process img0_buf:{} img1_buf{} seg0_buf{} seg1_buf{}",img0_buf.size(),img1_buf.size(),seg0_buf.size(),seg1_buf.size());

        static TicToc ticToc;
        ticToc.tic();

        ///下面以img0的时间戳为基准，找到与img0相近的图片
        SegImage img;
        img.color0= getImageFromMsg(img0_buf.front());
        img.time0=img0_buf.front()->header.stamp.toSec();
        img0_buf.pop();

        //sgLogger->debug("sync_process color0");

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

        //sgLogger->debug("sync_process color1");

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
            //sgLogger->debug("sync_process seg0");

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
            //sgLogger->debug("sync_process seg1");
        }

        m_buf.unlock();

        warn_s("----------Time : {} ----------",img.time0);

        ///rgb to gray
        if(img.gray0.empty()){
            if(Config::SLAM!=SlamType::RAW){
                img.setGrayImageGpu();
            }
            else{
                img.setGrayImage();
            }
        }
        else{
            if(Config::SLAM!=SlamType::RAW){
                img.setColorImageGpu();
            }
            else{
                img.setColorImage();
            }
        }

        static TicToc tt;

        ///set mask
        if(!Config::isInputSeg){
            if(Config::SLAM!=SlamType::RAW){
                tt.tic();
                infer->forward_tensor(img.color0,img.mask_tensor,img.insts_info);
                info_s("sync_process forward: {}",tt.toc_then_tic());
                if(Config::SLAM==SlamType::NAIVE){
                    img.setMaskGpuSimple();
                }
                else if(Config::SLAM == SlamType::DYNAMIC){
                    //img.setMask();
                    img.setMaskGpu();
                }
                info_s("sync_process setMask: {}",tt.toc_then_tic());
            }
        }
        else{
            if(Config::Dataset == DatasetType::VIODE){
                if(Config::SLAM==SlamType::NAIVE){
                    VIODE::setViodeMaskSimple(img);
                }
                else if(Config::SLAM == SlamType::DYNAMIC){
                    VIODE::setViodeMask(img);
                    //VIODE::setViodeMaskSimple(img);
                }
            }
            info_s("sync_process setMask: {}",tt.toc_then_tic());
        }

        infer->push_back(img);
        info_s("sync_process all:{} ms\n",ticToc.toc());

        /*cv::Mat show;
        cv::scaleAdd(img.color0,0.5,img.seg0,show);*/
        /*cv::imshow("show",img.merge_mask);
        cv::waitKey(1);*/

        /*cnt++;
        if(cnt==5){
            cv::imwrite("color0.png",img.color0);
            cv::imwrite("color1.png",img.color1);
        }*/
    }

    warn_s("sync_process 线程退出");
}




void feature_track()
{
    static TicToc tt;
    int cnt;
    while(Config::ok.load(std::memory_order_seq_cst)){
        if(auto img = infer->wait_for_result();img){
            tt.tic();
            if(Config::SLAM==SlamType::DYNAMIC){
                feature_tracker->insts_tracker->vel_map = estimator->insts_manager.getInstancesVelocity();
                FeatureMap features = feature_tracker->trackSemanticImage(*img);
                auto instances=feature_tracker->insts_tracker->setOutputFeature();
                estimator->push_back(img->time0,features,instances);
            }
            else if(Config::SLAM==SlamType::NAIVE){
                FeatureMap features = feature_tracker->trackImageNaive(*img);
                estimator->push_back(img->time0,features);
            }
            else{
                FeatureMap features = feature_tracker->trackImage(*img);
                estimator->push_back(img->time0,features);
            }

            ///发布跟踪可视化图像
            if (Config::SHOW_TRACK){
                pubTrackImage(feature_tracker->imTrack, img->time0);
                /*cv::imshow("img",feature_tracker->imTrack);
                cv::waitKey(1);*/
                /*string label=to_string(img.time0)+".jpg";
                if(saved_name.count(label)!=0){
                    cv::imwrite(label,imgTrack);
                }*/
            }

            info_t("**************feature_track:{} ms****************\n",tt.toc());
        }
    }
    warn_t("feature_track 线程退出");
}






void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Vector3d acc(dx, dy, dz);
    Vector3d gyr(rx, ry, rz);
    estimator->inputIMU(t, acc, gyr);
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        warn_v("restart the e!");
        estimator->clearState();
        estimator->setParameter();
    }
}

void terminal_callback(const std_msgs::BoolConstPtr &terminal_msg)
{
    if (terminal_msg->data == true)
    {
        cerr<<"terminal the e!"<<endl;
        ros::shutdown();
        Config::ok.store(false,std::memory_order_seq_cst);
    }
}

void imu_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true){
        warn_v("use IMU!");
        estimator->changeSensorType(1, Config::STEREO);
    }
    else{
        warn_v("disable IMU!");
        estimator->changeSensorType(0, Config::STEREO);
    }
}

void cam_switch_callback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true)
    {
        warn_v("use stereo!");
        estimator->changeSensorType(Config::USE_IMU, 1);
    }
    else
    {
        warn_v("use mono camera (left)!");
        estimator->changeSensorType(Config::USE_IMU, 0);
    }
}




int main(int argc, char **argv)
{
    if(argc != 2){
        cerr<<"please input: rosrun vins vins_node [config file]"<<endl;
        return 1;
    }
    string config_file = argv[1];
    cout<<fmt::format("config_file:{}",argv[1])<<endl;

    ros::init(argc, argv, "dynamic_vins");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);


    try{
        Config cfg(config_file);
        estimator.reset(new Estimator());
        infer.reset(new Infer);
        feature_tracker = std::make_unique<FeatureTracker>();
    }
    catch(std::runtime_error &e){
        vioLogger->critical(e.what());
        cerr<<e.what()<<endl;
        return -1;
    }

    estimator->setParameter();
    feature_tracker->readIntrinsicParameter(Config::CAM_NAMES);


#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    cout<<"waiting for image and imu..."<<endl;
    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(Config::IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_img0 = n.subscribe(Config::IMAGE0_TOPIC, 100, img0_callback);
    ros::Subscriber sub_img1 = n.subscribe(Config::IMAGE1_TOPIC, 100, img1_callback);

    ros::Subscriber sub_seg0,sub_seg1;
    if(Config::isInputSeg){
        sub_seg0 = n.subscribe(Config::IMAGE0_SEGMENTATION_TOPIC, 100, seg0_callback);
        sub_seg1 = n.subscribe(Config::IMAGE1_SEGMENTATION_TOPIC, 100, seg1_callback);
    }

    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, restart_callback);
    ros::Subscriber sub_terminal = n.subscribe("/vins_terminal", 100, terminal_callback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, imu_switch_callback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, cam_switch_callback);

    vioLogger->flush();

    std::thread vio_thread{&Estimator::processMeasurements, estimator};
    std::thread sync_thread{sync_process};
    std::thread fk_thread{feature_track};

    ros::spin();

    sync_thread.join();
    fk_thread.join();
    vio_thread.join();
    spdlog::drop_all();

    cerr<<"vins结束"<<endl;

    return 0;
}



