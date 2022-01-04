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
/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <chrono>
#include <future>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <spdlog/spdlog.h>

#include "parameters.h"
#include "estimator/estimator.h"
#include "utility/visualization.h"
#include "estimator/dynamic.h"
#include "featureTracker/segment_image.h"
#include "utility/viode_utils.h"
#include "featureTracker/feature_tracker.h"
#include "FlowEstimating/flow_estimator.h"
#include "FlowEstimating/flow_visual.h"

constexpr int kQueueSize=200;
constexpr double kDelay=0.005;

Estimator::Ptr estimator;
InstanceSegmentor::Ptr inst_segmentor;
FeatureTracker::Ptr feature_tracker;
FlowEstimator::Ptr flow_estimator;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> seg0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
queue<sensor_msgs::ImageConstPtr> seg1_buf;
std::mutex m_buf;

void Img0Callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    if(img0_buf.size() < kQueueSize){
        img0_buf.push(img_msg);
    }
    m_buf.unlock();
}

void Seg0Callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    if(seg0_buf.size() < kQueueSize)
        seg0_buf.push(img_msg);
    m_buf.unlock();
}

void Img1Callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    if(img1_buf.size() < kQueueSize)
        img1_buf.push(img_msg);
    m_buf.unlock();
}

void Seg1Callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    if(seg1_buf.size() < kQueueSize)
        seg1_buf.push(img_msg);
    m_buf.unlock();
}


inline cv::Mat GetImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr= cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::BGR8);
    return ptr->image.clone();
}





SegImage SyncProcess()
{
    SegImage img;

    while(Config::ok.load(std::memory_order_seq_cst))
    {
        if(inst_segmentor->GetQueueSize() >= kInferImageListSize){
            std::this_thread::sleep_for(50ms);
            continue;
        }
        m_buf.lock();
        //等待图片
        if((Config::is_input_seg && (img0_buf.empty() || img1_buf.empty() || seg0_buf.empty() || seg1_buf.empty())) ||
           (!Config::is_input_seg && (img0_buf.empty() || img1_buf.empty()))) {
            m_buf.unlock();
            std::this_thread::sleep_for(2ms);
            continue;
        }

        ///下面以img0的时间戳为基准，找到与img0相近的图片
        img.color0= GetImageFromMsg(img0_buf.front());
        img.time0=img0_buf.front()->header.stamp.toSec();
        img0_buf.pop();

        img.time1=img1_buf.front()->header.stamp.toSec();
        if(img.time0 + kDelay < img.time1){ //img0太早了
            m_buf.unlock();
            std::this_thread::sleep_for(2ms);
            continue;
        }
        else if(img.time1 + kDelay < img.time0){ //img1太早了
            while(std::abs(img.time0 - img.time1) > kDelay){
                img1_buf.pop();
                img.time1=img1_buf.front()->header.stamp.toSec();
            }
        }
        img.color1= GetImageFromMsg(img1_buf.front());
        img1_buf.pop();


        if(Config::is_input_seg)
        {
            img.seg0_time=seg0_buf.front()->header.stamp.toSec();
            if(img.time0 + kDelay < img.seg0_time){ //img0太早了
                m_buf.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            else if(img.seg0_time + kDelay < img.time0){ //seg0太早了
                while(std::abs(img.time0 - img.seg0_time) > kDelay){
                    seg0_buf.pop();
                    img.seg0_time=seg0_buf.front()->header.stamp.toSec();
                }
            }
            img.seg0= GetImageFromMsg(seg0_buf.front());
            seg0_buf.pop();
            //sg_logger->debug("sync_process seg0");

            img.seg1_time=seg1_buf.front()->header.stamp.toSec();
            if(img.time0 + kDelay < img.seg1_time){ //img0太早了
                m_buf.unlock();
                std::this_thread::sleep_for(2ms);
                continue;
            }
            else if(img.seg1_time + kDelay < img.time0){ //seg1太早了
                while(std::abs(img.time0 - img.seg1_time) > kDelay){
                    seg1_buf.pop();
                    img.seg1_time=seg1_buf.front()->header.stamp.toSec();
                }
            }
            img.seg1= GetImageFromMsg(seg1_buf.front());
            seg1_buf.pop();
            //sg_logger->debug("sync_process seg1");
        }

        m_buf.unlock();

        break;
    }

    return img;
}



void ImageProcess()
{
    int cnt = 0;
    while(Config::ok.load(std::memory_order_seq_cst))
    {
        SegImage img = SyncProcess();
        WarnS("----------Time : {} ----------", img.time0);

        ///rgb to gray
        if(img.gray0.empty()){
            if(Config::slam != SlamType::kRaw)
                img.SetGrayImageGpu();
            else
                img.SetGrayImage();
        }
        else{
            if(Config::slam != SlamType::kRaw)
                img.SetColorImageGpu();
            else
                img.SetColorImage();
        }

        static TicToc tt;
        torch::Tensor img_tensor = Pipeline::ImageToTensor(img.color0);
        //torch::Tensor img_tensor = Pipeline::ImageToTensor(img.color0_gpu);

        ///异步检测光流
        torch::Tensor flow;
        std::thread flow_thread([&flow](torch::Tensor &img){
            flow = flow_estimator->Forward(img);
            },std::ref(img_tensor));

        ///实例分割
        if(!Config::is_input_seg){
            if(Config::slam != SlamType::kRaw){
                tt.tic();
                inst_segmentor->ForwardTensor(img_tensor, img.mask_tensor, img.insts_info);
                InfoS("sync_process forward: {}", tt.toc_then_tic());
                if(Config::slam == SlamType::kNaive)
                    img.SetMaskGpuSimple();
                else if(Config::slam == SlamType::kDynamic)
                    img.SetMaskGpu();
                InfoS("sync_process SetMask: {}", tt.toc_then_tic());
            }
        }
        else{
            if(Config::dataset == DatasetType::kViode){
                if(Config::slam == SlamType::kNaive)
                    VIODE::SetViodeMaskSimple(img);
                else if(Config::slam == SlamType::kDynamic)
                    VIODE::SetViodeMask(img);
            }
            InfoS("sync_process SetMask: {}", tt.toc_then_tic());
        }

        flow_thread.join();

        img.flow = flow;
        inst_segmentor->PushBack(img);

        /*cv::Mat show;
        cv::cvtColor(img.merge_mask,show,CV_GRAY2BGR);
        cv::scaleAdd(img.color0,0.5,show,show);
        //cv::Mat show = VisualFlow(flow);
        cv::imshow("show",show);
        cv::waitKey(1);*/
    }

    WarnS("ImageProcess 线程退出");
}



void FeatureTrack()
{
    static TicToc tt;
    int cnt;
    while(Config::ok.load(std::memory_order_seq_cst)){
        if(auto img = inst_segmentor->WaitForResult();img){
            tt.tic();
            if(Config::slam == SlamType::kDynamic){
                feature_tracker->insts_tracker->set_vel_map(estimator->insts_manager.vel_map());
                FeatureMap features = feature_tracker->TrackSemanticImage(*img);
                auto instances= feature_tracker->insts_tracker->SetOutputFeature();
                estimator->PushBack(img->time0, features, instances);
            }
            else if(Config::slam == SlamType::kNaive){
                FeatureMap features = feature_tracker->TrackImageNaive(*img);
                estimator->PushBack(img->time0, features);
            }
            else{
                FeatureMap features = feature_tracker->TrackImage(*img);
                estimator->PushBack(img->time0, features);
            }

            ///发布跟踪可视化图像
            if (Config::kShowTrack){
                PubTrackImage(feature_tracker->img_track, img->time0);
                /*cv::imshow("img",feature_tracker->img_track);
                cv::waitKey(1);*/
                /*string label=to_string(img.time0)+".jpg";
                if(saved_name.count(label)!=0){
                    cv::imwrite(label,imgTrack);
                }*/
            }

            InfoT("**************feature_track:{} ms****************\n", tt.toc());
        }
    }
    WarnT("FeatureTrack 线程退出");
}






void ImuCallback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    Vector3d acc(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
    Vector3d gyr(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
    estimator->InputIMU(t, acc, gyr);
}

void RestartCallback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true){
        WarnV("restart the e!");
        estimator->ClearState();
        estimator->SetParameter();
    }
}

void TerminalCallback(const std_msgs::BoolConstPtr &terminal_msg)
{
    if (terminal_msg->data == true){
        cerr<<"terminal the e!"<<endl;
        ros::shutdown();
        Config::ok.store(false,std::memory_order_seq_cst);
    }
}

void ImuSwitchCallback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true){
        WarnV("use IMU!");
        estimator->ChangeSensorType(1, Config::STEREO);
    }
    else{
        WarnV("disable IMU!");
        estimator->ChangeSensorType(0, Config::STEREO);
    }
}

void CamSwitchCallback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true){
        WarnV("use stereo!");
        estimator->ChangeSensorType(Config::USE_IMU, 1);
    }
    else{
        WarnV("use mono camera (left)!");
        estimator->ChangeSensorType(Config::USE_IMU, 0);
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
        inst_segmentor.reset(new InstanceSegmentor);
        feature_tracker = std::make_unique<FeatureTracker>();
        flow_estimator = std::make_unique<FlowEstimator>();
    }
    catch(std::runtime_error &e){
        vio_logger->critical(e.what());
        cerr<<e.what()<<endl;
        return -1;
    }

    estimator->SetParameter();
    feature_tracker->ReadIntrinsicParameter(Config::kCamPath);


#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif

    cout<<"waiting for image and imu..."<<endl;
    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(Config::kImuTopic, 2000, ImuCallback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_img0 = n.subscribe(Config::kImage0Topic, 100, Img0Callback);
    ros::Subscriber sub_img1 = n.subscribe(Config::kImage1Topic, 100, Img1Callback);

    ros::Subscriber sub_seg0,sub_seg1;
    if(Config::is_input_seg){
        sub_seg0 = n.subscribe(Config::kImage0SegTopic, 100, Seg0Callback);
        sub_seg1 = n.subscribe(Config::kImage1SegTopic, 100, Seg1Callback);
    }

    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, RestartCallback);
    ros::Subscriber sub_terminal = n.subscribe("/vins_terminal", 100, TerminalCallback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, ImuSwitchCallback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, CamSwitchCallback);

    vio_logger->flush();

    std::thread vio_thread{&Estimator::ProcessMeasurements, estimator};
    std::thread sync_thread{ImageProcess};
    std::thread fk_thread{FeatureTrack};

    ros::spin();

    sync_thread.join();
    fk_thread.join();
    vio_thread.join();
    spdlog::drop_all();

    cerr<<"vins结束"<<endl;

    return 0;
}



