/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <memory>
#include <thread>
#include <mutex>
#include <chrono>
#include <future>
#include <filesystem>
#include <iostream>
#include <optional>

#include <spdlog/spdlog.h>
#include <ros/ros.h>

#include "utils/def.h"
#include "utils/parameters.h"
#include "utils/visualization.h"
#include "utils/dataset/viode_utils.h"
#include "utils/call_back.h"
#include "flow/flow_estimator.h"
#include "flow/flow_visual.h"
#include "det3d/detector3d.h"
#include "det2d/detector2d.h"
#include "estimator/estimator.h"
#include "estimator/dynamic.h"
#include "front_end/segment_image.h"
#include "front_end/front_end.h"
#include "front_end/front_end_parameters.h"

namespace dynamic_vins{\

namespace fs=std::filesystem;
using namespace std;

constexpr int kInferImageListSize=30;


Estimator::Ptr estimator;
Detector2D::Ptr detector2d;
Detector3D::Ptr detector3d;
FlowEstimator::Ptr flow_estimator;
FeatureTracker::Ptr feature_tracker;
CallBack* callback;


class ImageQueue{
public:
    void PushBack(SegImage& img){
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if(seg_img_list_.size() < kInferImageListSize){
            seg_img_list_.push_back(img);
        }
        queue_cond_.notify_one();
    }

    int GetQueueSize(){
        std::unique_lock<std::mutex> lock(queue_mutex_);
        return (int)seg_img_list_.size();
    }

    std::optional<SegImage> WaitForResult() {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if(!queue_cond_.wait_for(lock, 30ms, [&]{return !seg_img_list_.empty();}))
            return std::nullopt;
        //queue_cond_.wait(lock,[&]{return !seg_img_list_.empty();});
        SegImage frame=std::move(seg_img_list_.front());
        seg_img_list_.pop_front();
        return frame;
    }

    std::mutex queue_mutex_;
    std::condition_variable queue_cond_;
    std::list<SegImage> seg_img_list_;
};

ImageQueue image_queue;


/**
 * 前端线程,包括同步图像流,实例分割,光流估计,3D目标检测等
 */
void ImageProcess()
{
    int cnt = 0;
    TicToc tt, t_all;

    while(cfg::ok.load(std::memory_order_seq_cst))
    {
        if(image_queue.GetQueueSize() >= kInferImageListSize){
            std::this_thread::sleep_for(50ms);
            continue;
        }
        ///同步获取图像
        Debugs("Start sync");
        SegImage img = callback->SyncProcess();
        Warns("----------Time : {} ----------", std::to_string(img.time0));
        t_all.Tic();

        if(img.color0.rows!=cfg::kInputHeight || img.color0.cols!=cfg::kInputWidth){
            cerr<<fmt::format("The input image sizes is:{}x{},but config size is:{}x{}",
                              img.color0.rows,img.color0.cols,cfg::kInputHeight,cfg::kInputWidth)<<endl;
            std::terminate();
        }

        ///rgb to gray
        tt.Tic();
        if(img.gray0.empty()){
            img.SetGrayImageGpu();
        }
        else{
            img.SetColorImageGpu();
        }

        ///均衡化
        //static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        //clahe->apply(img.gray0, img.gray0);
        //if(!img.gray1.empty())
        //    clahe->apply(img.gray1, img.gray1);

        torch::Tensor img_tensor = Pipeline::ImageToTensor(img.color0);
        //torch::Tensor img_clone = img_tensor.clone();
        //torch::Tensor img_tensor = Pipeline::ImageToTensor(img.color0_gpu);

        ///启动光流估计线程
        if(cfg::slam != SlamType::kRaw && cfg::use_dense_flow ){
            if(cfg::use_preprocess_flow){
                flow_estimator->SynchronizeReadFlow(img.seq);
            }
            else{
                flow_estimator->SynchronizeForward(img_tensor);
            }
        }
        Infos("ImageProcess prepare: {}", tt.TocThenTic());

        ///实例分割
        tt.Tic();
        if(cfg::slam != SlamType::kRaw){
            if(!cfg::is_input_seg){
                detector2d->ForwardTensor(img_tensor, img.mask_tensor, img.insts_info);
                if(cfg::slam == SlamType::kNaive)
                    img.SetBackgroundMask();
                else if(cfg::slam == SlamType::kDynamic)
                    img.SetMask();
            }
            else{
                if(cfg::dataset == DatasetType::kViode){
                    if(cfg::slam == SlamType::kNaive)
                        VIODE::SetViodeMaskSimple(img);
                    else if(cfg::slam == SlamType::kDynamic)
                        VIODE::SetViodeMask(img);
                }
            }
            Infos("ImageProcess SetMask: {}", tt.TocThenTic());

            ///读取离线检测的3D包围框
            img.boxes = detector3d->ReadBox3D(img.seq);
        }

        //log
        //for(auto &inst : img.insts_info)
        //    Debugs("img.insts_info id:{} min_pt:({},{}),max_pt:({},{})",inst.id,inst.min_pt.x,inst.min_pt.y,inst.max_pt.x,inst.max_pt.y);

        ///获得光流估计
        if(cfg::use_dense_flow){
            cv::Mat flow_cv;
            if(cfg::use_preprocess_flow){
                flow_cv = flow_estimator->WaitingReadFlowImage();
            }
            else{
                auto flow_tensor = flow_estimator->WaitingForwardResult();
                flow_tensor = flow_tensor.to(torch::kCPU);
                img.flow = cv::Mat(flow_tensor.sizes()[1],flow_tensor.sizes()[2],CV_8UC2,flow_tensor.data_ptr()).clone();
            }

            if(!flow_cv.empty()){
                img.flow = flow_cv;
            }
            else{
                img.flow = cv::Mat(img.color0.size(),CV_32FC2,cv::Scalar_<float>(0,0));
            }
        }


        image_queue.PushBack(img);
        /*cv::Mat show;
        cv::cvtColor(img.inv_merge_mask,show,CV_GRAY2BGR);
        cv::scaleAdd(img.color0,0.5,show,show);
        cv::imshow("show",show);
        cv::waitKey(1);*/
        /*if(flow_tensor.defined()){
            cv::Mat show = VisualFlow(flow_tensor);
            cv::imshow("show",show);
            cv::waitKey(1);
        }*/

        Warns("ImageProcess, all:{} ms \n",t_all.Toc());
    }

    Warns("ImageProcess 线程退出");
}


/**
 * 特征跟踪线程
 */
void FeatureTrack()
{
    TicToc tt;
    int cnt;
    while(cfg::ok.load(std::memory_order_seq_cst))
    {
        if(auto img = image_queue.WaitForResult();img){
            tt.Tic();
            Warnt("----------Time : {} ----------", std::to_string(img->time0));
            FeatureFrame frame;
            frame.time = img->time0;

            ///前端跟踪
            if(cfg::slam == SlamType::kDynamic){
                feature_tracker->insts_tracker->set_vel_map(estimator->insts_manager.vel_map());
                frame.features  = feature_tracker->TrackSemanticImage(*img);
                frame.instances = feature_tracker->insts_tracker->GetOutputFeature();
            }
            else if(cfg::slam == SlamType::kNaive){
                frame.features = feature_tracker->TrackImageNaive(*img);
            }
            else{
                frame.features = feature_tracker->TrackImage(*img);
            }
            ///将数据传入到后端
            if(!cfg::is_only_frontend){
                if(cfg::dataset == DatasetType::kKitti){
                    estimator->PushBack(frame);
                }
                else{
                    if((cnt++)%2==0){ //对于其它数据集,控制输入数据到后端的频率
                        estimator->PushBack(frame);
                    }
                }
            }

            ///发布跟踪可视化图像
            if (fe_para::is_show_track){
                PubTrackImage(feature_tracker->img_track(), img->time0);
                /*cv::imshow("img",feature_tracker->img_track);
                cv::waitKey(1);*/
                /*string label=to_string(img.time0)+".jpg";
                if(saved_name.count(label)!=0){
                    cv::imwrite(label,imgTrack);
                }*/
            }
            Infot("**************feature_track:{} ms****************\n", tt.Toc());
        }
    }
    Warnt("FeatureTrack 线程退出");
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
        Warnv("restart the e!");
        estimator->ClearState();
        estimator->SetParameter();
    }
}

void TerminalCallback(const std_msgs::BoolConstPtr &terminal_msg)
{
    if (terminal_msg->data == true){
        cerr<<"terminal the e!"<<endl;
        ros::shutdown();
        cfg::ok.store(false,std::memory_order_seq_cst);
    }
}

void ImuSwitchCallback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true){
        Warnv("use IMU!");
        estimator->ChangeSensorType(1, cfg::is_stereo);
    }
    else{
        Warnv("disable IMU!");
        estimator->ChangeSensorType(0, cfg::is_stereo);
    }
}

void CamSwitchCallback(const std_msgs::BoolConstPtr &switch_msg)
{
    if (switch_msg->data == true){
        Warnv("use stereo!");
        estimator->ChangeSensorType(cfg::is_use_imu, 1);
    }
    else{
        Warnv("use mono camera (left)!");
        estimator->ChangeSensorType(cfg::is_use_imu, 0);
    }
}


int Run(int argc, char **argv){
    ros::init(argc, argv, "dynamic_vins");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    string cfg_file = argv[1];
    cout<<fmt::format("cfg_file:{}",argv[1])<<endl;

    try{
        cfg cfg(cfg_file);
        estimator.reset(new Estimator(cfg_file));
        detector2d.reset(new Detector2D(cfg_file));
        detector3d.reset(new Detector3D(cfg_file));
        feature_tracker = std::make_unique<FeatureTracker>(cfg_file);
        callback = new CallBack();
        flow_estimator = std::make_unique<FlowEstimator>(cfg_file);
    }
    catch(std::runtime_error &e){
        cerr<<e.what()<<endl;
        return -1;
    }

    estimator->SetParameter();

    cout<<"waiting for image and imu..."<<endl;
    registerPub(n);

    ros::Subscriber sub_imu = n.subscribe(cfg::kImuTopic, 2000, ImuCallback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber sub_img0 = n.subscribe(cfg::kImage0Topic, 100, &CallBack::Img0Callback,callback);
    ros::Subscriber sub_img1 = n.subscribe(cfg::kImage1Topic, 100, &CallBack::Img1Callback,callback);

    ros::Subscriber sub_seg0,sub_seg1;
    if(cfg::is_input_seg){
        sub_seg0 = n.subscribe(cfg::kImage0SegTopic, 100, &CallBack::Seg0Callback,callback);
        sub_seg1 = n.subscribe(cfg::kImage1SegTopic, 100, &CallBack::Seg1Callback,callback);
    }

    ros::Subscriber sub_restart = n.subscribe("/vins_restart", 100, RestartCallback);
    ros::Subscriber sub_terminal = n.subscribe("/vins_terminal", 100, TerminalCallback);
    ros::Subscriber sub_imu_switch = n.subscribe("/vins_imu_switch", 100, ImuSwitchCallback);
    ros::Subscriber sub_cam_switch = n.subscribe("/vins_cam_switch", 100, CamSwitchCallback);

    MyLogger::vio_logger->flush();

    std::thread sync_thread{ImageProcess};
    std::thread fk_thread{FeatureTrack};
    std::thread vio_thread;
    if(!cfg::is_only_frontend)
        vio_thread = std::thread(&Estimator::ProcessMeasurements, estimator);

    ros::spin();

    cfg::ok= false;

    sync_thread.join();
    fk_thread.join();
    vio_thread.join();
    spdlog::drop_all();
    delete callback;

    cout<<"vins结束"<<endl;
}


}

int main(int argc, char **argv)
{
    if(argc != 2){
        std::cerr<<"please input: rosrun vins vins_node [cfg file]"<< std::endl;
        return 1;
    }

    return dynamic_vins::Run(argc,argv);
}



