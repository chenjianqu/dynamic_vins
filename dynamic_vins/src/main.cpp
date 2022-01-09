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
#include <filesystem>

#include <spdlog/spdlog.h>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>

#include "parameters.h"
#include "estimator/estimator.h"
#include "utility/visualization.h"
#include "estimator/dynamic.h"
#include "featureTracker/segment_image.h"
#include "utility/viode_utils.h"
#include "featureTracker/feature_tracker.h"
#include "FlowEstimating/flow_estimator.h"
#include "FlowEstimating/flow_visual.h"
#include "utility/call_back.h"

namespace dynamic_vins{\

namespace fs=std::filesystem;


Estimator::Ptr estimator;
InstanceSegmentor::Ptr inst_segmentor;
FeatureTracker::Ptr feature_tracker;
CallBack* callback;


//摆烂了，直接读取离线估计的光流
cv::Mat ReadFlowTensor(double time){
    static vector<fs::path> names;
    constexpr char* dataset_path = "/home/chen/temp/flow0/";
    if(names.empty()){
        fs::path dir_path(dataset_path);
        if(!fs::exists(dir_path))
            return {};
        fs::directory_iterator dir_iter(dir_path);
        for(auto &it : dir_iter)
            names.emplace_back(it.path().filename());
        std::sort(names.begin(),names.end());
    }
    cv::Mat read_flow;
    string input_time = std::to_string(time);
    //二分查找
    int low=0,high=names.size()-1;
    while(low<=high){
        int mid=(low+high)/2;
        string name=names[mid];
        string name_time = name.substr(0,name.find_last_of('_'));
        double n_time=std::atof(name_time.c_str());
        if(input_time==name_time){
            string n_path = (dataset_path/names[mid]).string();
            read_flow = cv::optflow::readOpticalFlow(n_path);
            break;
        }
        else if(n_time>time){
            high = mid-1;
        }
        else{
            low = mid+1;
        }
    }
    return read_flow;
    /*if(read_flow.empty())
        return {};
    torch::Tensor tensor = torch::from_blob(read_flow.data, {read_flow.rows,read_flow.cols ,2}, torch::kFloat32).to(torch::kCUDA);
    tensor = tensor.permute({2,0,1});
    return tensor;*/
}




void ImageProcess()
{
    int cnt = 0;
    while(cfg::ok.load(std::memory_order_seq_cst))
    {
        if(inst_segmentor->GetQueueSize() >= kInferImageListSize){
            std::this_thread::sleep_for(50ms);
            continue;
        }
        Debugs("Start sync");
        SegImage img = callback->SyncProcess();
        Warns("----------Time : {} ----------", img.time0);

        ///rgb to gray
        if(img.gray0.empty()){
            if(cfg::slam != SlamType::kRaw)
                img.SetGrayImageGpu();
            else
                img.SetGrayImage();
        }
        else{
            if(cfg::slam != SlamType::kRaw)
                img.SetColorImageGpu();
            else
                img.SetColorImage();
        }

        static TicToc tt;
        torch::Tensor img_tensor = Pipeline::ImageToTensor(img.color0);
        //torch::Tensor img_clone = img_tensor.clone();
        //torch::Tensor img_tensor = Pipeline::ImageToTensor(img.color0_gpu);
        ///启动光流估计线程
        if(cfg::slam == SlamType::kDynamic){
            //feature_tracker->insts_tracker->StartFlowEstimating(img_tensor);
        }

        ///实例分割
        if(!cfg::is_input_seg){
            if(cfg::slam != SlamType::kRaw){
                tt.Tic();
                inst_segmentor->ForwardTensor(img_tensor, img.mask_tensor, img.insts_info);
                Infos("sync_process forward: {}", tt.TocThenTic());
                if(cfg::slam == SlamType::kNaive)
                    img.SetMaskGpuSimple();
                else if(cfg::slam == SlamType::kDynamic)
                    img.SetMaskGpu();
                Infos("sync_process SetMask: {}", tt.TocThenTic());
            }
        }
        else{
            tt.Tic();
            if(cfg::dataset == DatasetType::kViode){
                if(cfg::slam == SlamType::kNaive)
                    VIODE::SetViodeMaskSimple(img);
                else if(cfg::slam == SlamType::kDynamic)
                    VIODE::SetViodeMask(img);
            }
            Infos("sync_process SetMask: {}", tt.TocThenTic());
        }

        tt.Tic();

        ///等待光流估计结果
        //auto flow_tensor = feature_tracker->insts_tracker->WaitingFlowEstimating();
        auto flow_cv = ReadFlowTensor(img.time0);
        if(!flow_cv.empty()){
            img.flow = flow_cv;
        }
        else{
            Warns("Can not find :{}", std::to_string(img.time0));
            img.flow = cv::Mat(img.color0.size(),CV_32FC2,cv::Scalar_<float>(0,0));
        }

        Debugs("ReadFlowTensor:{}", tt.TocThenTic());

        inst_segmentor->PushBack(img);
        /*cv::Mat show;
        cv::cvtColor(img.merge_mask,show,CV_GRAY2BGR);
        cv::scaleAdd(img.color0,0.5,show,show);*/
        /*if(flow_tensor.defined()){
            cv::Mat show = VisualFlow(flow_tensor);
            cv::imshow("show",show);
            cv::waitKey(1);
        }*/
    }

    Warns("ImageProcess 线程退出");
}



void FeatureTrack()
{
    static TicToc tt;
    int cnt;
    while(cfg::ok.load(std::memory_order_seq_cst)){
        if(auto img = inst_segmentor->WaitForResult();img){
            tt.Tic();
            if(cfg::slam == SlamType::kDynamic){
                feature_tracker->insts_tracker->set_vel_map(estimator->insts_manager.vel_map());
                FeatureMap features = feature_tracker->TrackSemanticImage(*img);
                auto instances= feature_tracker->insts_tracker->SetOutputFeature();
                estimator->PushBack(img->time0, features, instances);
            }
            else if(cfg::slam == SlamType::kNaive){
                FeatureMap features = feature_tracker->TrackImageNaive(*img);
                estimator->PushBack(img->time0, features);
            }
            else{
                FeatureMap features = feature_tracker->TrackImage(*img);
                estimator->PushBack(img->time0, features);
            }

            ///发布跟踪可视化图像
            if (cfg::kShowTrack){
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
        estimator.reset(new Estimator());
        inst_segmentor.reset(new InstanceSegmentor);
        feature_tracker = std::make_unique<FeatureTracker>();
        callback = new CallBack();
    }
    catch(std::runtime_error &e){
        vio_logger->critical(e.what());
        cerr<<e.what()<<endl;
        return -1;
    }

    estimator->SetParameter();
    feature_tracker->ReadIntrinsicParameter(cfg::kCamPath);

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

    vio_logger->flush();

    std::thread sync_thread{ImageProcess};
    std::thread fk_thread{FeatureTrack};
    std::thread vio_thread{&Estimator::ProcessMeasurements, estimator};


    ros::spin();

    sync_thread.join();
    fk_thread.join();
    vio_thread.join();
    spdlog::drop_all();

    cerr<<"vins结束"<<endl;

}


}

int main(int argc, char **argv)
{
    if(argc != 2){
        cerr<<"please input: rosrun vins vins_node [cfg file]"<<endl;
        return 1;
    }

    return dynamic_vins::Run(argc,argv);
}



