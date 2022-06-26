/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <memory>
#include <thread>
#include <mutex>
#include <chrono>
#include <future>
#include <iostream>

#include <spdlog/spdlog.h>
#include <ros/ros.h>

#include "utils/def.h"
#include "utils/parameters.h"
#include "utils/io/visualization.h"
#include "utils/io/io_parameters.h"
#include "utils/dataset/viode_utils.h"
#include "utils/dataset/coco_utils.h"
#include "utils/io/dataloader.h"
#include "estimator/estimator.h"
#include "front_end/semantic_image.h"
#include "front_end/background_tracker.h"
#include "front_end/front_end_parameters.h"
#include "image_process/image_process.h"

namespace dynamic_vins{\

namespace fs=std::filesystem;
using namespace std;



Estimator::Ptr estimator;
ImageProcessor::Ptr processor;
FeatureTracker::Ptr feature_tracker;
InstsFeatManager::Ptr insts_tracker;
CallBack* callback;
Dataloader::Ptr dataloader;

ImageQueue image_queue;


/**
 * 前端线程,包括同步图像流,实例分割,光流估计,3D目标检测等
 */
void ImageProcess()
{
    int cnt = 0;
    double time_sum=0;
    TicToc t_all;
    ImageViewer viewer;

    while(cfg::ok.load(std::memory_order_seq_cst))
    {
        if(image_queue.size() >= kImageQueueSize){
            cerr<<"ImageProcess image_queue.size() >= kImageQueueSize,blocked"<<endl;
            std::this_thread::sleep_for(50ms);
            continue;
        }
        ///同步获取图像
        Debugs("Start sync");

        SemanticImage img;
        if(io_para::use_dataloader){
            img = dataloader->LoadStereo();
        }
        else{
            img = callback->SyncProcess();
        }

        std::cout<<"image seq_id:"<<img.seq<<std::endl;

        ///结束程序
        if(img.color0.empty()){
            cfg::ok = false;
            break;
        }

        Warns("----------Time : {} ----------", std::to_string(img.time0));
        t_all.Tic();

        if(img.color0.rows!=cfg::kInputHeight || img.color0.cols!=cfg::kInputWidth){
            cerr<<fmt::format("The input image sizes is:{}x{},but config size is:{}x{}",
                              img.color0.rows,img.color0.cols,cfg::kInputHeight,cfg::kInputWidth)<<endl;
            std::terminate();
        }

        processor->Run(img);

/*
        //可视化
        if(img.seq%10==0){
            string save_name = cfg::kDatasetSequence+"_"+to_string(img.seq)+"_det2d.png";
            cv::imwrite(save_name,img.merge_mask);

            cv::Mat img_w=img.color0.clone();
            for(auto &box3d:img.boxes3d){
                box3d->VisCorners2d(img_w,BgrColor("white",false),*cam0);
            }
             save_name = cfg::kDatasetSequence+"_"+to_string(img.seq)+"_det3d.png";
            cv::imwrite(save_name,img_w);
        }*/

        double time_cost=t_all.Toc();
        time_sum+=time_cost;
        cnt++;

        Warns("ImageProcess, all:{} ms \n",time_cost);


        if(io_para::is_show_input){
            cv::Mat img_show;
            if(!img.inv_merge_mask.empty()){
                cv::cvtColor(img.inv_merge_mask,img_show,CV_GRAY2BGR);
                cv::scaleAdd(img.color0,0.5,img_show,img_show);
            }
            else{
                img_show = cv::Mat(cv::Size(img.color0.cols,img.color0.rows),CV_8UC3,cv::Scalar(255,255,255));
            }
            cv::resize(img_show,img_show,cv::Size(),0.5,0.5);

            /*if(flow_tensor.defined()){
               cv::Mat show = VisualFlow(flow_tensor);
               cv::imshow("show",show);
               cv::waitKey(1);
            }*/

            viewer.ImageShow(img_show,io_para::kImageDatasetPeriod);
        }
        else{
            viewer.Delay(io_para::kImageDatasetPeriod);
        }

        ///将结果存放到消息队列中
        if(!cfg::is_only_imgprocess){
            image_queue.push_back(img);
        }



    }

    Infos("Image Process Avg cost:{} ms",time_sum/cnt);

    Warns("ImageProcess 线程退出");
}


/**
 * 特征跟踪线程
 */
void FeatureTrack()
{
    TicToc tt;
    int cnt=0;
    double time_sum=0;
    while(cfg::ok.load(std::memory_order_seq_cst))
    {
        if(auto img = image_queue.request_image();img){
            tt.Tic();
            Warnt("----------Time : {} ----------", std::to_string(img->time0));
            SemanticFeature frame;
            frame.time = img->time0;
            frame.seq_id = img->seq;

            ///前端跟踪
            if(cfg::slam == SlamType::kDynamic){
                insts_tracker->SetEstimatedInstancesInfo(estimator->im.GetOutputInstInfo());
                TicToc t_i;
                //开启另一个线程检测动态特征点
                std::thread t_inst_track = std::thread(&InstsFeatManager::InstsTrack, insts_tracker.get(), *img);

                frame.features  = feature_tracker->TrackSemanticImage(*img);

                t_inst_track.join();
                frame.instances = insts_tracker->Output();

                Infot("TrackSemanticImage 动态检测线程总时间:{} ms", t_i.TocThenTic());

                if(fe_para::is_show_track){
                    insts_tracker->DrawInsts(feature_tracker->img_track());
                }

                /*if(img->seq%10==0){
                    cv::Mat img_w=img->color0.clone();
                    string save_name = cfg::kDatasetSequence+"_"+std::to_string(img->seq)+"_inst.png";
                    insts_tracker->DrawInsts(img_w);

                    cv::imwrite(save_name,img_w);
                }*/

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
                    feature_queue.push_back(frame);
                }
                else{
                    if(cnt%2==0){ //对于其它数据集,控制输入数据到后端的频率
                        feature_queue.push_back(frame);
                    }
                }
            }

            double time_cost=tt.Toc();
            time_sum += time_cost;
            cnt++;


            ///发布跟踪可视化图像
            if (fe_para::is_show_track){
                ImagePublisher::Pub(feature_tracker->img_track(),"image_track");
                /*cv::imshow("img",feature_tracker->img_track);
                cv::waitKey(1);*/
                /*string label=to_string(img.time0)+".jpg";
                if(saved_name.count(label)!=0){
                    cv::imwrite(label,imgTrack);
                }*/
            }
            Infot("**************feature_track:{} ms****************\n", time_cost);
        }
    }
    Infot("Feature Tracking Avg cost:{} ms",time_sum/cnt);

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

    string file_name = argv[1];
    cout<<fmt::format("cfg_file:{}",argv[1])<<endl;

    try{
        cfg cfg(file_name);
        ///初始化logger
        MyLogger::InitLogger(file_name);
        ///初始化相机模型
        InitCamera(file_name);
        ///初始化局部参数
        coco::SetParameters(file_name);
        if(cfg::dataset == DatasetType::kViode){
            VIODE::SetParameters(file_name);
        }
        io_para::SetParameters(file_name);

        estimator.reset(new Estimator(file_name));
        processor.reset(new ImageProcessor(file_name));

        feature_tracker = std::make_unique<FeatureTracker>(file_name);
        insts_tracker.reset(new InstsFeatManager(file_name));
    }
    catch(std::runtime_error &e){
        cerr<<e.what()<<endl;
        return -1;
    }

    estimator->SetParameter();

    ros::Subscriber sub_imu,sub_img0,sub_img1;
    ros::Subscriber sub_seg0,sub_seg1;
    ros::Subscriber sub_restart,sub_terminal,sub_imu_switch,sub_cam_switch;

    if(io_para::use_dataloader){
        dataloader = std::make_shared<Dataloader>();
    }
    else{
        callback = new CallBack();

        sub_imu = n.subscribe(io_para::kImuTopic, 2000, ImuCallback,
                                              ros::TransportHints().tcpNoDelay());
        sub_img0 = n.subscribe(io_para::kImage0Topic, 100, &CallBack::Img0Callback,callback);
        sub_img1 = n.subscribe(io_para::kImage1Topic, 100, &CallBack::Img1Callback,callback);

        if(cfg::is_input_seg){
            sub_seg0 = n.subscribe(io_para::kImage0SegTopic, 100, &CallBack::Seg0Callback,callback);
            sub_seg1 = n.subscribe(io_para::kImage1SegTopic, 100, &CallBack::Seg1Callback,callback);
        }

        sub_restart = n.subscribe("/vins_restart", 100, RestartCallback);
        sub_terminal = n.subscribe("/vins_terminal", 100, TerminalCallback);
        sub_imu_switch = n.subscribe("/vins_imu_switch", 100, ImuSwitchCallback);
        sub_cam_switch = n.subscribe("/vins_cam_switch", 100, CamSwitchCallback);
    }

    Publisher::e = estimator;
    Publisher::RegisterPub(n);

    MyLogger::vio_logger->flush();

    cout<<"waiting for image and imu..."<<endl;

    std::thread sync_thread{ImageProcess};
    std::thread fk_thread;
    std::thread vio_thread;

    if(!cfg::is_only_imgprocess){
        fk_thread = std::thread(FeatureTrack);

        if(!cfg::is_only_frontend){
            vio_thread = std::thread(&Estimator::ProcessMeasurements, estimator);
        }
    }

    ros::spin();

    cfg::ok= false;
    cv::destroyAllWindows();

    sync_thread.join();
    fk_thread.join();
    vio_thread.join();
    spdlog::drop_all();
    delete callback;

    cout<<"vins结束"<<endl;

    return 0;
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



