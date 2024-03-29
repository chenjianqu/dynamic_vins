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

#include "basic/def.h"
#include "utils/parameters.h"
#include "utils/io/visualization.h"
#include "utils/io/publisher_map.h"
#include "utils/io/io_parameters.h"
#include "utils/io/feature_serialization.h"
#include "utils/dataset/viode_utils.h"
#include "utils/dataset/coco_utils.h"
#include "utils/io/dataloader.h"
#include "utils/io/image_viewer.h"
#include "utils/io/system_call_back.h"
#include "basic/semantic_image.h"
#include "basic/semantic_image_queue.h"
#include "basic/feature_queue.h"
#include "estimator/estimator.h"
#include "front_end/background_tracker.h"
#include "front_end/front_end_parameters.h"
#include "image_process/image_process.h"
#include "utils/io/output.h"

namespace dynamic_vins{\

namespace fs=std::filesystem;
using namespace std;

Estimator::Ptr estimator;
ImageProcessor::Ptr processor;
FeatureTracker::Ptr feature_tracker;
InstsFeatManager::Ptr insts_tracker;
SystemCallBack* sys_callback;
Dataloader::Ptr dataloader;

SemanticImageQueue image_queue;


/**
 * 前端线程,包括同步图像流,实例分割,光流估计,3D目标检测等
 */
void ImageProcess()
{
    int cnt = 0;
    double time_sum=0;
    TicToc t_all;
    ImageViewer viewer;

    while(cfg::ok==true){

        if(image_queue.size() >= kImageQueueSize){
            cerr<<"ImageProcess image_queue.size() >= kImageQueueSize,blocked"<<endl;
            std::this_thread::sleep_for(50ms);
            continue;
        }
        Debugs("Start sync");

        ///同步获取图像
        SemanticImage img;
        if(io_para::use_dataloader){
            img = dataloader->LoadStereo();
        }
        else{
            img = sys_callback->SyncProcess();
        }

        std::cout<<"image seq_id:"<<img.seq<<std::endl;
        Warns("----------Time : {} ----------", std::to_string(img.time0));
        t_all.Tic();

        ///结束程序
        if(img.color0.empty()){
            cfg::ok = false;
            ros::shutdown();
            break;
        }

        if(img.color0.rows!=cfg::kInputHeight || img.color0.cols!=cfg::kInputWidth){
            cerr<<fmt::format("The input image sizes is:{}x{},but config size is:{}x{}",
                              img.color0.rows,img.color0.cols,cfg::kInputHeight,cfg::kInputWidth)<<endl;
            std::terminate();
        }

        ///入口程序
        processor->Run(img);

//        ///3D目标检测结果保存
//        if(img.seq%10==0){
//            string save_name = cfg::kDatasetSequence+"_"+to_string(img.seq)+"_det2d.png";
//            cv::imwrite(save_name,img.merge_mask);
//
//            cv::Mat img_w=img.color0.clone();
//            for(auto &box3d:img.boxes3d){
//                box3d->VisCorners2d(img_w,BgrColor("white",false),*cam0);
//            }
//             save_name = cfg::kDatasetSequence+"_"+to_string(img.seq)+"_det3d.png";
//            cv::imwrite(save_name,img_w);
//        }

        double time_cost=t_all.Toc();
        time_sum+=time_cost;
        cnt++;

        Warns("ImageProcess, all:{} ms \n",time_cost);

//        ///测试，输出图像和Mask
//        if(img.seq==587){
//            cv::imwrite(io_para::kOutputFolder + cfg::kDatasetSequence + "/color0.png",img.color0);
//            cv::imwrite(io_para::kOutputFolder + cfg::kDatasetSequence + "/color1.png",img.color1);
//            cv::imwrite(io_para::kOutputFolder + cfg::kDatasetSequence + "/merge_mask.png",img.merge_mask);
//        }

        if(io_para::is_show_input){
            cv::Mat img_show;

//            ///实例分割结果可视化
//            if(!img.merge_mask.empty()){
//                cv::cvtColor(img.merge_mask,img_show,CV_GRAY2BGR);
//                ///将Mask设置蓝色的
//                vector<cv::Mat> mat_vec;
//                cv::split(img_show,mat_vec);
//                mat_vec[0] = mat_vec[0]*255;//蓝色
//                cv::merge(mat_vec,img_show);
//
//                cv::scaleAdd(img_show,0.8,img.color0,img_show);
//            }
//            else{
//                img_show = cv::Mat(cv::Size(img.color0.cols,img.color0.rows),CV_8UC3,cv::Scalar(255,255,255));
//            }
//            cv::resize(img_show,img_show,cv::Size(),0.4,0.4);


            ///3D目标检测结果可视化
            img_show = img.color0.clone();
            int id=0;
            for(auto &box:img.boxes3d){
                box->VisCorners2d(img_show, cv::Scalar(255, 255, 255),cam_s.cam0);
            }
            viewer.ImageShow(img_show,io_para::kImageDatasetPeriod,1);

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
    while(cfg::ok==true)
    {
        if(auto img = image_queue.request_image();img){
            tt.Tic();
            Warnt("----------Time : {} ----------", std::to_string(img->time0));

            FrontendFeature frame;
            frame.time = img->time0;
            frame.seq_id = img->seq;

            ///前端跟踪
            if(cfg::slam == SLAM::kDynamic){
                insts_tracker->SetEstimatedInstancesInfo(estimator->im.GetOutputInstInfo());
                TicToc t_i;

                for(auto &[inst_id,inst]: insts_tracker->instances){
                    inst.is_curr_visible=false;
                    inst.box2d.reset();
                    inst.box3d.reset();
                }

                ///MOT
                if(cfg::dataset == DatasetType::kKitti || cfg::dataset==DatasetType::kCustom){
                    insts_tracker->AddInstancesByTracking(*img);
                }
                else if(cfg::dataset == DatasetType::kViode){
                    insts_tracker->AddViodeInstances(*img);
                }
                else{
                    throw std::runtime_error("FeatureTrack()::MOT not is implemented, as dataset is "+cfg::dataset_name);
                }

                Debugt("FeatureTrack() MOT {} ms",t_i.TocThenTic());

                ///将静态物体的mask去掉
                int static_inst_cnt = 0;
                insts_tracker->ExecInst([&](unsigned int key, InstFeat& inst){
                    if(inst.roi && inst.box2d &&
                    insts_tracker->estimated_info.find(key)!=insts_tracker->estimated_info.end() &&
                    insts_tracker->estimated_info[key].is_static){
                        static_inst_cnt++;
                        int roi_rows = inst.roi->mask_cv.rows;
                        int roi_cols = inst.roi->mask_cv.cols;
                        int offset_row = inst.box2d->rect.tl().y;
                        int offset_col = inst.box2d->rect.tl().x;
                        Debugt("roi_rows:{} roi_cols:{} offset_row:{} offset_col:{}",roi_rows,roi_cols,offset_row,offset_col);
                        for(int row=0;row<roi_rows;++row){
                            for(int col=0;col<roi_cols;++col){
                                if(inst.roi->mask_cv.at<uchar>(row,col)>=0.5){
                                    img->merge_mask.at<uchar>(row + offset_row, col + offset_col) = 0;
                                }
                            }
                        }
                    }
                });
                if(static_inst_cnt>0){
                    img->merge_mask_gpu.upload(img->merge_mask);
                    cv::cuda::bitwise_not(img->merge_mask_gpu,img->inv_merge_mask_gpu);
                    img->inv_merge_mask_gpu.download(img->inv_merge_mask);
                }

                Debugt("FeatureTrack() SetInstMask {} ms",t_i.TocThenTic());

                ///开启一个线程检测动态特征点
                std::thread t_inst_track = std::thread(&InstsFeatManager::InstsTrack, insts_tracker.get(), *img);

                ///执行背景区域跟踪
                frame.features  = feature_tracker->TrackSemanticImage(*img);

                t_inst_track.join();

                frame.instances = insts_tracker->Output();
                Infot("TrackSemanticImage 动态检测线程总时间:{} ms", t_i.TocThenTic());

                if(fe_para::is_show_track){
                    insts_tracker->DrawInsts(feature_tracker->img_track());
                }

                ///保存多目标跟踪结果
                if(cfg::dst_mode){
                    if(cfg::slam==SLAM::kDynamic && cfg::dataset == DatasetType::kKitti){
                        SaveMotTrajectory(insts_tracker->instances,img->seq);
                    }
                }

//                if(img->seq%10==0){
//                    cv::Mat img_w=img->color0.clone();
//                    string save_name = cfg::kDatasetSequence+"_"+std::to_string(img->seq)+"_inst.png";
//                    insts_tracker->DrawInsts(img_w);
//                    cv::imwrite(save_name,img_w);
//                }
            }
            else if(cfg::slam == SLAM::kNaive){
                frame.features = feature_tracker->TrackImageNaive(*img);
            }
            else{
                if(cfg::use_line){
                    frame.features = feature_tracker->TrackImageLine(*img);
                }
                else{
                    frame.features = feature_tracker->TrackImage(*img);
                }
            }

            ///TODO DEBUG
            //string serialize_path = cfg::kBasicDir + "/data/output/serialization/";
            //serialize_path += fmt::format("{}_point.txt",frame.seq_id);
            //SerializePointFeature(serialize_path,frame.features.points);//序列化
            //frame.features.points = DeserializePointFeature(serialize_path);//反序列化

            //serialize_path += fmt::format("{}_line.txt",frame.seq_id);
            //SerializeLineFeature(serialize_path,frame.features.lines);//序列化
            //frame.features.lines = DeserializeLineFeature(serialize_path);//反序列化


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
                PublisherMap::PubImage(feature_tracker->img_track(),"image_track");
                //cv::imshow("img",feature_tracker->img_track);
                //cv::waitKey(1);
                //string label=to_string(img.time0)+".jpg";
                //if(saved_name.count(label)!=0){
                //    cv::imwrite(label,imgTrack);
                //}
            }
            Infot("**************feature_track:{} ms****************\n", time_cost);
        }
    }
    Infot("Feature Tracking Avg cost:{} ms",time_sum/cnt);

    Warnt("FeatureTrack 线程退出");
}



int Run(int argc, char **argv){
    ros::init(argc, argv, "dynamic_vins");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    string file_name = argv[1];
    cout<<fmt::format("cfg_file:{}",file_name)<<endl;

    string seq_name = argv[2];
    cout<<fmt::format("seq_name:{}",seq_name)<<endl;

    try{
        cfg cfg(file_name,seq_name);
        ///初始化logger
        MyLogger::InitLogger(file_name);
        ///初始化相机模型
        cout<<"start init camera"<<endl;
        InitCamera(file_name,seq_name);
        ///初始化局部参数
        cout<<"start init dataset parameters"<<endl;
        coco::SetParameters(file_name);
        if(cfg::dataset == DatasetType::kViode){
            VIODE::SetParameters(file_name);
        }
        cout<<"start init io"<<endl;
        io_para::SetParameters(file_name,seq_name);

        cout<<"start init three threads"<<endl;

        estimator.reset(new Estimator(file_name));
        processor.reset(new ImageProcessor(file_name,seq_name));

        feature_tracker = std::make_unique<FeatureTracker>(file_name);
        if(cfg::slam==SLAM::kDynamic){
            insts_tracker.reset(new InstsFeatManager(file_name));
        }
    }
    catch(std::runtime_error &e){
        cerr<<e.what()<<endl;
        return -1;
    }

    estimator->SetParameter();
    cout<<"init completed"<<endl;

    if(io_para::use_dataloader){
        dataloader = std::make_shared<Dataloader>(io_para::kImageDatasetLeft,io_para::kImageDatasetRight);
    }
    else{
        ///订阅消息
        sys_callback = new SystemCallBack(estimator,n);
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

    delete sys_callback;

    cout<<"vins结束"<<endl;

    return 0;
}


}

int main(int argc, char **argv)
{
    if(argc != 3){
        std::cerr<<"please input: rosrun dynamic_vins dynamic_vins [cfg file] [seq_name]"<< std::endl;
        return 1;
    }

    return dynamic_vins::Run(argc,argv);
}



