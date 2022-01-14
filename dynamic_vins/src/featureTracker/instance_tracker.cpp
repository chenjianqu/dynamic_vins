/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "instance_tracker.h"
#include "segment_image.h"
#include "utils.h"
#include "utility/viode_utils.h"
#include "FlowEstimating/flow_visual.h"

namespace dynamic_vins{\

using std::unordered_map;
using std::make_pair;
using std::map;
namespace idx = torch::indexing;


std::default_random_engine randomEngine;
std::uniform_int_distribution<unsigned int> color_rd(0,255);

InstFeat::InstFeat():color(color_rd(randomEngine),color_rd(randomEngine),color_rd(randomEngine))
{}

InstFeat::InstFeat(unsigned int id_, int class_id_): id(id_), class_id(class_id_),
color(color_rd(randomEngine),color_rd(randomEngine),color_rd(randomEngine))
{}


InstsFeatManager::InstsFeatManager()
{
    lk_optical_flow = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(21, 21), 3, 30);
    lk_optical_flow_back = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(21, 21),1,30,true);
    std::array<int64_t, 2> orig_dim{int64_t(cfg::kRow), int64_t(cfg::kCol)};
    mot_tracker = std::make_unique<DeepSORT>(orig_dim);
    //flow_estimator_ = std::make_unique<FlowEstimator>();

    orb_matcher_ = cv::DescriptorMatcher::create("BruteForce-Hamming");
}


void InstsFeatManager::InstsTrack(SegImage img)
{
    TicToc tic_toc;
    curr_time=img.time0;

    if(cfg::dataset == DatasetType::kKitti){
        AddInstancesByTracking(img);
    }
    else if(cfg::dataset == DatasetType::kViode){
        AddViodeInstances(img);
    }
    else{
        throw std::runtime_error("have not this dataset type");
    }
    exist_inst_ = !img.insts_info.empty();
    Infot("instsTrack AddInstances:{} ms", tic_toc.TocThenTic());

    if(exist_inst_){
        ///形态学运算
        //ErodeMaskGpu(img.merge_mask_gpu, img.merge_mask_gpu);
        img.merge_mask_gpu.download(img.merge_mask);
        for(auto& [key,inst] : instances_){
            if(inst.last_frame_cnt < global_frame_id)
                inst.lost_num++;
            else if(inst.last_frame_cnt == global_frame_id)
                inst.lost_num=0;
        }
        Infot("instsTrack set mask_img:{} ms", tic_toc.TocThenTic());
        //对每个目标进行光流跟踪
        for(auto & [ key,inst] : instances_){
            if(inst.last_points.empty() || inst.lost_num>0)
                continue;
            inst.curr_points.clear();
            Debugt("inst:{} last_points:{} mask({}x{},type:{})",
                   inst.id, inst.last_points.size(), inst.mask_img.rows, inst.mask_img.cols, inst.mask_img.type());
            ///光流跟踪
            auto status = FeatureTrackByLK(prev_img.gray0,img.gray0,inst.last_points,inst.curr_points);
            /*auto status = FeatureTrackByLKGpu(lk_optical_flow, lk_optical_flow_back, prev_img.gray0_gpu,
                                              img.gray0_gpu, inst.last_points, inst.curr_points);*/
            /*ImageTranslate(img.gray0,detect_img,row_shift,col_shift);
            vector<uchar> status = flowTrack(last_img.gray0,detect_img,inst.last_points,inst.curr_points);
            for(auto& pt:inst.curr_points){
                pt.y -= row_shift;
                pt.x -= col_shift;
            }*/
            if(cfg::dataset == DatasetType::kViode){
                for(size_t i=0;i<status.size();++i){
                    //if(status[i] && kViode::PixelToKey(inst.curr_points[i],img.seg0)!=key) status[i]=0;
                    if(status[i] && inst.mask_img.at<uchar>(inst.curr_points[i]) == 0)
                        status[i]=0;
                }
            }
            else{
                for(size_t i=0;i<status.size();++i)
                    if(status[i] && inst.mask_img.at<uchar>(inst.curr_points[i]) == 0)
                        status[i]=0;
            }
            ReduceVector(inst.curr_points, status);
            ReduceVector(inst.ids, status);
            ReduceVector(inst.last_points, status);
        }
        Infot("instsTrack flowTrack:{} ms", tic_toc.TocThenTic());

        ///对每个特征点增加观测次数
        for(auto & [ key,inst] : instances_){
            if(inst.last_points.empty() || inst.lost_num>0)
                continue;
            for (auto &n : inst.track_cnt) n++;
        }
        /*
         ///RANSANC剔除点
         if constexpr (false){
            if(!instances.empty()){
                for(auto& pair: instances)
                {
                    auto &inst=pair.second;
                    if(inst.curr_points.size()<8 || inst.last_points.size()<8)
                        continue;
                    auto status=RejectWithF(inst,img.gray0.cols,img.gray0.rows);
                    reduceVector(inst.last_points, status);
                    reduceVector(inst.curr_points, status);
                    ReduceVector(inst.ids, status);
                }
            }
        }
        if constexpr (false){
            for(auto & [key,inst] : instances){
                inst.visual_new_points.clear();
                if(inst.lost_num>0 || inst.curr_points.size()>INST_FEAT_NUM)
                    continue;
                vector<cv::Point2f> new_pts;
                size_t new_num=INST_FEAT_NUM - inst.curr_points.size();
                cv::goodFeaturesToTrack(img.gray0, new_pts, (int)new_num, 0.01, DYNAMIC_MIN_DIST, inst.mask_img); //检测新的角点
                for(int i=0;i<std::min(new_pts.size(),new_num);++i){
                    inst.curr_points.emplace_back(new_pts[i]);
                    inst.ids.emplace_back(global_id_count++);
                    inst.visual_new_points.emplace_back(new_pts[i]);
                }
                //printf("cur_pts:%d new_pts:%d\n",inst.curr_points.size(),new_pts.size());
            }
        }*/
        if(exist_inst_){
            ///添加新的特征点前的准备
            int max_new_detect=0;
            for(auto & [key,inst] : instances_){
                if(inst.lost_num>0 || inst.curr_points.size()>= cfg::kMaxDynamicCnt)
                    continue;
                max_new_detect += (cfg::kMaxDynamicCnt - (int)inst.curr_points.size());
            }
            ///添加新的特征点
            if(max_new_detect > 0){
                mask_background = img.merge_mask;
                for(auto & [key,inst] : instances_){
                    if(inst.lost_num>0 || inst.curr_points.size()>=cfg::kMaxDynamicCnt)
                        continue;
                    inst.visual_points_pair.clear();
                    inst.visual_right_points_pair.clear();
                    inst.visual_new_points.clear();
                    for(size_t i=0;i<inst.curr_points.size();++i){
                        inst.visual_points_pair.emplace_back(inst.last_points[i],inst.curr_points[i]);//用于可视化
                        cv::circle(mask_background, inst.curr_points[i], cfg::kMinDynamicDist, 0, -1);//设置mask
                    }
                }
                Infot("instsTrack prepare detect:{} ms", tic_toc.TocThenTic());
                /*cv::goodFeaturesToTrack(img.gray0, new_pts, (int)new_detect, 0.01, DYNAMIC_MIN_DIST, mask_bg_new); //检测新的角点
                cv::imshow("mask_background",mask_background);
                cv::waitKey(0);
                cv::cuda::threshold(img.merge_mask_gpu,img.merge_mask_gpu,0.5,255,CV_8UC1);*/
                mask_background_gpu.upload(mask_background);
                auto new_pts = DetectShiTomasiCornersGpu(max_new_detect, img.gray0_gpu, mask_background_gpu);
                visual_new_points_ = new_pts;
                Debugt("instsTrack actually detect num:{}", new_pts.size());
                for(auto &pt : new_pts){
                    int index_inst=-1;
                    for(auto &[key,inst] : instances_){
                        if(inst.lost_num>0 || inst.curr_points.size()>cfg::kMaxDynamicCnt) continue;
                        if(inst.mask_img.at<uchar>(pt) >= 1){
                            index_inst=(int)key;
                            break;
                        }
                    }
                    if(index_inst!=-1){
                        instances_[index_inst].curr_points.emplace_back(pt);
                        instances_[index_inst].ids.emplace_back(global_id_count++);
                        instances_[index_inst].visual_new_points.emplace_back(pt);
                        instances_[index_inst].track_cnt.push_back(1);
                    }
                }
                Infot("instsTrack detectNewFeaturesGPU:{} ms", tic_toc.TocThenTic());
            }

        }


        for(auto& [key,inst] : instances_){
            ///去畸变和计算归一化坐标
            inst.curr_un_points= UndistortedPts(inst.curr_points, camera_);
            ///计算特征点的速度
            SetIdPointPair(inst.ids,inst.curr_un_points,inst.curr_id_pts);
            PtsVelocity(curr_time - last_time, inst.ids, inst.curr_un_points,
                        inst.prev_id_pts, inst.pts_velocity);
        }
        Infot("instsTrack UndistortedPts & PtsVelocity:{} ms", tic_toc.TocThenTic());

        /// 右边相机图像的跟踪
        if((!img.gray1.empty() || !img.gray1_gpu.empty()) && cfg::is_stereo){
            for(auto& [key,inst] : instances_){
                inst.right_points.clear();
                if(!inst.curr_points.empty() && inst.lost_num==0){
                    auto status= FeatureTrackByLK(img.gray0,img.gray1,inst.curr_points,inst.right_points);
                    /*auto status = FeatureTrackByLKGpu(lk_optical_flow, lk_optical_flow_back, img.gray0_gpu,
                                                      img.gray1_gpu,inst.curr_points, inst.right_points);*/
                    if(cfg::dataset == DatasetType::kViode){
                        for(size_t i=0;i<status.size();++i){
                            if(status[i] && VIODE::PixelToKey(inst.right_points[i], img.seg1) != key)
                                status[i]=0;
                        }
                    }
                    inst.right_ids = inst.ids;
                    ReduceVector(inst.right_points, status);
                    ReduceVector(inst.right_ids, status);
                    inst.right_un_points = UndistortedPts(inst.right_points, right_camera_);
                    SetIdPointPair(inst.right_ids,inst.right_un_points,inst.right_curr_id_pts);
                    PtsVelocity(curr_time - last_time, inst.right_ids, inst.right_un_points,
                                inst.right_prev_id_pts, inst.right_pts_velocity);
                }
            }
        }
        Infot("instsTrack flowTrack right:{} ms", tic_toc.TocThenTic());
        ManageInstances();
        ///输出实例数据
        Debugt("InstanceTracker:实例数量:{}", instances_.size());
        for(auto &[key,inst] : instances_){
            Debugt("Inst:{} size:{}", key, inst.curr_points.size());
        }
        for(auto& [key,inst] : instances_){
            inst.last_points=inst.curr_points;
            inst.prev_id_pts=inst.curr_id_pts;
            inst.right_prev_id_pts=inst.right_curr_id_pts;
        }
    }
    else
    {
        ManageInstances();
        for(auto& [key,inst] : instances_){
            inst.curr_points.clear();
            inst.curr_un_points.clear();
            inst.last_points.clear();
            inst.right_points.clear();
            inst.right_un_points.clear();
            inst.ids.clear();
            inst.right_ids.clear();
            inst.pts_velocity.clear();
            inst.right_pts_velocity.clear();
            inst.prev_id_pts.clear();
            inst.visual_points_pair.clear();
            inst.visual_right_points_pair.clear();
            inst.visual_new_points.clear();
            inst.track_cnt.clear();
        }
    }
    last_time=curr_time;
    global_frame_id++;
    prev_img = img;
}

/*

void InstsFeatManager::InstsFlowTrack(SegImage img)
{
    TicToc tic_toc;
    curr_time=img.time0;

    if(cfg::dataset == DatasetType::kKitti){
        //AddInstancesGPU(img);
        AddInstancesByTracking(img);
        Infot("instsTrack AddInstances:{} ms", tic_toc.TocThenTic());
        exist_inst_ = !img.insts_info.empty();
    }
    else if(cfg::dataset == DatasetType::kViode){
        AddViodeInstances(img);
        Infot("instsTrack addViodeInstancesBySegImg:{} ms", tic_toc.TocThenTic());
    }
    else{
        string msg="Have not this dataset Type";
        Criticalt(msg);
        throw std::runtime_error(msg);
    }

    //cv::Mat flow_show = VisualFlow(flow_tensor);
    //cv::imshow("flow",flow_show);
    //cv::waitKey(1);

    if(exist_inst_){
        ///形态学运算
        //Erode10Gpu(img.merge_mask_gpu,img.merge_mask_gpu);
        img.merge_mask_gpu.download(img.merge_mask);
        for(auto& [key,inst] : instances_){
            if(inst.last_frame_cnt < global_frame_id)
                inst.lost_num++;
            else if(inst.last_frame_cnt == global_frame_id)
                inst.lost_num=0;
        }

        ///对每个目标进行光流跟踪
        for(auto & [ key,inst] : instances_){
            if(inst.last_points.empty() || inst.lost_num>0)
                continue;
            inst.curr_points.clear();
            ///光流跟踪
            auto status = FeatureTrackByDenseFlow(img.flow,inst.last_points, inst.curr_points);
            for(size_t i=0;i<status.size();++i)
                if(status[i] && inst.mask_img.at<uchar>(inst.curr_points[i]) == 0)
                    status[i]=0;
            ReduceVector(inst.curr_points, status);
            ReduceVector(inst.ids, status);
            ReduceVector(inst.last_points, status);
        }

        Infot("instsTrack flowTrack:{} ms", tic_toc.TocThenTic());

        ///对每个特征点增加观测次数
        for(auto & [ key,inst] : instances_){
            if(inst.last_points.empty() || inst.lost_num>0)
                continue;
            for (auto &n : inst.track_cnt) n++;
        }

        ///添加新的特征点前的准备
        int max_new_detect=0;
        for(auto & [key,inst] : instances_){
            if(inst.lost_num>0 || inst.curr_points.size()>= cfg::kMaxDynamicCnt)
                continue;
            max_new_detect += (cfg::kMaxDynamicCnt - (int)inst.curr_points.size());
        }
        ///添加新的特征点
        if(max_new_detect > 0){
            mask_background = img.merge_mask;
            vector<cv::Point2f> points_existed;
            for(auto & [key,inst] : instances_){
                if(inst.lost_num>0 || inst.curr_points.size()>=cfg::kMaxDynamicCnt)continue;
                inst.visual_points_pair.clear();
                inst.visual_right_points_pair.clear();
                inst.visual_new_points.clear();
                for(size_t i=0;i<inst.curr_points.size();++i){
                    inst.visual_points_pair.emplace_back(inst.last_points[i],inst.curr_points[i]);//用于可视化
                    cv::circle(mask_background, inst.curr_points[i], cfg::kMinDynamicDist, 0, -1);//设置mask
                }
                points_existed.insert(points_existed.end(),inst.curr_points.begin(),inst.curr_points.end());
            }

            auto new_pts = DetectRegularCorners(max_new_detect,mask_background,points_existed);

            visual_new_points_.clear();
            for(auto &pt : new_pts){
                int index_inst=-1;
                for(auto &[key,inst] : instances_){
                    if(inst.lost_num>0 || inst.curr_points.size()>cfg::kMaxDynamicCnt) continue;
                    if(inst.mask_img.at<uchar>(pt) >= 1){
                        index_inst=(int)key;
                        break;
                    }
                }
                if(index_inst!=-1){
                    instances_[index_inst].curr_points.emplace_back(pt);
                    instances_[index_inst].ids.emplace_back(global_id_count++);
                    instances_[index_inst].visual_new_points.emplace_back(pt);
                    instances_[index_inst].track_cnt.push_back(1);
                    visual_new_points_.push_back(pt);
                }
            }
            Infot("instsTrack detectNewFeaturesGPU:{} ms", tic_toc.TocThenTic());
        }


        for(auto& [key,inst] : instances_){
            ///去畸变和计算归一化坐标
            inst.curr_un_points= UndistortedPts(inst.curr_points, camera_);
            ///计算特征点的速度
            SetIdPointPair(inst.ids, inst.curr_un_points,inst.curr_id_pts);
            PtsVelocity(curr_time - last_time, inst.ids, inst.curr_un_points,
                        inst.prev_id_pts, inst.pts_velocity);
        }
        Infot("instsTrack undistortedPts & ptsVelocity:{} ms", tic_toc.TocThenTic());

        /// 右边相机图像的跟踪

*/
/*if((!img.gray1.empty() || !img.gray1_gpu.empty()) && is_stereo_){
            for(auto& [key,inst] : instances_){
                inst.right_points.clear();
                if(!inst.curr_points.empty() && inst.lost_num==0){
                    //auto status= flowTrack (img.gray0,img.gray1,inst.curr_points,inst.right_points);
                    auto status = FeatureTrackByLKGpu(lk_optical_flow, lk_optical_flow_back, img.gray0_gpu,
                                                      img.gray1_gpu,
                                                      inst.curr_points, inst.right_points);
                    if(cfg::dataset == DatasetType::kViode){
                        for(size_t i=0;i<status.size();++i)
                            if(status[i] && VIODE::PixelToKey(inst.right_points[i], img.seg1) != key)
                                status[i]=0;
                    }
                    inst.right_ids = inst.ids;
                    ReduceVector(inst.right_points, status);
                    ReduceVector(inst.right_ids, status);
                    inst.right_un_points = UndistortedPts(inst.right_points, right_camera_);
                    PtsVelocity(curr_time - last_time, inst.right_ids, inst.right_un_points,
                                inst.right_prev_id_pts, inst.right_curr_id_pts, inst.right_pts_velocity);
                }
            }
        }*/
/*


        ManageInstances();
        ///输出实例数据
        for(auto &[key,inst] : instances_){
            Debugt("Inst:{} size:{}", key, inst.curr_points.size());
        }
        for(auto& [key,inst] : instances_){
            inst.last_points=inst.curr_points;
            inst.prev_id_pts=inst.curr_id_pts;
            inst.right_prev_id_pts=inst.right_curr_id_pts;
        }

    }
    else
    {
        ManageInstances();
        for(auto& [key,inst] : instances_){
            inst.curr_points.clear();
            inst.curr_un_points.clear();
            inst.last_points.clear();
            inst.right_points.clear();
            inst.right_un_points.clear();
            inst.ids.clear();
            inst.right_ids.clear();
            inst.pts_velocity.clear();
            inst.right_pts_velocity.clear();
            inst.prev_id_pts.clear();
            inst.visual_points_pair.clear();
            inst.visual_right_points_pair.clear();
            inst.visual_new_points.clear();
            inst.track_cnt.clear();
        }
    }

    last_time=curr_time;
    global_frame_id++;

    prev_img = img;
}

*/



void InstsFeatManager::InstsTrackByMatching(SegImage img)
{
    TicToc tic_toc;
    curr_time=img.time0;

    if(cfg::dataset == DatasetType::kKitti){
        AddInstancesByTracking(img);
    }
    else if(cfg::dataset == DatasetType::kViode){
        AddViodeInstances(img);
    }
    else{
        throw std::runtime_error("Have not this dataset type");
    }
    Infot("instsTrack AddInstances:{} ms", tic_toc.TocThenTic());

    exist_inst_ = !img.insts_info.empty();

    img.gray0_gpu.download(img.gray0);

    if(exist_inst_){
        ///形态学运算
        //ErodeMaskGpu(img.merge_mask_gpu, img.merge_mask_gpu);
        img.merge_mask_gpu.download(img.merge_mask);
        for(auto& [key,inst] : instances_){
            if(inst.last_frame_cnt < global_frame_id)
                inst.lost_num++;
            else if(inst.last_frame_cnt == global_frame_id)
                inst.lost_num=0;
        }
        cv::Ptr<cv::FeatureDetector> orb_detector = cv::ORB::create();

        //对每个目标进行特征检测
        for(auto & [ key,inst] : instances_){
            if(inst.orb_last_keypoints.empty() || inst.lost_num>0)
                continue;
            Debugt("{} last size:{} {}",key,inst.orb_last_keypoints.size(),inst.orb_last_descriptors.rows);
            std::vector<cv::KeyPoint> orb_keypoints;
            orb_detector->detect(img.gray0,orb_keypoints,inst.mask_img);
            cv::Mat orb_descriptors;
            orb_detector->compute(img.gray0,orb_keypoints,orb_descriptors);
            Debugt("{} detect:{} {}",key,orb_keypoints.size(),orb_descriptors.rows);

            inst.orb_descriptors.release();
            inst.orb_keypoints.clear();

            if(orb_keypoints.empty())
                continue;

            //特征匹配
            std::vector<cv::DMatch> matches;
            orb_matcher_->match(inst.orb_last_descriptors,orb_descriptors,matches);
            //计算最小距离和最大距离
            auto minmax = std::minmax_element(matches.begin(),matches.end(),[](const cv::DMatch &m1,const cv::DMatch &m2){
                return m1.distance < m2.distance;
            });
            double min_dist = minmax.first->distance;
            double max_dist = minmax.second->distance;
            Debugt("{} min:{} max:{}",key,min_dist,max_dist);

            std::vector<uchar> status;
            for(int i=0;i<inst.orb_last_descriptors.rows;++i){
                if(matches[i].distance <= std::max(2*min_dist,30.))
                    status.push_back(1);
                else
                    status.push_back(0);
                inst.orb_keypoints.push_back(orb_keypoints[matches[i].trainIdx]);
                if(status[i]){
                    inst.orb_descriptors.push_back(orb_descriptors.row(matches[i].trainIdx));
                }
            }
            inst.curr_points.clear();
            for(int i=0;i<inst.orb_keypoints.size();++i){
                inst.curr_points.emplace_back(inst.orb_keypoints[i].pt);
            }
            ReduceVector(inst.orb_keypoints, status);
            ReduceVector(inst.curr_points, status);
            ReduceVector(inst.ids, status);
            ReduceVector(inst.last_points, status);
            Debugt("{} success num:{}",key,inst.curr_points.size());
        }

        for(auto &[key,inst] : instances_){
            Debugt("{} after match{} {}",key,inst.orb_keypoints.size(),inst.orb_descriptors.rows);
        }

        Infot("instsTrack flowTrack:{} ms", tic_toc.TocThenTic());

        ///对每个特征点增加观测次数
        for(auto & [ key,inst] : instances_){
            if(inst.last_points.empty() || inst.lost_num>0)
                continue;
            for (auto &n : inst.track_cnt) n++;
        }

        ///添加新的特征点前的准备
        int max_new_detect=0;
        for(auto & [key,inst] : instances_){
            if(inst.lost_num>0 || inst.curr_points.size()>= cfg::kMaxDynamicCnt)
                continue;
            max_new_detect += (cfg::kMaxDynamicCnt - (int)inst.curr_points.size());
        }
        ///添加新的特征点
        if(max_new_detect > 0){
            mask_background = img.merge_mask;
            for(auto & [key,inst] : instances_){
                if(inst.lost_num>0 || inst.curr_points.size()>=cfg::kMaxDynamicCnt)
                    continue;
                inst.visual_points_pair.clear();
                inst.visual_right_points_pair.clear();
                inst.visual_new_points.clear();
                for(size_t i=0;i<inst.curr_points.size();++i){
                    inst.visual_points_pair.emplace_back(inst.last_points[i],inst.curr_points[i]);//用于可视化
                    cv::circle(mask_background, inst.curr_points[i], cfg::kMinDynamicDist, 0, -1);//设置mask
                }
            }
            Infot("instsTrack prepare detect:{} ms", tic_toc.TocThenTic());

            cv::Ptr<cv::FeatureDetector> orb_detector_all = cv::ORB::create(max_new_detect);
            std::vector<cv::KeyPoint> orb_keypoints_all;
            orb_detector_all->detect(img.gray0,orb_keypoints_all,mask_background);
            cv::Mat orb_descriptors_all;
            orb_detector_all->compute(img.gray0,orb_keypoints_all,orb_descriptors_all);
            Debugt("instsTrack actually detect num:{}", orb_keypoints_all.size());
            for(int i=0;i<orb_keypoints_all.size();++i){
                cv::KeyPoint pt = orb_keypoints_all[i];
                visual_new_points_.push_back(pt.pt);
                int index_inst=-1;
                for(auto &[key,inst] : instances_){
                    if(inst.lost_num>0 || inst.curr_points.size()>cfg::kMaxDynamicCnt) continue;
                    if(inst.mask_img.at<uchar>(pt.pt) >= 1){
                        index_inst=(int)key;
                        break;
                    }
                }
                if(index_inst!=-1){
                    instances_[index_inst].curr_points.emplace_back(pt.pt);
                    instances_[index_inst].ids.emplace_back(global_id_count++);
                    instances_[index_inst].visual_new_points.emplace_back(pt.pt);
                    instances_[index_inst].track_cnt.push_back(1);
                    instances_[index_inst].orb_keypoints.push_back(pt);
                    instances_[index_inst].orb_descriptors.push_back(orb_descriptors_all.row(i));
                }
            }
            for(auto &[key,inst] : instances_){
                Debugt("{} new add:{} {}",key,inst.orb_keypoints.size(),inst.orb_descriptors.rows);
            }

            Infot("instsTrack detectNewFeaturesGPU:{} ms", tic_toc.TocThenTic());
        }

        for(auto& [key,inst] : instances_){
            ///去畸变和计算归一化坐标
            inst.curr_un_points= UndistortedPts(inst.curr_points, camera_);
            ///计算特征点的速度
            SetIdPointPair(inst.ids,inst.curr_un_points,inst.curr_id_pts);
            PtsVelocity(curr_time - last_time, inst.ids, inst.curr_un_points,
                        inst.prev_id_pts, inst.pts_velocity);
        }
        Infot("instsTrack UndistortedPts & PtsVelocity:{} ms", tic_toc.TocThenTic());

        /*/// 右边相机图像的跟踪
        if((!img.gray1.empty() || !img.gray1_gpu.empty()) && cfg::is_stereo){
            for(auto& [key,inst] : instances_){
                inst.right_points.clear();
                if(!inst.curr_points.empty() && inst.lost_num==0){
                    auto status= FeatureTrackByLK(img.gray0,img.gray1,inst.curr_points,inst.right_points);
                    if(cfg::dataset == DatasetType::kViode){
                        for(size_t i=0;i<status.size();++i){
                            if(status[i] && VIODE::PixelToKey(inst.right_points[i], img.seg1) != key)
                                status[i]=0;
                        }
                    }
                    inst.right_ids = inst.ids;
                    ReduceVector(inst.right_points, status);
                    ReduceVector(inst.right_ids, status);
                    inst.right_un_points = UndistortedPts(inst.right_points, right_camera_);
                    SetIdPointPair(inst.right_ids,inst.right_un_points,inst.right_curr_id_pts);
                    PtsVelocity(curr_time - last_time, inst.right_ids, inst.right_un_points,
                                inst.right_prev_id_pts, inst.right_pts_velocity);
                }
            }
        }
        Infot("instsTrack flowTrack right:{} ms", tic_toc.TocThenTic());*/

        ManageInstances();
        ///输出实例数据
        Debugt("InstanceTracker:实例数量:{}", instances_.size());
        for(auto &[key,inst] : instances_){
            Debugt("Inst:{} size:{}", key, inst.curr_points.size());
        }
        for(auto& [key,inst] : instances_){
            inst.last_points=inst.curr_points;
            inst.prev_id_pts=inst.curr_id_pts;
            inst.right_prev_id_pts=inst.right_curr_id_pts;
        }
    }
    else
    {
        ManageInstances();
        for(auto& [key,inst] : instances_){
            inst.curr_points.clear();
            inst.curr_un_points.clear();
            inst.last_points.clear();
            inst.right_points.clear();
            inst.right_un_points.clear();
            inst.ids.clear();
            inst.right_ids.clear();
            inst.pts_velocity.clear();
            inst.right_pts_velocity.clear();
            inst.prev_id_pts.clear();
            inst.visual_points_pair.clear();
            inst.visual_right_points_pair.clear();
            inst.visual_new_points.clear();
            inst.track_cnt.clear();
        }
    }

    for(auto& [key,inst] : instances_){
        inst.orb_last_keypoints = inst.orb_keypoints;
        inst.orb_last_descriptors = inst.orb_descriptors;
    }

    last_time=curr_time;
    global_frame_id++;
    prev_img = img;
}



/**
 *删除不被观测到的实例,设置lost_num变量是为了防止有时候深度学习算法忽略了某帧
 */
void InstsFeatManager::ManageInstances()
{
    for(auto it=instances_.begin(),it_next=it; it != instances_.end(); it=it_next)
    {
        it_next++;
        auto &inst=it->second;
        if(inst.lost_num ==0 && inst.curr_points.empty()){
            inst.lost_num++;
        }
        if(inst.lost_num > 0){
            inst.lost_num++;
            if(inst.lost_num > 3){ //删除该实例
                instances_.erase(it);
            }
        }
    }
}


/**
 ** 用于将特征点传到VIO模块
 * @param result
 */
InstancesFeatureMap InstsFeatManager::GetOutputFeature()
{
    InstancesFeatureMap result;
    for(auto& [key,inst]: instances_)
    {
        if(inst.lost_num>0)continue;
        InstanceFeatureSimple  features_map;
        features_map.color = inst.color;
        for(int i=0;i<(int)inst.curr_un_points.size();++i){
            Eigen::Matrix<double,5,1> feat;
            feat<<inst.curr_un_points[i].x,inst.curr_un_points[i].y, 1 ,inst.pts_velocity[i].x,inst.pts_velocity[i].y;
            vector<Eigen::Matrix<double,5,1>> vp={feat};
            features_map.insert({inst.ids[i],vp});
        }
        int right_cnt=0;
        if(cfg::is_stereo){
            for(int i=0; i<(int)inst.right_un_points.size(); i++){
                auto r_id = inst.right_ids[i];
                if(features_map.count(r_id) ==0)
                    continue;
                Eigen::Matrix<double,5,1> feat;
                feat<<inst.right_un_points[i].x,inst.right_un_points[i].y, 1 ,inst.right_pts_velocity[i].x,inst.right_pts_velocity[i].y;
                features_map[r_id].push_back(feat);
                right_cnt++;
            }
        }
        result.insert({key,features_map});
        Debugt("inst_id:{} l_track:{} r_track:{}", key, features_map.size(), right_cnt);
    }
    return result;
}


/**
 * 根据语义标签，多线程设置mask
 * @param mask_img 原来的mask
 * @param semantic_img 语义标签图像
 */
void InstsFeatManager::AddViodeInstances(SegImage &img)
{
    cv::Mat seg = img.seg0;
    Debugt("start to addViodeInstancesBySegImg()");
    for(auto &[key,inst] : instances_){
        inst.mask_area=0;
    }
    Debugt("addViodeInstancesBySegImg merge insts");
    for(auto &inst_info : img.insts_info){
        auto key = inst_info.track_id;
        if(instances_.count(key) == 0){
            InstFeat instanceFeature(key, 0);
            instances_.insert({key, instanceFeature});
        }
        auto &inst = instances_[key];
        inst.mask_img = inst_info.mask_cv;
        inst.box_vel = cv::Point2f(0,0);
        inst.last_frame_cnt = global_frame_id;
        inst.last_time = img.time0;
        Debugt("inst:{} mask_img:({}x{}) local_mask:({}x{})", key, instances_[key].mask_img.rows,
               instances_[key].mask_img.cols,
               inst.mask_img.rows, inst.mask_img.cols);
    }
    for(auto &[key,inst]: instances_){
        if(inst.last_frame_cnt == global_frame_id){
            auto rect = cv::boundingRect(inst.mask_img);
            inst.box_min_pt= rect.tl();
            inst.box_max_pt = rect.br();
            inst.box_center_pt = (inst.box_min_pt+inst.box_max_pt)/2;
            inst.color = img.seg0.at<cv::Vec3b>(inst.box_center_pt);
        }
    }
}



/**
 *
 * @param instInfo
 * @param inst_mask_tensor
 * @param inst_mask_area
 * @return
 */
std::tuple<int,float,float> InstsFeatManager::GetMatchInst(InstInfo &instInfo, torch::Tensor &inst_mask_tensor)
{
    int h=(int)inst_mask_tensor.sizes()[0];
    int w=(int)inst_mask_tensor.sizes()[1];

    float inst_mask_area = inst_mask_tensor.sum(torch::IntArrayRef({0,1})).item().toFloat();
    cv::Point2f inst_center_pt = (instInfo.min_pt + instInfo.max_pt)/2.;

    int id_match=-1;
    float iou_max=0;
    for(const auto &[key, inst_j] : instances_){
        ///根据速度计算当前的物体
        cv::Point2i delta = inst_j.box_vel * inst_j.delta_time;
        auto curr_min_pt = cv::Point2i(inst_j.box_min_pt) + delta;
        auto curr_max_pt = cv::Point2i(inst_j.box_max_pt) + delta;
        cv::Point2f curr_center_pt = (curr_max_pt+curr_min_pt)/2.;

        auto delta_pt = curr_center_pt - inst_center_pt;
        float delta_pt_abs = std::abs(delta_pt.x)+std::abs(delta_pt.y);

        if(CalBoxIoU(instInfo.min_pt, instInfo.max_pt, curr_min_pt, curr_max_pt) > 0 || delta_pt_abs < 100){
            torch::Tensor intersection_mask;
            if(delta.x>=0 && delta.y>=0){
                intersection_mask = inst_mask_tensor.index({idx::Slice(delta.y,idx::None),idx::Slice(delta.x,idx::None)}) *
                        inst_j.mask_tensor.index({idx::Slice(idx::None,h-delta.y),idx::Slice(idx::None,w-delta.x)});
            }
            else if(delta.x<0 && delta.y>=0){
                intersection_mask = inst_mask_tensor.index({idx::Slice(delta.y,idx::None),idx::Slice(idx::None,w+delta.x)}) *
                        inst_j.mask_tensor.index({idx::Slice(idx::None,h-delta.y),idx::Slice(-delta.x,idx::None)});
            }
            else if(delta.x>=0 && delta.y<0){
                intersection_mask = inst_mask_tensor.index({idx::Slice(idx::None,h+delta.y),idx::Slice(delta.x,idx::None)}) *
                        inst_j.mask_tensor.index({idx::Slice(-delta.y,idx::None),idx::Slice(idx::None,w-delta.x)});
            }
            else if(delta.x<0 && delta.y<0){
                intersection_mask = inst_mask_tensor.index({idx::Slice(idx::None,h+delta.y),idx::Slice(idx::None,w+delta.x)}) *
                        inst_j.mask_tensor.index({idx::Slice(-delta.y,idx::None),idx::Slice(-delta.x,idx::None)});
            }
            float intersection_area = intersection_mask.sum(torch::IntArrayRef({0,1})).item().toFloat();
            float iou = intersection_area / (inst_mask_area + inst_j.mask_area - intersection_area);
            if(iou > iou_max){
                iou_max = iou;
                id_match = (int)inst_j.id;
            }
        }
        else{
        }
    }

    return {id_match,iou_max,inst_mask_area};
}

cv::Mat InstsFeatManager::AddInstances(SegImage &img)
{
    double current_time = img.time0;
    int n_inst = (int)img.insts_info.size();
    ///set inst time
    for(auto &[key,inst] : instances_){
        inst.delta_time = current_time - inst.last_time;
        inst.last_time = current_time;
    }
    if(img.insts_info.empty()){
        //mask_background.release();
        mask_background_gpu.release();
        return {};
    }
    assert(img.mask_tensor.sizes()[0] == img.insts_info.size());

    cv::Size mask_size((int)img.mask_tensor.sizes()[2],(int)img.mask_tensor.sizes()[1]);
    //mask_background = img.merge_mask;
    mask_background_gpu = img.merge_mask_gpu;
    for(int i=0; i < n_inst; ++i)
    {
        auto inst_mask_tensor = img.mask_tensor[i];
        auto instInfo = img.insts_info[i];
        ///寻找匹配的实例
        auto [id_match,iou_max,inst_mask_area] = GetMatchInst(instInfo, inst_mask_tensor);
        ///更新实例
        if(iou_max > 0.01 && instances_[id_match].class_id == instInfo.label_id){
            //instances[id_match].mask_img = img.inst_masks[i];
            instances_[id_match].mask_img_gpu = instInfo.mask_gpu;
            instances_[id_match].mask_tensor = inst_mask_tensor;
            instances_[id_match].box_min_pt = instInfo.min_pt;
            instances_[id_match].box_max_pt = instInfo.max_pt;
            instances_[id_match].mask_area = inst_mask_area;
            instances_[id_match].last_frame_cnt = global_frame_id;
            if(instances_[id_match].delta_time > 0){
                instances_[id_match].box_vel = (instInfo.mask_center - instances_[id_match].box_center_pt) / instances_[id_match].delta_time;
            }
            instances_[id_match].box_center_pt = instInfo.mask_center;
            //cout<<fmt::format("Update,id:{},iou_max:{}",instances[id_match].id,iou_max)<<endl;
        }
        ///创建实例
        else{
            unsigned int id=global_instance_id+i;
            InstFeat inst_feat(id, instInfo.label_id);
            //inst_feat.mask_img = img.inst_masks[i];
            inst_feat.mask_img_gpu = instInfo.mask_gpu;
            inst_feat.mask_tensor = inst_mask_tensor;
            inst_feat.box_min_pt = instInfo.min_pt;
            inst_feat.box_max_pt = instInfo.max_pt;
            inst_feat.box_center_pt = instInfo.mask_center;
            inst_feat.box_vel = cv::Point2f(0,0);
            inst_feat.class_id = instInfo.label_id;
            inst_feat.mask_area = inst_mask_area;
            inst_feat.last_time = current_time;
            inst_feat.last_frame_cnt = global_frame_id;
            instances_.insert({id, inst_feat});
            //cout<<fmt::format("Insert,id:{},iou_max:{}",id,iou_max)<<endl;
        }
    }
    global_instance_id+= n_inst;
    return ~mask_background;
}


void InstsFeatManager:: AddInstancesGPU(const SegImage &img)
{
    double current_time = img.time0;
    int n_inst = (int)img.insts_info.size();
    ///set inst time
    for(auto &[key,inst] : instances_){
        inst.delta_time = current_time - inst.last_time;
        inst.last_time = current_time;
    }
    if(img.insts_info.empty())
        return;
    assert(img.mask_tensor.sizes()[0] == img.insts_info.size());
    cv::Size mask_size((int)img.mask_tensor.sizes()[2],(int)img.mask_tensor.sizes()[1]);
    //mask_background = img.merge_mask;
    for(int i=0; i < n_inst; ++i)
    {
        auto inst_mask_tensor = img.mask_tensor[i];
        auto instInfo = img.insts_info[i];
        ///寻找匹配的实例
        auto [id_match,iou_max,inst_mask_area] = GetMatchInst(instInfo, inst_mask_tensor);
        ///更新实例
        if(iou_max > 0.01 && instances_[id_match].class_id == instInfo.label_id){
            //instances[id_match].mask_img = img.inst_masks[i];
            instances_[id_match].mask_img_gpu = instInfo.mask_gpu;
            instances_[id_match].mask_img = instInfo.mask_cv;
            instances_[id_match].mask_tensor = inst_mask_tensor;
            instances_[id_match].box_min_pt = instInfo.min_pt;
            instances_[id_match].box_max_pt = instInfo.max_pt;
            instances_[id_match].mask_area = inst_mask_area;
            instances_[id_match].last_frame_cnt = global_frame_id;
            if(instances_[id_match].delta_time > 0){
                instances_[id_match].box_vel = (instInfo.mask_center - instances_[id_match].box_center_pt) / instances_[id_match].delta_time;
            }
            instances_[id_match].box_center_pt = instInfo.mask_center;
            //cout<<fmt::format("Update,id:{},iou_max:{}",instances[id_match].id,iou_max)<<endl;
        }
        ///创建实例
        else{
            unsigned int id=global_instance_id+i;
            InstFeat inst_feat(id, instInfo.label_id);
            //inst_feat.mask_img = img.inst_masks[i];
            inst_feat.mask_img_gpu = instInfo.mask_gpu;
            inst_feat.mask_img = instInfo.mask_cv;
            inst_feat.mask_tensor = inst_mask_tensor;
            inst_feat.box_min_pt = instInfo.min_pt;
            inst_feat.box_max_pt = instInfo.max_pt;
            inst_feat.box_center_pt = instInfo.mask_center;
            inst_feat.box_vel = cv::Point2f(0,0);
            inst_feat.class_id = instInfo.label_id;
            inst_feat.mask_area = inst_mask_area;
            inst_feat.last_time = current_time;
            inst_feat.last_frame_cnt = global_frame_id;
            instances_.insert({id, inst_feat});
            //cout<<fmt::format("Insert,id:{},iou_max:{}",id,iou_max)<<endl;
        }
    }

    global_instance_id+= n_inst;
}



void InstsFeatManager:: AddInstancesByTracking(SegImage &img)
{
    double current_time = img.time0;
    int n_inst = (int)img.insts_info.size();
    if(img.insts_info.empty())
        return;
    assert(img.mask_tensor.sizes()[0] == img.insts_info.size());
    cv::Size mask_size((int)img.mask_tensor.sizes()[2],(int)img.mask_tensor.sizes()[1]);
    //mask_background = img.merge_mask;
    auto trks = mot_tracker->update(img.insts_info,img.color0);
    for(auto &inst : trks){
        unsigned int id=inst.track_id;
        if(instances_.count(id) == 0){
            InstFeat inst_feat(id, inst.label_id);
            inst_feat.mask_img_gpu = inst.mask_gpu;
            inst_feat.mask_img = inst.mask_cv;
            inst_feat.mask_tensor = inst.mask_tensor;
            inst_feat.box_min_pt = inst.min_pt;
            inst_feat.box_max_pt = inst.max_pt;
            inst_feat.box_center_pt = inst.mask_center;
            inst_feat.box_vel = cv::Point2f(0,0);
            inst_feat.class_id = inst.label_id;
            inst_feat.last_time = current_time;
            inst_feat.last_frame_cnt = global_frame_id;
            instances_.insert({id, inst_feat});
            Debugt("Create inst:{} cls:{}", id, inst.name);
        }
        else{
            instances_[id].mask_img_gpu = inst.mask_gpu;
            instances_[id].mask_img = inst.mask_cv;
            instances_[id].mask_tensor = inst.mask_tensor;
            instances_[id].box_min_pt = inst.min_pt;
            instances_[id].box_max_pt = inst.max_pt;
            instances_[id].last_frame_cnt = global_frame_id;
            instances_[id].box_center_pt = inst.mask_center;
            Debugt("Update inst:{} cls:{}", id, inst.name);
        }
    }
    global_instance_id+= n_inst;
}


vector<uchar> InstsFeatManager::RejectWithF(InstFeat &inst, int col, int row) const
{
    vector<cv::Point2f> un_cur_pts(inst.curr_points.size()), un_prev_pts(inst.last_points.size());
    for (unsigned int i = 0; i < inst.curr_points.size(); i++){
        Eigen::Vector3d tmp_p;
        camera_->liftProjective(Eigen::Vector2d(inst.curr_points[i].x, inst.curr_points[i].y), tmp_p);
        tmp_p.x() = kFocalLength * tmp_p.x() / tmp_p.z() + col / 2.0;
        tmp_p.y() = kFocalLength * tmp_p.y() / tmp_p.z() + row / 2.0;
        un_cur_pts[i] = cv::Point2f((float)(tmp_p.x()), (float)tmp_p.y());
        camera_->liftProjective(Eigen::Vector2d(inst.last_points[i].x, inst.last_points[i].y), tmp_p);
        tmp_p.x() = kFocalLength * tmp_p.x() / tmp_p.z() + col / 2.0;
        tmp_p.y() = kFocalLength * tmp_p.y() / tmp_p.z() + row / 2.0;
        un_prev_pts[i] = cv::Point2f((float)tmp_p.x(), (float)tmp_p.y());
    }
    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, cfg::kFThreshold,
                           0.99, status);
    return status;
}


void InstsFeatManager::VisualizeInst(cv::Mat &img)
{
    for(const auto &[id,inst]: instances_){
        if(inst.lost_num>0 || inst.curr_points.empty())
            continue;
        for(const auto &[pt1,pt2] : inst.visual_points_pair){
            //cv::circle(img, pt1, 2, cv::Scalar(255, 255, 255), 2);//上一帧的点
            cv::circle(img, pt2, 2, inst.color, 2);//当前帧的点
            //cv::arrowedLine(img, pt2, pt1, inst.color, 1, 8, 0, 0.2);
        }
        for(const auto &pt : inst.visual_new_points){
            cv::circle(img, pt, 2, inst.color, 2);
        }
        cv::rectangle(img,inst.box_min_pt,inst.box_max_pt,inst.color);
        //std::string label=fmt::format("id:{},tck:{}",id,inst.curr_points.size() - inst.visual_new_points.size());
        std::string label=fmt::format("id:{}",id);
        cv::putText(img, label, inst.feats_center_pt, cv::FONT_HERSHEY_SIMPLEX,
                    1.0, inst.color, 2);
        /*if(vel_map_.count(inst.id)!=0){
            auto anchor=inst.feats_center_pt;
            anchor.y += 40;
            double v_abs = vel_map_[inst.id].v.norm();
            cv::putText(img, fmt::format("v:{:.2f} m/s",v_abs),anchor,cv::FONT_HERSHEY_SIMPLEX,1.0,inst.color,2);
        }*/
    }
    cv::imshow("insts",img);
    cv::waitKey(1);
}


void InstsFeatManager::DrawInsts(cv::Mat& img)
{
    if(cfg::slam == SlamType::kDynamic){
        for(const auto &[id,inst]: instances_){
            if(inst.lost_num>0 || inst.curr_points.empty())
                continue;
            for(const auto &[pt1,pt2] : inst.visual_points_pair){
                //cv::circle(img, pt1, 2, cv::Scalar(255, 255, 255), 2);//上一帧的点
                cv::circle(img, pt2, 2, inst.color, 2);//当前帧的点
                //cv::arrowedLine(img, pt2, pt1, inst.color, 1, 8, 0, 0.2);
            }
            for(const auto &pt : inst.visual_new_points){
                cv::circle(img, pt, 3, cv::Scalar(255,255,255), 2);
            }
            std::string label=fmt::format("id:{},tck:{}",id,inst.curr_points.size() - inst.visual_new_points.size());
            //cv::putText(img, label, inst.box_center_pt, cv::FONT_HERSHEY_SIMPLEX, 1.0, inst.color, 2);
            DrawText(img, label, inst.color, inst.box_center_pt, 1.0, 2, false);

            /*if(vel_map_.count(inst.id)!=0){
                auto anchor=inst.feats_center_pt;
                anchor.y += 40;
                double v_abs = vel_map_[inst.id].v.norm();
                cv::putText(img, fmt::format("v:{:.2f} m/s",v_abs),anchor,cv::FONT_HERSHEY_SIMPLEX,1.0,inst.color,2);
            }*/

            if(cfg::dataset == DatasetType::kKitti){
                float rows_offset = img.rows /2;
                for(auto pt : inst.right_points){
                    pt.y+= rows_offset;
                    cv::circle(img, pt, 2, inst.color, 2);
                }
            }
            else{
                float cols_offset = img.cols /2;
                for(auto pt : inst.right_points){
                    pt.x+= cols_offset;
                    cv::circle(img, pt, 2, inst.color, 2);
                }
            }
        }

        /*for(const auto& pt : visual_new_points){
            cv::circle(img, pt, 3, cv::Scalar(0,0,255), 3);
        }*/

    }


}







}