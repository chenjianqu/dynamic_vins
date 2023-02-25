/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "dynamic_tracker.h"

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/io/pcd_io.h>

#include "basic/semantic_image.h"
#include "basic/def.h"
#include "front_end_parameters.h"
#include "utils/dataset/coco_utils.h"
#include "basic/point_landmark.h"
#include "utils/convert_utils.h"

namespace dynamic_vins{\

using std::unordered_map;
using std::make_pair;
using std::map;
namespace idx = torch::indexing;


InstsFeatManager::InstsFeatManager(const string& config_path)
{
    std::array<int64_t, 2> orig_dim{int64_t(fe_para::kInputHeight), int64_t(fe_para::kInputWidth)};
    mot_tracker = std::make_unique<DeepSORT>(config_path,orig_dim);
    //orb_matcher_ = cv::DescriptorMatcher::create("BruteForce-Hamming");
}


void InstsFeatManager::ClearState(){
    ExecInst([&](unsigned int key, InstFeat& inst){
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
    });
}


void InstsFeatManager::BoxAssociate2Dto3D(std::vector<Box3D::Ptr> &boxes)
{
    std::optional<Vec3d> center=std::nullopt;

    Debugt("Start BoxAssociate2Dto3D()");
    string log_text="BoxAssociate2Dto3D:\n";

    vector<bool> match_vec(boxes.size(),false);
    for(auto &[inst_id,inst] : instances){
        if(!inst.is_curr_visible){
            continue;
        }

//        auto it=estimated_info.find(inst_id);
//        if(it!=estimated_info.end()){
//            auto& estimated_inst = it->second;
//            if(estimated_inst.is_init && estimated_inst.is_init_velocity){
//                ///将物体的位姿递推到当前时刻
//                double time_ij = curr_time - estimated_inst.time;
//                Mat3d Roioj=Sophus::SO3d::exp(estimated_inst.a*time_ij).matrix();
//                Vec3d Poioj=estimated_inst.v*time_ij;
//                Mat3d R_woi = Roioj * estimated_inst.R;
//                Vec3d P_woi = Roioj * estimated_inst.P + Poioj;
//                center = P_woi;
//                ///生成物体的各个3D顶点
//                //Mat38d corners =  Box3D::GetCornersFromPose(R_woi,P_woi,estimated_inst.dims);
//            }
//        }

        vector<Box3D::Ptr> candidate_match;
        vector<int> candidate_idx;

        double max_iou=0;
        int max_idx=-1;
        for(size_t i=0;i<boxes.size();++i){
            if(match_vec[i])
                continue;

            cv::Rect inst_rect(inst.box2d->min_pt,inst.box2d->max_pt);
            cv::Rect proj_rect(boxes[i]->box2d.min_pt,boxes[i]->box2d.max_pt);
            float iou = Box2D::IoU(inst_rect,proj_rect);

            if(iou>0){
                log_text += fmt::format("inst:{}-box:{},name:({},{}),iou:{}\n",inst_id,i,inst.box2d->class_name,
                                        boxes[i]->class_name,iou);
            }

            ///类别要一致
            if(inst.box2d->class_name != boxes[i]->class_name)
                continue;

            ///判断3D目标检测得到的box 和 估计的3D box的距离
            if(center && (*center - boxes[i]->center_pt).norm() > 10)
                continue;

            if(iou>0.1){
                candidate_match.push_back(boxes[i]);
                candidate_idx.push_back(i);
            }
            //if(iou > max_iou){
            //    max_idx = i;
            //    max_iou = iou;
            //}
        }

        double min_dist= std::numeric_limits<double>::max();
        int min_idx=-1;
        for(int i=0;i<candidate_match.size();++i){
            if(candidate_match[i]->center_pt.norm() < min_dist){
                min_dist=candidate_match[i]->center_pt.norm();
                min_idx = candidate_idx[i];
            }
        }

        //if(max_iou > 0.1){
        //    match_vec[max_idx]=true;
        //    inst.box3d = boxes[max_idx];
        //    Debugt("id:{} box2d:{} box3d:{}",inst_id,coco::CocoLabel[inst.class_id],
        //           NuScenes::GetClassName(boxes[max_idx]->class_id));
        //}

        if(!candidate_match.empty()){
            match_vec[min_idx]=true;
            inst.box3d = boxes[min_idx];

            //log_text += fmt::format("result : id:{} box2d:{} box3d:{}\n",inst_id,inst.box2d->class_name,
            //                        boxes[min_idx]->class_name);
        }
    }

    Debugt(log_text);
}


/**
 * 构建额外点，并进行处理
 * 里面执行点云滤波和分割
 */
void InstsFeatManager::ProcessExtraPoints(){

    using PointCloud=pcl::PointCloud<pcl::PointXYZ>;

    TicToc t_all;

    pcl::RadiusOutlierRemoval<pcl::PointXYZ> radius_filter;
    radius_filter.setRadiusSearch(0.5);
    radius_filter.setMinNeighborsInRadius(10);//一米内至少有10个点

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance (1.); //设置近邻搜索的搜索半径为1.0m
    ec.setMinClusterSize (10);//设置一个聚类需要的最少点数目为100
    ec.setMaxClusterSize (25000); //设置一个聚类需要的最大点数目为25000

    /*std::queue<pair<unsigned int,PointCloud::Ptr>> pc_queue;
    std::mutex queue_mutex;
    std::atomic_bool filter_finished=false;

    ///线程1：点云采样和滤波
    std::thread t_filter([&](){
        ExecInst([&](unsigned int key, InstFeat& inst){
            if(inst.is_curr_visible){
                Debugt("ProcessExtraPoints() start t_filter");

                ///检测额外点,构建点云
                inst.DetectExtraPoints(curr_img.disp);

                ///转换为PCL点云
                PointCloud::Ptr pc = EigenToPclXYZ(inst.extra_points3d);
                PointCloud::Ptr pc_filtered(new PointCloud);

                ///半径滤波
                radius_filter.setInputCloud(pc);
                radius_filter.filter(*pc_filtered);

                if(pc_filtered->empty() || pc_filtered->points.size()<5){
                    return;
                }

                {
                    std::unique_lock<std::mutex> lock(queue_mutex);
                    pc_queue.push({key,pc_filtered});
                }
                Debugt("ProcessExtraPoints() end t_filter");
            }
        });
        filter_finished = true;
    });

    ///线程2：聚类分割
    while(true){
        bool is_empty =false;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(pc_queue.empty()){
                is_empty=true;
            }
        }
        if(filter_finished && is_empty){
            break;
        }
        else if(is_empty){
            std::this_thread::sleep_for(5ms);
            continue;
        }


        pair<unsigned int,PointCloud::Ptr> key_pc;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            key_pc = pc_queue.front();
            pc_queue.pop();
        }

        Debugt("ProcessExtraPoints() start segmentation pc.size:{}",key_pc.second->points.size());


        ///聚类分割
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        ec.setSearchMethod (tree);//设置点云的搜索机制
        std::vector<pcl::PointIndices> cluster_indices;
        ec.setInputCloud (key_pc.second);
        ec.extract (cluster_indices);//从点云中提取聚类，并将点云索引保存在cluster_indices中

        if(cluster_indices.empty()){
            return;
        }

        ///选择第一个簇作为分割结果
        PointCloud::Ptr segmented_pc(new PointCloud);
        auto &indices = cluster_indices[0].indices;
        segmented_pc->points.reserve(indices.size());
        for(auto &index:indices){
            segmented_pc->points.push_back(key_pc.second->points[index]);
        }
        segmented_pc->width = segmented_pc->points.size();
        segmented_pc->height=1;
        segmented_pc->is_dense = true;

        Debugt("ProcessExtraPoints() put segmentation");

        ///将结果转换eigen
        instances_[key_pc.first].extra_points3d = PclToEigen<pcl::PointXYZ>(segmented_pc);

        Debugt("ProcessExtraPoints() end segmentation");
    }

    if(t_filter.joinable())
        t_filter.join();*/

    ExecInst([&](unsigned int key, InstFeat& inst){
        if(inst.is_curr_visible){
            Debugt("ProcessExtraPoints() start t_filter");

            ///检测额外点,构建点云
            inst.DetectExtraPoints(curr_img.disp);

            ///转换为PCL点云
            PointCloud::Ptr pc = EigenToPclXYZ(inst.extra_points3d);
            inst.extra_points3d.clear();//先清空
            pc->width = pc->points.size();
            pc->height=1;

            ///半径滤波
            PointCloud::Ptr pc_filtered(new PointCloud);
            radius_filter.setInputCloud(pc);
            radius_filter.filter(*pc_filtered);

            if(pc_filtered->empty() || pc_filtered->points.size()<5){
                return;
            }

            pc_filtered->width = pc_filtered->points.size();
            pc_filtered->height=1;
            Debugt("ProcessExtraPoints() end t_filter");

            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
            ec.setSearchMethod (tree);//设置点云的搜索机制
            std::vector<pcl::PointIndices> cluster_indices;
            ec.setInputCloud (pc_filtered);
            ec.extract (cluster_indices);//从点云中提取聚类，并将点云索引保存在cluster_indices中

            if(cluster_indices.empty()){
                return;
            }

            ///选择第一个簇作为分割结果
            PointCloud::Ptr segmented_pc(new PointCloud);
            auto &indices = cluster_indices[0].indices;
            segmented_pc->points.reserve(indices.size());
            for(auto &index:indices){
                segmented_pc->points.push_back(pc_filtered->points[index]);
            }
            segmented_pc->width = segmented_pc->points.size();
            segmented_pc->height=1;
            segmented_pc->is_dense = true;

            ///TODO DEBUG
//            if(key==1){
//                const string object_base_path =
//                        "/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/point_cloud_temp/";
//                const string save_path_raw = object_base_path+fmt::format(
//                        "{}_{}_0_raw.pcd",PadNumber(curr_img.seq,6),key);
//                pcl::io::savePCDFile(save_path_raw,*pc);
//                const string save_path_filtered = object_base_path+fmt::format(
//                        "{}_{}_1_filtered.pcd",PadNumber(curr_img.seq,6),key);
//                pcl::io::savePCDFile(save_path_filtered,*pc_filtered);
//                const string save_path_segmented = object_base_path+fmt::format(
//                        "{}_{}_2_segmented.pcd",PadNumber(curr_img.seq,6),key);
//                pcl::io::savePCDFile(save_path_segmented,*segmented_pc);
//            }

            ///将结果转换eigen
            inst.extra_points3d = PclToEigen<pcl::PointXYZ>(segmented_pc);
        }
    });



    Debugt("ProcessExtraPoints() used time:{} ms",t_all.Toc());
}



/**
 * 跟踪动态物体
 * @param img
 */
void InstsFeatManager::InstsTrack(SemanticImage img)
{
    TicToc tic_toc;
    curr_time=img.time0;
    curr_img = img;

    //将当前帧未观测到的box设置状态
    for(auto& [key,inst] : instances){
        if(!inst.is_curr_visible){
            inst.lost_num++;
        }
        else{
            inst.lost_num=0;
        }
    }
    Infot("instsTrack AddInstances:{} ms", tic_toc.TocThenTic());

    ///2d box和3d box关联
    if(cfg::use_det3d){
        BoxAssociate2Dto3D(img.boxes3d);
    }

    is_exist_inst_ = !img.boxes2d.empty();

    if(is_exist_inst_){
        ///形态学运算
        ErodeMaskGpu(img.merge_mask_gpu, img.merge_mask_gpu);
        img.merge_mask_gpu.download(img.merge_mask);

        ///开启另一个线程处理点云
        std::thread t_process_extra = std::thread(&InstsFeatManager::ProcessExtraPoints, this);

        ///对每个目标进行光流跟踪
        ExecInst([&](unsigned int key, InstFeat& inst){
            if(inst.is_curr_visible){

                if(inst.roi->prev_roi_gray.empty()){
                    Debugt("prev_roi_gray.empty()==true id:{}",inst.id);
                }
                if(inst.last_points.empty()){
                    Debugt("inst.last_points.empty()==true id:{}",inst.id);
                }

                if(inst.roi->prev_roi_gray.empty() || inst.last_points.empty()){
                    return;
                }
                ///对两张图像进行padding，使得两张图像大小一致
                auto [prev_roi_gray_padded,roi_gray_padded] = InstanceImagePadding(inst.roi->prev_roi_gray,
                                                                      inst.roi->roi_gray);

                 ///DEBUG
//                if(img.seq>8 && img.seq<=15){
//                    cv::Mat merge;
//                    cv::vconcat(prev_roi_gray_padded,roi_gray_padded,merge);
//                    const string save_gray_path = fmt::format("/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/{}_{}_gray.png",img.seq,inst.id);
//                    cv::imwrite(save_gray_path,inst.roi->prev_roi_gray);
//                    const string save_mask_path = fmt::format("/home/chen/ws/dynamic_ws/src/dynamic_vins/data/output/single_instances/{}_{}_mask.png",img.seq,inst.id);
//                    cv::imwrite(save_mask_path,inst.roi->mask_cv);
//                }

                //inst.TrackLeft(roi_gray_padded,prev_roi_gray_padded,inst.box2d->roi->mask_cv);
                inst.TrackLeft(roi_gray_padded,prev_roi_gray_padded);

                Debugt("instsTrack id:{} curr_size:{}",inst.id,inst.curr_points.size());
            }
        });
        Infot("instsTrack flowTrack:{} ms", tic_toc.TocThenTic());


        ///角点检测
        ExecInst([&](unsigned int key, InstFeat& inst){
            if( inst.curr_points.size()>= fe_para::kMaxDynamicCnt) return;
            int max_new_detect = (fe_para::kMaxDynamicCnt - (int)inst.curr_points.size());
            inst.visual_points_pair.clear();
            inst.visual_right_points_pair.clear();
            inst.visual_new_points.clear();

            ErodeMask(inst.roi->mask_cv,inst.roi->mask_cv,5);

            cv::Mat inst_mask = inst.roi->mask_cv.clone();
            for(size_t i=0;i<inst.curr_points.size();++i){
                inst.visual_points_pair.emplace_back(inst.last_points[i],inst.curr_points[i]);//用于可视化
                cv::circle(inst_mask, inst.curr_points[i], fe_para::kMinDynamicDist, 0, -1);
            }

            ///添加新的特征点
            vector<cv::Point2f> new_pts;
            cv::goodFeaturesToTrack(inst.roi->roi_gray, new_pts, max_new_detect, 0.01,
                                    fe_para::kMinDynamicDist, inst_mask);

            for(auto &pt:new_pts){
                inst.curr_points.emplace_back(pt);
                inst.ids.emplace_back(InstFeat::global_id_count++);
                inst.visual_new_points.emplace_back(pt);
                inst.track_cnt.push_back(1);
            }

            Debugt("instsTrack id:{} add corners:{}",inst.id,inst.visual_new_points.size());
        });

        for(auto& [key,inst] : instances){
            if(!inst.is_curr_visible)
                continue;
            ///去畸变和计算归一化坐标
            inst.UndistortedPointsWithAddOffset(cam_t.cam0,inst.curr_points,inst.curr_un_points);

            //inst.UndistortedPts(camera_);
            ///计算特征点的速度
            inst.PtsVelocity(curr_time - last_time);
        }

        Infot("instsTrack UndistortedPts & PtsVelocity:{} ms", tic_toc.TocThenTic());

        /// 右边相机图像的跟踪
        if((!img.gray1.empty() || !img.gray1_gpu.empty()) && cfg::is_stereo){
            ExecInst([&](unsigned int key, InstFeat& inst){
                if(!inst.is_curr_visible)
                    return;
                //inst.TrackRight(img);
                inst.TrackRightByPad(img);
                inst.RightUndistortedPts(cam_t.cam1);
                inst.RightPtsVelocity(curr_time - last_time);
            });
        }
        Infot("instsTrack track right:{} ms", tic_toc.TocThenTic());

        ManageInstances();


        t_process_extra.join();//等待线程结束

        ExecInst([&](unsigned int key, InstFeat& inst){
            inst.PostProcess();
        });

    }
    else{
        ManageInstances();

        ClearState();
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
    for(auto it=instances.begin(),it_next=it; it != instances.end(); it=it_next){
        it_next++;
        auto &inst=it->second;
        if(inst.lost_num ==0 && !inst.box2d){
            inst.lost_num++;
        }
        if(inst.lost_num > 0){
            inst.lost_num++;
            if(inst.lost_num > 3){ //删除该实例
                instances.erase(it);
            }
        }
    }
}


/**
 ** 用于将特征点传到VIO模块
 * @param result
 */
std::map<unsigned int,FeatureInstance> InstsFeatManager::Output()
{
    std::map<unsigned int,FeatureInstance> result;

    string log_text= fmt::format("GetOutputFeature:{}\n",prev_img.seq);

    ExecInst([&](unsigned int key, InstFeat& inst){
        if(inst.lost_num>0 ||  !inst.is_curr_visible){
            return;
        }

        FeatureInstance  features_map;
        features_map.color = inst.color;
        features_map.box2d = inst.box2d;
        features_map.box3d = inst.box3d;
        features_map.points = inst.extra_points3d;

        for(int i=0;i<(int)inst.curr_un_points.size();++i){
            FeaturePoint::Ptr feat=std::make_shared<FeaturePoint>();

            feat->point.x() = inst.curr_un_points[i].x;
            feat->point.y() = inst.curr_un_points[i].y;
            feat->point.z()=1;

            feat->vel.x()=inst.pts_velocity[i].x;
            feat->vel.y()=inst.pts_velocity[i].y;

            feat->disp = prev_img.disp.at<float>(inst.curr_points[i]);

            features_map.features.insert({inst.ids[i],feat});
        }

        int right_cnt=0;
        if(cfg::is_stereo){
            for(int i=0; i<(int)inst.right_un_points.size(); i++){
                auto r_id = inst.right_ids[i];
                auto it=features_map.features.find(r_id);
                if(it==features_map.features.end())
                    continue;

                it->second->is_stereo=true;
                it->second->point_right.x() = inst.right_un_points[i].x;
                it->second->point_right.y() = inst.right_un_points[i].y;
                it->second->point_right.z() = 1;
                it->second->vel_right.x() = inst.right_pts_velocity[i].x;
                it->second->vel_right.y() = inst.right_pts_velocity[i].y;
            }
        }
        result.insert({key,features_map});
        log_text += fmt::format("inst_id:{} class:{} l_track:{} r_track:{} extra3d:{}\n",
                                key,inst.box2d->class_name,
                                inst.curr_points.size(), inst.right_points.size(),inst.extra_points3d.size());
    });
    Debugt(log_text);

    return result;
}


/**
 * 根据语义标签，多线程设置mask
 * @param mask_img 原来的mask
 * @param semantic_img 语义标签图像
 */
void InstsFeatManager::AddViodeInstances(SemanticImage &img)
{
    cv::Mat seg = img.seg0;
    Debugt("addViodeInstancesBySegImg merge insts");
    for(auto &det_box : img.boxes2d){
        auto key = det_box->track_id;
        if(instances.count(key) == 0){
            InstFeat instanceFeature(key);
            instances.insert({key, instanceFeature});
        }
        auto &inst = instances[key];
        inst.box2d = det_box;
        inst.roi->mask_cv = det_box->roi->mask_cv;
        inst.roi->mask_gpu = det_box->roi->mask_gpu;
        inst.roi->roi_gray = det_box->roi->roi_gray;
        inst.roi->roi_gpu = det_box->roi->roi_gpu;

        //inst.color = img.seg0.at<cv::Vec3b>(inst.box2d->);
        inst.is_curr_visible=true;
    }
}


/**
 *
 * @param instInfo
 * @param inst_mask_tensor
 * @param inst_mask_area
 * @return
 */
std::tuple<int,float,float> InstsFeatManager::GetMatchInst(Box2D &instInfo, torch::Tensor &inst_mask_tensor)
{
    /*int h=(int)inst_mask_tensor.sizes()[0];
    int w=(int)inst_mask_tensor.sizes()[1];

    float inst_mask_area = inst_mask_tensor.sum(torch::IntArrayRef({0,1})).item().toFloat();
    cv::Point2f inst_center_pt = (instInfo.min_pt + instInfo.max_pt)/2.;

    int id_match=-1;
    float iou_max=0;
    for(const auto &[key, inst_j] : instances_){
        ///根据速度计算当前的物体
        cv::Point2i delta = inst_j.box_vel * inst_j.delta_time;
        auto curr_min_pt = cv::Point2i(inst_j.box2d->min_pt) + delta;
        auto curr_max_pt = cv::Point2i(inst_j.box2d->max_pt) + delta;
        cv::Point2f curr_center_pt = (curr_max_pt+curr_min_pt)/2.;

        auto delta_pt = curr_center_pt - inst_center_pt;
        float delta_pt_abs = std::abs(delta_pt.x)+std::abs(delta_pt.y);

        if(BoxIoU(instInfo.min_pt, instInfo.max_pt, curr_min_pt, curr_max_pt) > 0 || delta_pt_abs < 100){
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
                intersection_mask = inst_mask_tensor.index({idx::Slice(idx::None,h+delta.y),
                                                            idx::Slice(idx::None,w+delta.x)}) *
                        inst_j.mask_tensor.index({idx::Slice(-delta.y,idx::None),idx::Slice(-delta.x,idx::None)});
            }
            float intersection_area = intersection_mask.sum(torch::IntArrayRef({0,1})).item().toFloat();
            float iou = intersection_area / (inst_mask_area + inst_j.mask_area - intersection_area);
            if(iou > iou_max){
                iou_max = iou;
                id_match = (int)inst_j.id;
            }
        }
    }

    return {id_match,iou_max,inst_mask_area};*/
}

cv::Mat InstsFeatManager::AddInstancesByIoU(SemanticImage &img)
{
/*    double current_time = img.time0;
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
    return ~mask_background;*/
}


void InstsFeatManager:: AddInstancesByIouWithGPU(const SemanticImage &img)
{
/*    double current_time = img.time0;
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

    global_instance_id+= n_inst;*/
}


void InstsFeatManager:: AddInstancesByTracking(SemanticImage &img)
{
    double current_time = img.time0;
    int n_inst = (int)img.boxes2d.size();
    if(img.boxes2d.empty())
        return;
    auto trks = mot_tracker->update(img.boxes2d, img.color0);

    string log_text="MOT AddInstancesByTracking:\n";

    for(auto &det_box : trks){
        unsigned int id=det_box->track_id;
        auto it=instances.find(id);
        if(it == instances.end()){
            InstFeat inst_feat(id);
            inst_feat.box2d = det_box;
            inst_feat.roi = det_box->roi;
            inst_feat.is_curr_visible=true;
            instances.insert({id, inst_feat});
            log_text += fmt::format("Create inst:{} cls:{} min_pt:({},{}),max_pt:({},{})\n",
                                    id, det_box->class_name,inst_feat.box2d->min_pt.x,inst_feat.box2d->min_pt.y,
                                    inst_feat.box2d->max_pt.x,inst_feat.box2d->max_pt.y);
        }
        else{
            it->second.box2d = det_box;
            it->second.roi->mask_cv = det_box->roi->mask_cv;
            it->second.roi->mask_gpu = det_box->roi->mask_gpu;
            it->second.roi->roi_gray = det_box->roi->roi_gray;
            it->second.roi->roi_gpu = det_box->roi->roi_gpu;
            it->second.is_curr_visible=true;

            log_text += fmt::format("Update inst:{} cls:{} min_pt:({},{}),max_pt:({},{})\n",
                                    id, det_box->class_name, it->second.box2d->min_pt.x, it->second.box2d->min_pt.y,
                                    it->second.box2d->max_pt.x, it->second.box2d->max_pt.y);
        }
    }
    Debugt(log_text);
}


vector<uchar> InstsFeatManager::RejectWithF(InstFeat &inst, int col, int row) const
{
    vector<cv::Point2f> un_cur_pts(inst.curr_points.size()), un_prev_pts(inst.last_points.size());
    for (unsigned int i = 0; i < inst.curr_points.size(); i++){
        Eigen::Vector3d tmp_p;
        cam_t.cam0->liftProjective(Eigen::Vector2d(inst.curr_points[i].x, inst.curr_points[i].y), tmp_p);
        tmp_p.x() = kFocalLength * tmp_p.x() / tmp_p.z() + col / 2.0;
        tmp_p.y() = kFocalLength * tmp_p.y() / tmp_p.z() + row / 2.0;
        un_cur_pts[i] = cv::Point2f((float)(tmp_p.x()), (float)tmp_p.y());
        cam_t.cam0->liftProjective(Eigen::Vector2d(inst.last_points[i].x, inst.last_points[i].y), tmp_p);
        tmp_p.x() = kFocalLength * tmp_p.x() / tmp_p.z() + col / 2.0;
        tmp_p.y() = kFocalLength * tmp_p.y() / tmp_p.z() + row / 2.0;
        un_prev_pts[i] = cv::Point2f((float)tmp_p.x(), (float)tmp_p.y());
    }
    vector<uchar> status;
    cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, fe_para::kFThreshold,
                           0.99, status);
    return status;
}



/**
 * 实例特征点的可视化
 * @param img
 */
void InstsFeatManager::DrawInsts(cv::Mat& img)
{
    if(cfg::slam != SLAM::kDynamic)
        return;

    for(const auto &[id,inst]: instances){
        if(inst.lost_num>0 || !inst.is_curr_visible)
            continue;

        ///绘制2D边界框
        //cv::rectangle(img,inst.box2d->min_pt,inst.box2d->max_pt,inst.color,2);

        ///绘制新的点
//        for(const auto &pt : inst.visual_new_points){
//            cv::circle(img,
//                       cv::Point2f(pt.x+inst.box2d->rect.tl().x,pt.y+inst.box2d->rect.tl().y),
//                       2, cv::Scalar(255,0,0), -1);
//        }

        ///绘制额外点
        //for(const auto &pt:inst.extra_points){
        //    cv::circle(img, cv::Point2f(pt.x+inst.box2d->rect.tl().x,pt.y+inst.box2d->rect.tl().y),
        //               2, cv::Scalar(0,0,255), -1);
        //}

        ///绘制光流
        //cv::Scalar arrowed_color = inst.color*1.5;
        for(const auto &[pt1,pt2] : inst.visual_points_pair){
            //cv::circle(img, pt1, 2, cv::Scalar(255, 255, 255), 2);//上一帧的点
            //cv::arrowedLine(img, pt1, pt2, cv::Scalar(255, 255, 255), 1, 8, 0, 0.15);
            cv::circle(img,
                       cv::Point2f(pt2.x+inst.box2d->rect.tl().x,pt2.y+inst.box2d->rect.tl().y),
                       3, cv::Scalar(0,0,0), -1);
            cv::circle(img,
                       cv::Point2f(pt2.x+inst.box2d->rect.tl().x,pt2.y+inst.box2d->rect.tl().y),
                       2, inst.color, -1);//当前帧的点
        }

        //if(vel_map_.count(inst.id)!=0){
        //    auto anchor=inst.feats_center_pt;
        //    anchor.y += 40;
        //    double v_abs = vel_map_[inst.id].v.norm();
        //    cv::putText(img, fmt::format("v:{:.2f} m/s",v_abs),anchor,cv::FONT_HERSHEY_SIMPLEX,1.0,inst.color,2);
        //}

        ///绘制右相机点
        if(cfg::is_vertical_draw){
            float rows_offset = img.rows /2.;
            for(auto pt : inst.right_points){
                pt.y+= rows_offset;
                cv::circle(img, pt, 2, inst.color, -1);
            }
        }
        else{
            float cols_offset = img.cols /2.;
            for(auto pt : inst.right_points){
                pt.x+= cols_offset;
                cv::circle(img, pt, 2, inst.color, -1);
            }
        }

        ///绘制检测的3D边界框
        if(inst.box3d){
            //cv::rectangle(img,inst.box3d->box2d.min_pt,inst.box3d->box2d.max_pt,
            //              cv::Scalar(255,255,255),2);
            if(estimated_info.count(id)>0 && estimated_info[id].is_static){
                inst.box3d->VisCorners2d(img,cv::Scalar(255,255,255),cam_t.cam0);//绘制投影3D-2D框
            }
            else{
                inst.box3d->VisCorners2d(img,inst.color,cam_t.cam0);
            }
        }

        //std::string label=fmt::format("{}-{}",id,inst.curr_points.size() - inst.visual_new_points.size());
        std::string label=fmt::format("{}",id);
        //cv::putText(img, label, inst.box_center_pt, cv::FONT_HERSHEY_SIMPLEX, 1.0, inst.color, 2);
        DrawText(img, label, inst.color, inst.box2d->center_pt(), 1.0, 2, false);
    }


}







}