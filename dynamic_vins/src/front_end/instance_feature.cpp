
//
// Created by chen on 2022/5/8.
//

#include "instance_feature.h"

#include "utils/def.h"
#include "utils/dataset/viode_utils.h"
#include "front_end_parameters.h"
#include "utils/dataset/coco_utils.h"
#include "utils/dataset/nuscenes_utils.h"

namespace dynamic_vins{\





/**
 * 计算特征点的像素速度
 */
void InstFeat::PtsVelocity(double dt){

    pts_velocity.clear();
    curr_id_pts.clear();

    for (unsigned int i = 0; i < ids.size(); i++){
        curr_id_pts.insert({ids[i], curr_un_points[i]});
    }

    // caculate points velocity
    if (!prev_id_pts.empty()){
        for (unsigned int i = 0; i < curr_un_points.size(); i++){
            auto it = prev_id_pts.find( ids[i]);
            if (it != prev_id_pts.end()){
                double v_x = (curr_un_points[i].x - it->second.x) / dt;
                double v_y = (curr_un_points[i].y - it->second.y) / dt;
                pts_velocity.emplace_back(v_x, v_y);
            }
            else{
                pts_velocity.emplace_back(0, 0);
            }
        }
    }
    else{
        for (unsigned int i = 0; i < curr_un_points.size(); i++){
            pts_velocity.emplace_back(0, 0);
        }
    }


}


void InstFeat::RightPtsVelocity(double dt){

    right_pts_velocity.clear();
    right_curr_id_pts.clear();

    for (unsigned int i = 0; i < right_ids.size(); i++){
        right_curr_id_pts.insert({right_ids[i], right_un_points[i]});
    }

    // caculate points velocity
    if (!right_prev_id_pts.empty()){
        for (unsigned int i = 0; i < right_un_points.size(); i++){
            auto it = right_prev_id_pts.find( right_ids[i]);
            if (it != right_prev_id_pts.end()){
                double v_x = (right_un_points[i].x - it->second.x) / dt;
                double v_y = (right_un_points[i].y - it->second.y) / dt;
                right_pts_velocity.emplace_back(v_x, v_y);
            }
            else{
                right_pts_velocity.emplace_back(0, 0);
            }
        }
    }
    else{
        for (unsigned int i = 0; i < right_un_points.size(); i++){
            right_pts_velocity.emplace_back(0, 0);
        }
    }

}


/**
 * 对特征点进行去畸变,返回归一化特征点
 * @param pts
 * @param cam
 * @return
 */
void InstFeat::UndistortedPts(PinHoleCamera::Ptr &cam)
{
    curr_un_points.clear();
    for (auto & pt : curr_points){
        Eigen::Vector2d a(pt.x, pt.y);
        Eigen::Vector3d b;
        cam->LiftProjective(a, b);//将特征点反投影到归一化平面，并去畸变
        curr_un_points.emplace_back(b.x() / b.z(), b.y() / b.z());
    }
}

void InstFeat::RightUndistortedPts(PinHoleCamera::Ptr &cam)
{
    right_un_points.clear();
    for (auto & pt : right_points){
        Eigen::Vector2d a(pt.x, pt.y);
        Eigen::Vector3d b;
        cam->LiftProjective(a, b);//将特征点反投影到归一化平面，并去畸变
        right_un_points.emplace_back(b.x() / b.z(), b.y() / b.z());
    }
}


void InstFeat::TrackLeft(SemanticImage &img,SemanticImage &prev_img,bool dense_flow){
    if(last_points.empty())
        return;
    curr_points.clear();
    //Debugt("inst:{} last_points:{} mask({}x{},type:{})",  id, last_points.size(), mask_img.rows, mask_img.cols, mask_img.type());
    //光流跟踪
    vector<uchar> status;
    if(dense_flow){
        status = FeatureTrackByDenseFlow(img.flow,last_points, curr_points);
    }
    else{
        status = FeatureTrackByLK(prev_img.gray0,img.gray0,last_points,curr_points);
    }
    if(!mask_img.empty()){
        for(size_t i=0;i<status.size();++i){
            if(status[i] && mask_img.at<uchar>(curr_points[i]) == 0)
                status[i]=0;
        }
    }
    //删除跟踪失败的点
    ReduceVector(curr_points, status);
    ReduceVector(ids, status);
    ReduceVector(last_points, status);
    ReduceVector(track_cnt, status);

    for (auto &n : track_cnt)
        n++;
}


void InstFeat::TrackLeftGPU(SemanticImage &img,SemanticImage &prev_img,
                  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_forward,
                  cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_backward){
    if(last_points.empty())
        return;
    curr_points.clear();

    if(prev_img.gray0_gpu.empty()){
        prev_img.gray0_gpu.upload(prev_img.gray0);
    }
    if(img.gray0_gpu.empty()){
        img.gray0_gpu.upload(img.gray0);
    }

    vector<uchar> status = FeatureTrackByLKGpu(lk_forward, lk_backward,
                                               prev_img.gray0_gpu,
                                               img.gray0_gpu,last_points, curr_points);

    if(!mask_img.empty()){
        for(size_t i=0;i<status.size();++i){
            if(status[i] && mask_img.at<uchar>(curr_points[i]) == 0)
                status[i]=0;
        }
    }
    //删除跟踪失败的点
    ReduceVector(curr_points, status);
    ReduceVector(ids, status);
    ReduceVector(last_points, status);
    ReduceVector(track_cnt, status);

    for (auto &n : track_cnt)
        n++;
}



void InstFeat::TrackRight(SemanticImage &img){
    if(curr_points.empty())
        return;
    right_points.clear();

    auto status= FeatureTrackByLK(img.gray0,img.gray1,curr_points,right_points);
    if(cfg::dataset == DatasetType::kViode){
        for(size_t i=0;i<status.size();++i){
            if(status[i] && VIODE::PixelToKey(right_points[i], img.seg1) != id )
                status[i]=0;
        }
    }
    else{
        ///TODO
    }
    right_ids = ids;
    ReduceVector(right_points, status);
    ReduceVector(right_ids, status);
}

void InstFeat::TrackRightGPU(SemanticImage &img,
                   cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_forward,
                   cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> lk_backward){

    if(curr_points.empty())
        return;

    right_points.clear();

    if(img.gray0_gpu.empty()){
        img.gray0_gpu.upload(img.gray0);
    }
    if(img.gray1_gpu.empty()){
        img.gray1_gpu.upload(img.gray1);
    }

    vector<uchar> status = FeatureTrackByLKGpu(lk_forward, lk_backward,
                                               img.gray0_gpu,
                                               img.gray1_gpu,curr_points, right_points);

    if(cfg::dataset == DatasetType::kViode){
        for(size_t i=0;i<status.size();++i){
            if(status[i] && VIODE::PixelToKey(right_points[i], img.seg1) != id )
                status[i]=0;
        }
    }
    else{
        ///TODO
    }
    right_ids = ids;
    ReduceVector(right_points, status);
    ReduceVector(right_ids, status);
}


/**
 * 根据track_cnt对特征点、id进行重新排序
 * @param cur_pts
 * @param track_cnt
 * @param ids
 */
void InstFeat::SortPoints()
{
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    for (size_t i = 0; i < curr_points.size(); i++)
        cnt_pts_id.emplace_back(track_cnt[i], std::make_pair(curr_points[i], ids[i]));

    //排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b){
        return a.first > b.first;
    });

    curr_points.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &[t_cnt,pt_id] : cnt_pts_id){
        auto &[pt,fid]=pt_id;
        curr_points.push_back(pt);
        ids.push_back(fid);
        track_cnt.push_back(t_cnt);
    }
}


/**
 * 检测新的特征点
 * @param img
 * @param use_gpu
 */
void InstFeat::DetectNewFeature(SemanticImage &img,bool use_gpu,const cv::Mat &mask){
    int n_max_cnt = fe_para::kMaxCnt - (int)curr_points.size();
    if ( n_max_cnt < 10){
        return;
    }
    Debugt("DetectNewFeature | id:{} n_max_cnt:{}",id, n_max_cnt);

    /// 设置mask

    cv::Mat mask_detect;
    if(mask.empty()){
        if(mask_img.empty()){
            if(img.exist_inst)
                mask_img = img.inv_merge_mask.clone();
            else
                mask_img = cv::Mat(img.color0.rows,img.color0.cols,CV_8UC1,cv::Scalar(255));
        }

        for(const auto& pt : curr_points){
            cv::circle(mask_img, pt, fe_para::kMinDist, 0, -1);
        }

        mask_detect = mask_img;
    }
    else{
        mask_detect = mask;
    }


    ///特征检测
    vector<cv::Point2f > n_pts;
    if(use_gpu){
        if(img.gray0_gpu.empty()){
            img.gray0_gpu.upload(img.gray0);
        }

        cv::cuda::GpuMat mask_detect_gpu(mask_detect);

        n_pts = DetectShiTomasiCornersGpu(n_max_cnt, img.gray0_gpu, mask_detect_gpu);
    }
    else{
        cv::goodFeaturesToTrack(img.gray0, n_pts, n_max_cnt, 0.01,
                                fe_para::kMinDist, mask_detect);
    }

    ///保存特征
    for (auto &p : n_pts){
        curr_points.push_back(p);
        ids.push_back(global_id_count++);
        track_cnt.push_back(1);
        visual_new_points.push_back(p);
    }
}


void InstFeat::RemoveOutliers(std::set<unsigned int> &removePtsIds)
{
    vector<uchar> status;
    for (unsigned int id : ids){
        if(auto itSet = removePtsIds.find(id);itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }
    ReduceVector(last_points, status);
    ReduceVector(ids, status);
    ReduceVector(track_cnt, status);
}










}



