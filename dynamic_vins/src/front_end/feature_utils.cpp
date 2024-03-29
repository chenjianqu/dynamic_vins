/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_utils.h"

#include <random>
#include <chrono>
#include <map>

#include "front_end_parameters.h"
#include "utils/log_utils.h"

using std::pair;

namespace dynamic_vins{\

using Tensor = torch::Tensor;


/**
 * LK光流估计
 * @param img1
 * @param img2
 * @param pts1
 * @param pts2
 * @return
 */
std::vector<uchar> FeatureTrackByLK(const cv::Mat &img1, const cv::Mat &img2, vector<cv::Point2f> &pts1,
                                    vector<cv::Point2f> &pts2,bool flow_back){
    std::vector<uchar> status;
    std::vector<float> err;
    if(img1.empty() || img2.empty() || pts1.empty()){
        throw std::runtime_error("FeatureTrackByLK() input wrong, received at least one of parameter are empty");
    }
    //前向光流计算
    cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2,status, err,
                             cv::Size(21, 21), 3);

    //反向光流计算 判断之前光流跟踪的特征点的质量
    if(fe_para::is_flow_back){
        vector<uchar> reverse_status;
        std::vector<cv::Point2f> reverse_pts = pts1;
        cv::calcOpticalFlowPyrLK(img2, img1, pts2, reverse_pts,
                                 reverse_status, err, cv::Size(21, 21), 1,
                                 cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3);
        for(size_t i = 0; i < status.size(); i++){
            if(status[i] && reverse_status[i] && PointDistance(pts1[i], reverse_pts[i]) <= 0.5)
                status[i] = 1;
            else
                status[i] = 0;
        }
    }
    ///将落在图像外面的特征点的状态删除
    for (size_t i = 0; i < pts2.size(); ++i){
        if (status[i] && !InBorder(pts2[i], img2.rows, img2.cols))
            status[i] = 0;
    }

    return status;
}


/**
 * 执行光流跟踪（CUDA）
 * @param lkOpticalFlow
 * @param lkOpticalFlowBack
 * @param img_prev
 * @param img_next
 * @param pts_prev
 * @param pts_next
 * @param flow_back
 * @return
 */
std::vector<uchar> FeatureTrackByLKGpu(const cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>& lkOpticalFlow,
                                       const cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>& lkOpticalFlowBack,
                                       const cv::cuda::GpuMat &img_prev, const cv::cuda::GpuMat &img_next,
                                       std::vector<cv::Point2f> &pts_prev, std::vector<cv::Point2f> &pts_next,
                                       bool flow_back){
    if(img_prev.empty() || img_next.empty() || pts_prev.empty()){
        std::string msg="flowTrack() input wrong, received at least one of parameter are empty";
        Errort(msg);
        throw std::runtime_error(msg);
    }
    auto getValidStatusSize=[](const std::vector<uchar> &stu){
        int cnt=0;
        for(const auto s : stu) if(s)cnt++;
        return cnt;
    };

    std::vector<float> err;
    cv::cuda::GpuMat d_prevPts;
    Points2GpuMat(pts_prev, d_prevPts);
    cv::cuda::GpuMat d_nextPts;
    cv::cuda::GpuMat d_status;

    lkOpticalFlow->calc(img_prev,img_next,d_prevPts,d_nextPts,d_status);

    std::vector<uchar> status;
    GpuMat2Status(d_status, status);
    GpuMat2Points(d_nextPts, pts_next);
    int forward_success=getValidStatusSize(status);
    Debugt("flowTrackGpu forward success:{}", forward_success);

    //反向光流计算 判断之前光流跟踪的特征点的质量
    if(flow_back){
        cv::cuda::GpuMat d_reverse_status;
        cv::cuda::GpuMat d_reverse_pts = d_prevPts;
        lkOpticalFlowBack->calc(img_next,img_prev,d_nextPts,d_reverse_pts,d_reverse_status);
        std::vector<uchar> reverse_status;
        GpuMat2Status(d_reverse_status, reverse_status);

        std::vector<cv::Point2f> pts_prev_reverse;
        GpuMat2Points(d_reverse_pts, pts_prev_reverse);
        //constexpr float SAVE_RATIO=0.2f;
        //if(int inv_success = getValidStatusSize(reverse_status); inv_success*1.0 / forward_success > SAVE_RATIO){
        for(size_t i = 0; i < reverse_status.size(); i++){
            if(status[i] && reverse_status[i] && PointDistance(pts_prev[i], pts_prev_reverse[i]) <= 1.)
                status[i] = 1;
            else
                status[i] = 0;
        }
        Debugt("flowTrackGpu backward success:{}", getValidStatusSize(status));
        //}
        /*else{
            std::vector<std::tuple<int,float>> feats_dis(status.size());
            for(int i=0;i<status.size();++i){
                float d = distance(pts_prev[i], pts_prev_reverse[i]);
                feats_dis[i] = {i,d};
            }
            std::sort(feats_dis.begin(),feats_dis.end(),[](auto &a,auto &b){
                return std::get<1>(a) < std::get<1>(b);//根据dis低到高排序
            });
            const int SAVE_FEAT_NUM = forward_success * SAVE_RATIO;
            for(int i=0,cnt=0;i<status.size();++i){
                int j=std::get<0>(feats_dis[i]);
                if(status[j] && cnt<SAVE_FEAT_NUM){
                    cnt++;
                }
                else{
                    status[j]=0;
                }
            }
            Warnt("flowTrackGpu backward success:{},so save:{}",getValidStatusSize(reverse_status),getValidStatusSize(status));
        }*/
    }

    ///将落在图像外面的特征点的状态删除
    for (size_t i = 0; i < pts_next.size(); ++i){
        if (status[i] && !InBorder(pts_next[i], img_next.rows, img_next.cols))
            status[i] = 0;
    }
    Debugt("flowTrackGpu input:{} final_success:{}", status.size(), getValidStatusSize(status));
    return status;
}


/**
 * 叠加两个mask，结果写入到第一个maks中
 * @param mask1
 * @param mask2
 */
void SuperpositionMask(cv::Mat &mask1, const cv::Mat &mask2)
{
    for (int i = 0; i < mask1.rows; i++) {
        uchar* mask1_ptr=mask1.data+i*mask1.step;
        uchar* mask2_ptr=mask2.data+i*mask2.step;
        for (int j = 0; j < mask1.cols; j++) {
            if(mask1_ptr[0]==255 && mask2_ptr[0]<128){
                mask1_ptr[0]=0;
            }
            mask1_ptr+=1;
            mask2_ptr+=1;
        }
    }
}


/**
 * 对特征点进行去畸变,返回归一化特征点
 * @param pts
 * @param cam
 * @return
 */
vector<cv::Point2f> UndistortedPts(vector<cv::Point2f> &pts,camodocal::CameraPtr &cam)
{
    vector<cv::Point2f> un_pts;
    for (auto & pt : pts){
        Eigen::Vector2d a(pt.x, pt.y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);//将特征点反投影到归一化平面，并去畸变
        un_pts.emplace_back(b.x() / b.z(), b.y() / b.z());
    }
    return un_pts;
}


/**
 * 根据光流执行点跟踪
 * @param flow
 * @param pts1
 * @param pts2
 * @return
 */
vector<uchar> FeatureTrackByDenseFlow(cv::Mat &flow,
                                      vector<cv::Point2f> &pts1,
                                      vector<cv::Point2f> &pts2){
    assert(flow.type()==CV_32FC2);
    int n = pts1.size();
    vector<uchar> status(n);
    pts2.resize(n);
    for(int i=0;i<n;++i){
        auto delta = flow.at<cv::Vec2f>(pts1[i]);
        pts2[i].x = pts1[i].x + delta[0];
        pts2[i].y = pts1[i].y + delta[1];
        if(0 <= pts2[i].x && pts2[i].x <= flow.cols && 0 <= pts2[i].y && pts2[i].y <= flow.rows)
            status[i]=1;
        else
            status[i]=0;
    }
    return status;
}


/**
 * 栅格采样像素点
 * @param detect_num 采样数量
 * @param inst_mask mask
 * @param step 采样步长
 * @param rect 采样的ROI
 * @return
 */
std::vector<cv::Point2f> DetectRegularCorners(int detect_num, const cv::Mat &inst_mask,int step ,cv::Rect rect){
    int cnt=0;
    int row_start=0,row_end=inst_mask.rows,col_start=0,col_end=inst_mask.cols;
    if(!rect.empty()){
        row_start = rect.tl().y;
        col_start = rect.tl().x;
        row_end = rect.br().y;
        col_end = rect.br().x;
    }
    vector<cv::Point2f> vec;
    constexpr int kMinDist=3;
    for(int i=row_start;i<row_end;i+=step){
        for(int j=col_start;j<col_end;j+=step){
            if(inst_mask.at<uchar>(i,j) > 0){
                vec.emplace_back(j,i);
                cnt++;
            }
        }
    }

    std::shuffle(vec.begin(),vec.end(),
                 std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));

    return vector<cv::Point2f>(vec.begin(),vec.begin()+std::min(detect_num,(int)vec.size()));
}


/**
 * 计算特征点的像素速度
 * @param dt
 * @param ids
 * @param curr_un_pts
 * @param prev_id_pts 上一帧中的 ID_特征点
 * @param output_velocity
 */
void PtsVelocity(double dt,
                 vector<unsigned int> &ids,
                 vector<cv::Point2f> &curr_un_pts,
                 std::map<unsigned int, cv::Point2f> &prev_id_pts,
                 vector<cv::Point2f> &output_velocity){
    output_velocity.clear();
    if (!prev_id_pts.empty()){
        for (unsigned int i = 0; i < curr_un_pts.size(); i++){
            if (auto it = prev_id_pts.find(ids[i]);it != prev_id_pts.end()){
                double v_x = (curr_un_pts[i].x - it->second.x) / dt;
                double v_y = (curr_un_pts[i].y - it->second.y) / dt;
                output_velocity.emplace_back(v_x, v_y);
            }
            else{
                output_velocity.emplace_back(0, 0);
            }
        }
    }
    else{
        for(unsigned int i = 0; i < curr_un_pts.size(); i++)
            output_velocity.emplace_back(0, 0);
    }
}


/**
 * 根据track_cnt对特征点、id进行重新排序
 * @param cur_pts
 * @param track_cnt
 * @param ids
 */
void SortPoints(std::vector<cv::Point2f> &cur_pts, std::vector<int> &track_cnt, std::vector<unsigned int> &ids)
{
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;
    for (size_t i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.emplace_back(track_cnt[i], std::make_pair(cur_pts[i], ids[i]));
    ///根据根据次数进行排序
    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b){
        return a.first > b.first;
    });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &[t_cnt,pt_id] : cnt_pts_id){
        auto &[pt,id]=pt_id;
        cur_pts.push_back(pt);
        ids.push_back(id);
        track_cnt.push_back(t_cnt);
    }
}


/**
 * 检测Shi-Tomasi角点（CUDA）
 * @param detect_num
 * @param img
 * @param mask
 * @param min_dist
 * @return
 */
std::vector<cv::Point2f> DetectShiTomasiCornersGpu(int detect_num, const cv::cuda::GpuMat &img,
                                                   const cv::cuda::GpuMat &mask,int min_dist)
{
    auto detector = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, detect_num, 0.01, min_dist);
    cv::cuda::GpuMat d_new_pts;
    detector->detect(img,d_new_pts,mask);
    std::vector<cv::Point2f> points;
    GpuMat2Points(d_new_pts, points);
    return points;
}


/**
 * 根据整数映射一个颜色
 * @param n
 * @return
 */
cv::Scalar ColorMapping(int64_t n) {
    auto bit_get = [](int64_t x, int64_t i) {
        return x & (1 << i);
    };

    int64_t r = 0, g = 0, b = 0;
    int64_t i = n;
    for (int64_t j = 7; j >= 0; --j) {
        r |= bit_get(i, 0) << j;
        g |= bit_get(i, 1) << j;
        b |= bit_get(i, 2) << j;
        i >>= 3;
    }
    return cv::Scalar(b, g, r);
}


/**
 * 在图像上绘制文件
 * @param img 图像
 * @param str 文字
 * @param color 文字的颜色
 * @param pos 文字的位置
 * @param scale 文字的大小
 * @param thickness 文字的字宽
 * @param reverse
 */
void DrawText(cv::Mat &img, const std::string &str, const cv::Scalar &color, const cv::Point& pos,
              float scale, int thickness, bool reverse) {
    auto t_size = cv::getTextSize(str, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, nullptr);
    cv::Point bottom_left, upper_right;
    if (reverse) {
        upper_right = pos;
        bottom_left = cv::Point(upper_right.x - t_size.width, upper_right.y + t_size.height);
    } else {
        bottom_left = pos;
        upper_right = cv::Point(bottom_left.x + t_size.width, bottom_left.y - t_size.height);
    }

    cv::rectangle(img, bottom_left, upper_right, color, -1);
    cv::putText(img, str, bottom_left, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255),thickness);
}


/**
 * 将两个图像padding到相同大小
 * @param img1
 * @param img2
 * @return
 */
tuple<cv::Mat,cv::Mat> InstanceImagePadding(cv::Mat &img1,cv::Mat &img2){
    int rows = std::max(img1.rows,img2.rows);
    int cols = std::max(img1.cols,img2.cols);
    cv::Mat img1_padded,img2_padded;
    cv::copyMakeBorder(img1,img1_padded,0,rows-img1.rows,0,cols-img1.cols,cv::BORDER_CONSTANT,cv::Scalar(0));
    cv::copyMakeBorder(img2,img2_padded,0,rows-img2.rows,0,cols-img2.cols,cv::BORDER_CONSTANT,cv::Scalar(0));
    return {img1_padded,img2_padded};
}


}
