/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "feature_utils.h"

namespace dynamic_vins{\

using Tensor = torch::Tensor;

std::vector<uchar> FeatureTrackByLK(const cv::Mat &img1, const cv::Mat &img2, vector<cv::Point2f> &pts1, vector<cv::Point2f> &pts2)
{
    std::vector<uchar> status;
    std::vector<float> err;

    if(img1.empty() || img2.empty() || pts1.empty()){
        std::string msg="flowTrack() input wrong, received at least one of parameter are empty";
        tk_logger->error(msg);
        throw std::runtime_error(msg);
    }

    cv::calcOpticalFlowPyrLK(img1, img2, pts1, pts2,status, err, cv::Size(21, 21), 3);

    //反向光流计算 判断之前光流跟踪的特征点的质量
    if(Config::kFlowBack){
        vector<uchar> reverse_status;
        std::vector<cv::Point2f> reverse_pts = pts1;
        cv::calcOpticalFlowPyrLK(img2, img1, pts2, reverse_pts,
                                 reverse_status, err, cv::Size(21, 21), 1,
                                 cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);
        //cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 3);
        for(size_t i = 0; i < status.size(); i++){
            if(status[i] && reverse_status[i] && distance(pts1[i], reverse_pts[i]) <= 0.5)
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



std::vector<uchar> FeatureTrackByLKGpu(const cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>& lkOpticalFlow,
                                       const cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow>& lkOpticalFlowBack,
                                       const cv::cuda::GpuMat &img_prev, const cv::cuda::GpuMat &img_next,
                                       std::vector<cv::Point2f> &pts_prev, std::vector<cv::Point2f> &pts_next){
    if(img_prev.empty() || img_next.empty() || pts_prev.empty()){
        std::string msg="flowTrack() input wrong, received at least one of parameter are empty";
        tk_logger->error(msg);
        throw std::runtime_error(msg);
    }

    static auto getValidStatusSize=[](const std::vector<uchar> &stu){
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
    DebugT("flowTrackGpu forward success:{}", forward_success);

    //反向光流计算 判断之前光流跟踪的特征点的质量
    if(Config::kFlowBack){
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
            if(status[i] && reverse_status[i] && distance(pts_prev[i], pts_prev_reverse[i]) <= 1.)
                status[i] = 1;
            else
                status[i] = 0;
        }
        DebugT("flowTrackGpu backward success:{}", getValidStatusSize(status));
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
            tk_logger->warn("flowTrackGpu backward success:{},so save:{}",getValidStatusSize(reverse_status),getValidStatusSize(status));
        }*/

    }

    ///将落在图像外面的特征点的状态删除
    for (size_t i = 0; i < pts_next.size(); ++i){
        if (status[i] && !InBorder(pts_next[i], img_next.rows, img_next.cols))
            status[i] = 0;
    }

    DebugT("flowTrackGpu input:{} final_success:{}", status.size(), getValidStatusSize(status));

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
vector<cv::Point2f> UndistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
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



std::vector<cv::Point2f> DetectRegularCorners(int detect_num, const cv::Mat &inst_mask,
                                              std::vector<cv::Point2f> &curr_pts, cv::Rect rect){
    cv::Mat mask = inst_mask.clone();
    for(int i=0;i<curr_pts.size();++i){
        cv::circle(mask,curr_pts[i],5,0,-1);
    }
    int cnt=0;
    int row_start=0,row_end=mask.rows,col_start=0,col_end=mask.cols;
    if(!rect.empty()){
        row_start = rect.tl().y;
        col_start = rect.tl().x;
        row_end = rect.br().y;
        col_end = rect.br().x;
    }
    vector<cv::Point2f> output;
    constexpr int kMinDist=5;
    for(int i=row_start;i<row_end && cnt<detect_num;i+=kMinDist){
        for(int j=col_start;j<col_end && cnt<detect_num;j+=kMinDist){
            if(mask.at<uchar>(i,j) > 0){
                output.push_back(cv::Point2f(j,i));
                cnt++;
            }
        }
    }
    return output;
}




}