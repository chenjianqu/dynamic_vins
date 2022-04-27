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

#include <opencv2/cudaimgproc.hpp>

#include "front_end_parameters.h"
#include "front_end.h"
#include "utils/dataset/viode_utils.h"

namespace dynamic_vins{\


FeatureTracker::FeatureTracker(const string& config_path)
{
    fe_para::SetParameters(config_path);

    n_id = 0;
    Debugt("init FeatureTracker");

    lk_optical_flow = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30);
    lk_optical_flow_back = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30, true);

    if(cfg::slam == SlamType::kDynamic){
        insts_tracker.reset(new InstsFeatManager(config_path));
    }
}




FeatureMap FeatureTracker::TrackImage(SegImage &img)
{
    TicToc t_r,tt;
    cur_time = img.time0;
    cur_img = img;
    row = img.color0.rows;
    col = img.color0.cols;

    cur_pts.clear();

    if (!prev_pts.empty()){
        vector<uchar> status = FeatureTrackByLK(prev_img.gray0, img.gray0, prev_pts, cur_pts);
        ReduceVector(prev_pts, status);
        ReduceVector(cur_pts, status);
        ReduceVector(ids, status);
        ReduceVector(track_cnt, status);
    }
    for (auto &n : track_cnt) n++;

    Infot("TrackImage | flowTrack:{} ms", tt.TocThenTic());

    //RejectWithF();
    TicToc t_m;
    SortPoints(cur_pts, track_cnt, ids);
    mask = cv::Mat(cur_img.gray0.rows,cur_img.gray0.cols,CV_8UC1,cv::Scalar(255));
    for(const auto& pt : cur_pts) cv::circle(mask, pt, fe_para::kMinDist, 0, -1);
    TicToc t_t;
    int n_max_cnt = fe_para::kMaxCnt - static_cast<int>(cur_pts.size());
    if (n_max_cnt > 0)
        cv::goodFeaturesToTrack(cur_img.gray0, n_pts, fe_para::kMaxCnt - cur_pts.size(),
                                0.01, fe_para::kMinDist, mask);
    else
        n_pts.clear();

    for (auto &p : n_pts){
        cur_pts.push_back(p);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }

    Infot("TrackImage | goodFeaturesToTrack:{} ms", tt.TocThenTic());

    cur_un_pts = UndistortedPts(cur_pts, cam0);
    pts_velocity = PtsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    Infot("TrackImage | un&&vel:{} ms", tt.TocThenTic());

    if(cfg::is_stereo && !cur_img.gray1.empty())
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if(!cur_pts.empty())
        {
            vector<uchar> status= FeatureTrackByLK(cur_img.gray0, cur_img.gray1, cur_pts, cur_right_pts);

            ids_right = ids;
            ReduceVector(cur_right_pts, status);
            ReduceVector(ids_right, status);
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            ReduceVector(pts_velocity, status);
            */
            cur_un_right_pts = UndistortedPts(cur_right_pts, cam1);
            right_pts_velocity = PtsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        prev_un_right_pts_map = cur_un_right_pts_map;

        Infot("TrackImage | flowTrack right:{} ms", tt.TocThenTic());
    }

    if(fe_para::is_show_track)
        DrawTrack(cur_img, ids, cur_pts, cur_right_pts, prev_left_map);

    Infot("TrackImage | DrawTrack right:{} ms", tt.TocThenTic());

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;

    prev_left_map.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prev_left_map[ids[i]] = cur_pts[i];

    return SetOutputFeats();
}


FeatureMap FeatureTracker::SetOutputFeats()
{
    FeatureMap fm;
    //left cam
    for (size_t i = 0; i < ids.size(); i++){
        constexpr int camera_id = 0;
        Vec7d xyz_uv_velocity;
        xyz_uv_velocity << cur_un_pts[i].x, cur_un_pts[i].y, 1,
        cur_pts[i].x, cur_pts[i].y,
        pts_velocity[i].x, pts_velocity[i].y;
        fm[ids[i]].emplace_back(camera_id,  xyz_uv_velocity);
        //Debugt("id:{} cam:{}",ids[i],camera_id);
    }
    //stereo
    if (cfg::is_stereo && !cur_img.gray1.empty()){
        for (size_t i = 0; i < ids_right.size(); i++){
            constexpr int camera_id = 1;
            Vec7d xyz_uv_velocity;
            xyz_uv_velocity << cur_un_right_pts[i].x, cur_un_right_pts[i].y, 1,
            cur_right_pts[i].x, cur_right_pts[i].y,
            right_pts_velocity[i].x, right_pts_velocity[i].y;
            fm[ids_right[i]].emplace_back(camera_id,  xyz_uv_velocity);
            //Debugt("id:{} cam:{}",ids_right[i],camera_id);
        }
    }
    return fm;
}



FeatureMap FeatureTracker::TrackImageNaive(SegImage &img)
{
    TicToc t_r,tt;
    cur_time = img.time0;
    cur_img = img;
    row = img.color0.rows;
    col = img.color0.cols;

    cur_pts.clear();

    cur_img.gray0_gpu.download(cur_img.gray0);
    cur_img.gray1_gpu.download(cur_img.gray1);

    Debugt("TrackImageNaive | input mask:{}", DimsToStr(img.inv_merge_mask_gpu.size()));

    ///形态学运算
    if(cur_img.exist_inst){
        static auto erode_kernel = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(10,10),cv::Point(-1,-1));
        static auto erode_filter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE,CV_8UC1,erode_kernel);
        erode_filter->apply(img.inv_merge_mask_gpu,img.inv_merge_mask_gpu);
        img.inv_merge_mask_gpu.download(img.inv_merge_mask);
    }

    ///特征点跟踪
    if (!prev_pts.empty())
    {
        Debugt("trackImageNaive | prev_pts.size:{}", prev_pts.size());
        vector<uchar> status;
        if(cfg::use_background_flow){
            status = FeatureTrackByDenseFlow(cur_img.flow, prev_pts, cur_pts);
        }
        else{
            status = FeatureTrackByLKGpu(lk_optical_flow, lk_optical_flow_back, prev_img.gray0_gpu,
                                                      cur_img.gray0_gpu,prev_pts, cur_pts);
        }

        if(cur_img.exist_inst){
            for(int i=0;i<status.size();++i){
                if(status[i] && cur_img.inv_merge_mask.at<uchar>(cur_pts[i]) == 0 )
                    status[i]=0;
            }
        }

        ReduceVector(prev_pts, status);
        ReduceVector(cur_pts, status);
        ReduceVector(ids, status);
        ReduceVector(track_cnt, status);
        Debugt("trackImageNaive | cur_pts.size:{}", cur_pts.size());
    }
    Infot("trackImageNaive | flowTrack left:{} ms", tt.TocThenTic());

    for (auto &n : track_cnt) n++;
    //RejectWithF();
    TicToc t_m;
    TicToc t_t;

    ///特征点检测
    SortPoints(cur_pts, track_cnt, ids);
    if(cur_img.exist_inst)
        mask = cur_img.inv_merge_mask.clone();
    else
        mask = cv::Mat(cur_img.color0.rows,cur_img.color0.cols,CV_8UC1,cv::Scalar(255));

    for(const auto& pt : cur_pts)
        cv::circle(mask, pt, fe_para::kMinDist, 0, -1);
    //cv::threshold(mask,mask,128,255,cv::THRESH_BINARY);
    mask_gpu.upload(mask);

    if (int n_max_cnt = fe_para::kMaxCnt - (int)cur_pts.size(); n_max_cnt > 10){
        Warnt("trackImageNaive | n_max_cnt:{}", n_max_cnt);
        n_pts = DetectShiTomasiCornersGpu(n_max_cnt, cur_img.gray0_gpu, mask_gpu);
        //n_pts = detectNewFeaturesGPU(n_max_cnt,cur_img.gray0_gpu,mask);
        visual_new_pts = n_pts;
        for (auto &p : n_pts){
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
        Debugt("trackImageNaive | cur_pts.size:{}", cur_pts.size());
    }
    else{
        n_pts.clear();
    }

    Infot("trackImageNaive | detect feature:{} ms", tt.TocThenTic());

    ///特征点矫正和计算速度
    cur_un_pts = UndistortedPts(cur_pts, cam0);
    pts_velocity = PtsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    Infot("trackImageNaive | vel&&un:{} ms", tt.TocThenTic());

    ///右图像跟踪
    if(cfg::is_stereo && (!cur_img.gray1.empty() || !cur_img.gray1_gpu.empty()) && !cur_pts.empty())
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        Debugt("trackImageNaive | flowTrack right start");
        //std::vector<uchar> status= flowTrack(cur_img.gray0,cur_img.gray1,cur_pts, cur_right_pts);
        vector<uchar> status= FeatureTrackByLKGpu(lk_optical_flow, lk_optical_flow_back, cur_img.gray0_gpu,
                                                  cur_img.gray1_gpu,
                                                  cur_pts, cur_right_pts);
        Debugt("trackImageNaive | flowTrack right finish");
        if(cfg::dataset == DatasetType::kViode){
            for(int i=0;i<(int)status.size();++i){
                if(status[i] && VIODE::IsDynamic(cur_right_pts[i], cur_img.seg1))
                    status[i]=0;
            }
        }
        else if(!cur_img.seg1.empty()){
            for(int i=0;i<(int)status.size();++i){
                if(status[i] && cur_img.seg1.at<uchar>(cur_right_pts[i]) >0 )
                    status[i]=0;
            }
        }
        ids_right = ids;
        ReduceVector(cur_right_pts, status);
        ReduceVector(ids_right, status);
        // only keep left-right pts
        /*
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ReduceVector(cur_un_pts, status);
        reduceVector(pts_velocity, status);
        */
        cur_un_right_pts = UndistortedPts(cur_right_pts, cam1);
        right_pts_velocity = PtsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        prev_un_right_pts_map = cur_un_right_pts_map;

        Debugt("trackImageNaive | cur_right_pts.size:{}", cur_right_pts.size());
        Infot("trackImageNaive | flow_track right:{} ms", tt.TocThenTic());
    }

    if(fe_para::is_show_track)
        DrawTrack(cur_img, ids, cur_pts, cur_right_pts, prev_left_map);
    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;

    prev_left_map.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prev_left_map[ids[i]] = cur_pts[i];

    return SetOutputFeats();
}



void FeatureTracker::RejectWithF()
{
/*    if (cur_pts.size() >= 8)
    {
        Debugt("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++){
            Eigen::Vector3d tmp_p;
            m_camera[0]->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
            tmp_p.x() = kFocalLength * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = kFocalLength * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = kFocalLength * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = kFocalLength * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }
        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, cfg::kFThreshold, 0.99, status);
        size_t size_a = cur_pts.size();
        ReduceVector(prev_pts, status);
        ReduceVector(cur_pts, status);
        ReduceVector(cur_un_pts, status);
        ReduceVector(ids, status);
        ReduceVector(track_cnt, status);
        Debugt("FM ransac: {} -> {}:{}", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        Debugt("FM ransac costs: {} ms", t_f.Toc());
    }*/
}



void FeatureTracker::ShowUndistortion(const string &name)
{
/*    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++){
        for (int j = 0; j < row; j++){
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.emplace_back(b.x() / b.z(), b.y() / b.z());
        }
    }

    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * kFocalLength + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * kFocalLength + row / 2;
        pp.at<float>(2, 0) = 1.0;
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 &&
        pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600){
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) =
                    cur_img.gray0.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);*/
}




vector<cv::Point2f> FeatureTracker::PtsVelocity(vector<int> &id_vec,
                                                vector<cv::Point2f> &pts,
                                                std::map<int, cv::Point2f> &cur_id_pts,
                                                std::map<int, cv::Point2f> &prev_id_pts) const{
    vector<cv::Point2f> pts_vel;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < id_vec.size(); i++){
        cur_id_pts.insert({id_vec[i], pts[i]});
    }
    // caculate points velocity
    if (!prev_id_pts.empty()){
        double dt = cur_time - prev_time;
        for (unsigned int i = 0; i < pts.size(); i++){
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(id_vec[i]);
            if (it != prev_id_pts.end()){
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_vel.emplace_back(v_x, v_y);
            }
            else{
                pts_vel.emplace_back(0, 0);
            }
        }
    }
    else{
        for (unsigned int i = 0; i < cur_pts.size(); i++){
            pts_vel.emplace_back(0, 0);
        }
    }
    return pts_vel;
}



void FeatureTracker::DrawTrack(const SegImage &img,
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts,
                               vector<cv::Point2f> &curRightPts,
                               std::map<int, cv::Point2f> &prevLeftPts){
    if(!img.inv_merge_mask_gpu.empty()){
        cv::cuda::GpuMat img_show_gpu;
        cv::cuda::cvtColor(img.inv_merge_mask_gpu,img_show_gpu,CV_GRAY2BGR);
        //cv::cuda::cvtColor(mask_gpu,img_show_gpu,CV_GRAY2BGR);
        cv::cuda::scaleAdd(img_show_gpu,0.5,img.color0_gpu,img_show_gpu);
        img_show_gpu.download(img_track_);
    }
    else{
        img_track_ = img.color0;
    }

    if (cfg::is_stereo && !img.color1.empty()){
        if(cfg::dataset == DatasetType::kKitti){
            cv::vconcat(img_track_, img.color1, img_track_);
        }else{
            cv::hconcat(img_track_, img.color1, img_track_);
        }
    }
    //cv::cvtColor(img_track, img_track, CV_GRAY2RGB);
    for (size_t j = 0; j < curLeftPts.size(); j++){
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(img_track_, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    for (auto & pt : visual_new_pts)
        cv::circle(img_track_, pt, 2, cv::Scalar(255, 255, 255), 2);
    if (cfg::is_stereo && !img.color1.empty() ){
        if(cfg::dataset == DatasetType::kKitti){
            for (auto &rightPt : curRightPts){
                rightPt.y += (float)img.color0.rows;
                cv::circle(img_track_, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            }
        }
        else{
            for (auto &rightPt : curRightPts){
                rightPt.x += (float)img.color0.cols;
                cv::circle(img_track_, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            }
        }
    }
    for (size_t i = 0; i < curLeftIds.size(); i++){
        if(auto it = prevLeftPts.find(curLeftIds[i]); it != prevLeftPts.end()){
            cv::arrowedLine(img_track_, curLeftPts[i], it->second, cv::Scalar(0, 255, 0),
                            1, 8, 0, 0.2);
        }
    }
}


void FeatureTracker::SetPrediction(std::map<int, Eigen::Vector3d> &predictPts)
{
/*    predict_pts.clear();
    predict_pts_debug.clear();
    for (size_t i = 0; i < ids.size(); i++){
        int id = ids[i];
        if (auto it= predictPts.find(id); it != predictPts.end()){
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(it->second, tmp_uv);
            predict_pts.emplace_back(tmp_uv.x(), tmp_uv.y());
            predict_pts_debug.emplace_back(tmp_uv.x(), tmp_uv.y());
        }
        else{
            predict_pts.push_back(prev_pts[i]);
        }
    }*/
}


void FeatureTracker::RemoveOutliers(std::set<int> &removePtsIds)
{
    vector<uchar> status;
    for (int & id : ids){
        if(auto itSet = removePtsIds.find(id);itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }
    ReduceVector(prev_pts, status);
    ReduceVector(ids, status);
    ReduceVector(track_cnt, status);
}



FeatureMap FeatureTracker::TrackSemanticImage(SegImage &img)
{
    TicToc t_r,tt;
    cur_time = img.time0;
    cur_img = img;
    row = img.color0.rows;
    col = img.color0.cols;
    cur_pts.clear();

    //为了防止两个线程同时访问图片内存，这里进行复制
    cur_img.gray0_gpu = img.gray0_gpu.clone();
    cur_img.gray1_gpu = img.gray1_gpu.clone();

    cur_img.gray0_gpu.download(cur_img.gray0);
    cur_img.gray1_gpu.download(cur_img.gray1);

    ///开启另一个线程检测动态特征点
    std::thread t_inst_track;
    TicToc t_i;
    if(cfg::use_dense_flow){
        t_inst_track = std::thread(&InstsFeatManager::InstsFlowTrack, insts_tracker.get(), img);
    }
    else{
        t_inst_track = std::thread(&InstsFeatManager::InstsTrack, insts_tracker.get(), img);
    }
    //std::thread t_inst_track(&InstsFeatManager::InstsTrackByMatching, insts_tracker.get(), img);

    if(img.exist_inst){
        ErodeMaskGpu(cur_img.inv_merge_mask_gpu, cur_img.inv_merge_mask_gpu);///形态学运算
        cur_img.inv_merge_mask_gpu.download(cur_img.inv_merge_mask);
    }

    ///特征点跟踪
    if (!prev_pts.empty())
    {
        Debugt("TrackSemanticImage | prev_pts.size:{}", prev_pts.size());
        vector<uchar> status = FeatureTrackByLK(prev_img.gray0, cur_img.gray0, prev_pts, cur_pts);
        //vector<uchar> status=flowTrackGpu(lk_optical_flow,lk_optical_flow_back,prev_img.gray0_gpu,cur_img.gray0_gpu,prev_pts,cur_pts);
        //剔除落在动态区域上的特征点
        if(cur_img.exist_inst){
            for(int i=0;i<(int)status.size();++i)
                if(status[i] && cur_img.inv_merge_mask.at<uchar>(cur_pts[i]) < 123 )
                    status[i]=0;
        }
        int cnt=0;
        for(auto e:status)if(e)cnt++;
        ReduceVector(prev_pts, status);
        ReduceVector(cur_pts, status);
        ReduceVector(ids, status);
        ReduceVector(track_cnt, status);
        Debugt("TrackSemanticImage | cur_pts.size:{}", cur_pts.size());
    }
    Infot("TrackSemanticImage | flowTrack left:{} ms", tt.TocThenTic());

    for (auto &n : track_cnt) n++;

    //RejectWithF();
    TicToc t_m, t_t;

    ///检测新的特征点
    SortPoints(cur_pts, track_cnt, ids);
    if(cur_img.exist_inst)
        mask = cur_img.inv_merge_mask.clone();
    else
        mask = cv::Mat(cur_img.color0.rows,cur_img.color0.cols,CV_8UC1,cv::Scalar(255));
    for(const auto& pt : cur_pts)
        cv::circle(mask, pt, fe_para::kMinDist, 0, -1);
    //mask_gpu.upload(mask);

    if (int n_max_cnt = fe_para::kMaxCnt - (int)cur_pts.size(); n_max_cnt > 10){
        //n_pts = detectNewFeaturesGPU(n_max_cnt,cur_img.gray0_gpu,mask_gpu);
        cv::goodFeaturesToTrack(cur_img.gray0, n_pts, n_max_cnt, 0.01, fe_para::kMinDist, mask);
        visual_new_pts = n_pts;
        for (auto &p : n_pts){
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
        Debugt("TrackSemanticImage | cur_pts.size:{}", cur_pts.size());
    }
    else{
        n_pts.clear();
    }
    Infot("TrackSemanticImage | detect feature:{} ms", tt.TocThenTic());

    ///矫正特征点,并计算特征点的速度
    cur_un_pts = UndistortedPts(cur_pts, cam0);
    pts_velocity = PtsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);
    //Infot("TrackSemanticImage | vel&&un:{} ms", tt.TocThenTic());

    if(cfg::is_stereo && (!cur_img.gray1.empty() || !cur_img.gray1_gpu.empty()) && !cur_pts.empty())
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        //Debugt("TrackSemanticImage | flowTrack right start");

        std::vector<uchar> status= FeatureTrackByLK(cur_img.gray0, cur_img.gray1, cur_pts, cur_right_pts);
        //vector<uchar> status=flowTrackGpu(lk_optical_flow,lk_optical_flow_back,cur_img.gray0_gpu,cur_img.gray1_gpu,cur_pts,cur_right_pts);
        //Debugt("TrackSemanticImage | flowTrack right finish");
        if(cfg::dataset == DatasetType::kViode){
            for(int i=0;i<(int)status.size();++i)
                if(status[i] && VIODE::IsDynamic(cur_right_pts[i], cur_img.seg1))
                    status[i]=0;
        }
        else if(!cur_img.seg1.empty()){
            for(int i=0;i<(int)status.size();++i)
                if(status[i] && cur_img.seg1.at<uchar>(cur_right_pts[i]) >0 )
                    status[i]=0;
        }

        ids_right = ids;
        ReduceVector(cur_right_pts, status);
        ReduceVector(ids_right, status);
        // only keep left-right pts
        /*
        ReduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        reduceVector(cur_un_pts, status);
        reduceVector(pts_velocity, status);
        */
        cur_un_right_pts = UndistortedPts(cur_right_pts, cam1);
        right_pts_velocity = PtsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        prev_un_right_pts_map = cur_un_right_pts_map;
        Debugt("TrackSemanticImage | cur_right_pts.size:{}", cur_right_pts.size());
        Debugt("TrackSemanticImage | flow_track right:{}", tt.TocThenTic());
    }
    if(fe_para::is_show_track)
        DrawTrack(cur_img, ids, cur_pts, cur_right_pts, prev_left_map);

    t_inst_track.join();
    Infot("TrackSemanticImage 动态检测线程总时间:{} ms", t_i.TocThenTic());

    if(fe_para::is_show_track)
        insts_tracker->DrawInsts(img_track_);



    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;

    prev_left_map.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prev_left_map[ids[i]] = cur_pts[i];

    //cv::imshow("img_track",img_track);
    //cv::waitKey(1);

    return SetOutputFeats();

}


}
