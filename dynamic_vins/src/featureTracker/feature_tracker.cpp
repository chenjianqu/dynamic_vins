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
#include "feature_tracker.h"
#include "utility/viode_utils.h"

namespace dynamic_vins{\


FeatureTracker::FeatureTracker()
{
    n_id = 0;
    Debugt("init FeatureTracker");
    insts_tracker.reset(new InstsFeatManager);
    lk_optical_flow = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30);
    lk_optical_flow_back = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30, true);
}


void FeatureTracker::SortPoints(std::vector<cv::Point2f> &cur_pts, std::vector<int> &track_cnt, std::vector<int> &ids)
{
    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.emplace_back(track_cnt[i], std::make_pair(cur_pts[i], ids[i]));

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


FeatureMap FeatureTracker::TrackImage(SegImage &img)
{
    TicToc t_r,tt;
    cur_time = img.time0;
    cur_img = img;
    row = img.color0.rows;
    col = img.color0.cols;
    /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
    cur_pts.clear();

    if (!prev_pts.empty()){
        vector<uchar> status = FeatureTrackByLK(prev_img.gray0, img.gray0, prev_pts, cur_pts);
        ReduceVector(prev_pts, status);
        ReduceVector(cur_pts, status);
        ReduceVector(ids, status);
        ReduceVector(track_cnt, status);
    }
    for (auto &n : track_cnt) n++;

    Infot("TrackImage | flowTrack:{}", tt.TocThenTic());

    //RejectWithF();
    TicToc t_m;
    SortPoints(cur_pts, track_cnt, ids);
    mask = cv::Mat(cur_img.gray0.rows,cur_img.gray0.cols,CV_8UC1,cv::Scalar(255));
    for(const auto& pt : cur_pts){
        cv::circle(mask, pt, cfg::kMinDist, 0, -1);
    }
    TicToc t_t;
    int n_max_cnt = cfg::kMaxCnt - static_cast<int>(cur_pts.size());
    if (n_max_cnt > 0)
        cv::goodFeaturesToTrack(cur_img.gray0, n_pts, cfg::kMaxCnt - cur_pts.size(),
                                0.01, cfg::kMinDist, mask);
    else
        n_pts.clear();

    for (auto &p : n_pts){
        cur_pts.push_back(p);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }

    Infot("TrackImage | goodFeaturesToTrack:{}", tt.TocThenTic());

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    Infot("TrackImage | un&&vel:{}", tt.TocThenTic());

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
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        prev_un_right_pts_map = cur_un_right_pts_map;

        Infot("TrackImage | flowTrack right:{}", tt.TocThenTic());
    }

    if(cfg::is_show_track)
        drawTrack(cur_img, ids, cur_pts, cur_right_pts, prev_left_map);

    Infot("TrackImage | drawTrack right:{}", tt.TocThenTic());

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
    for (size_t i = 0; i < ids.size(); i++){
        double x = cur_un_pts[i].x;
        double y = cur_un_pts[i].y;
        constexpr double z = 1;
        double p_u = cur_pts[i].x;
        double p_v = cur_pts[i].y;
        constexpr int camera_id = 0;
        double velocity_x = pts_velocity[i].x;
        double velocity_y = pts_velocity[i].y;
        Vec7d xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        fm[ids[i]].emplace_back(camera_id,  xyz_uv_velocity);
    }
    if (cfg::is_stereo && !cur_img.gray1.empty()){
        for (size_t i = 0; i < ids_right.size(); i++){
            double x = cur_un_right_pts[i].x;
            double y = cur_un_right_pts[i].y;
            constexpr double z = 1;
            double p_u = cur_right_pts[i].x;
            double p_v = cur_right_pts[i].y;
            constexpr int camera_id = 1;
            double velocity_x = right_pts_velocity[i].x;
            double velocity_y = right_pts_velocity[i].y;
            Vec7d xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            fm[ids_right[i]].emplace_back(camera_id,  xyz_uv_velocity);
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
    /*
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img, cur_img);
        if(!rightImg.empty())
            clahe->apply(rightImg, rightImg);
    }
    */
    cur_pts.clear();

    cur_img.gray0_gpu.download(cur_img.gray0);
    cur_img.gray1_gpu.download(cur_img.gray1);

    ///形态学运算
    Erode10Gpu(img.inv_merge_mask_gpu,img.inv_merge_mask_gpu);
    img.inv_merge_mask_gpu.download(img.inv_merge_mask);

    if (!prev_pts.empty())
    {
        Debugt("trackImageNaive | prev_pts.size:{}", prev_pts.size());
        //vector<uchar> status = flowTrack(prev_img.gray0,cur_img.gray0,prev_pts, cur_pts);
        vector<uchar> status= FeatureTrackByLKGpu(lk_optical_flow, lk_optical_flow_back, prev_img.gray0_gpu,
                                                  cur_img.gray0_gpu,
                                                  prev_pts, cur_pts);
        if(!cur_img.inv_merge_mask.empty()){
            for(int i=0;i<status.size();++i){
                if(status[i] && cur_img.inv_merge_mask.at<uchar>(cur_pts[i]) == 0 )
                    status[i]=0;
            }
        }
        else{
            tk_logger->error("cur_img.inv_merge_mask is empty");
        }
        ReduceVector(prev_pts, status);
        ReduceVector(cur_pts, status);
        ReduceVector(ids, status);
        ReduceVector(track_cnt, status);
        Debugt("trackImageNaive | cur_pts.size:{}", cur_pts.size());
    }

    Infot("trackImageNaive | flowTrack left:{}", tt.TocThenTic());


    for (auto &n : track_cnt) n++;


    //RejectWithF();
    TicToc t_m;
    TicToc t_t;


    SortPoints(cur_pts, track_cnt, ids);
    if(!cur_img.inv_merge_mask.empty())
        mask = cur_img.inv_merge_mask.clone();
    else
        mask = cv::Mat(cur_img.color0.rows,cur_img.color0.cols,CV_8UC1,cv::Scalar(255));

    for(const auto& pt : cur_pts){
        cv::circle(mask, pt, cfg::kMinDist, 0, -1);
    }

    //cv::threshold(mask,mask,128,255,cv::THRESH_BINARY);
    mask_gpu.upload(mask);

    if (int n_max_cnt = cfg::kMaxCnt - (int)cur_pts.size(); n_max_cnt > 10){
        //if(cur_img.gray0.empty()) cur_img.gray0_gpu.download(cur_img.gray0);
        //cv::goodFeaturesToTrack(cur_img.gray0, n_pts, n_max_cnt, 0.01, cfg::kMinDist, mask);
        /*debug_t("trackImageNaive | semantic_mask size:{}x{} type:{} ",semantic_mask.rows,semantic_mask.cols,semantic_mask.type());
        DebugT("trackImageNaive | mask size:{}x{} type:{} ",mask.rows,mask.cols,mask.type());
        debug_t("trackImageNaive | mask_gpu size:{}x{} type:{} ",mask_gpu.rows,mask_gpu.cols,mask_gpu.type());
        DebugT("trackImageNaive | img.gray0_gpu size:{}x{} type:{} ",cur_img.gray0_gpu.rows,cur_img.gray0_gpu.cols,cur_img.gray0_gpu.type());*/
        /*cv::imshow("mask",mask);
        cv::waitKey(1);*/

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
    else
        n_pts.clear();

    Infot("trackImageNaive | detect feature:{}", tt.TocThenTic());

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    Infot("trackImageNaive | vel&&un:{}", tt.TocThenTic());

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
        cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
        right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        prev_un_right_pts_map = cur_un_right_pts_map;

        Debugt("trackImageNaive | cur_right_pts.size:{}", cur_right_pts.size());
        Infot("trackImageNaive | flow_track right:{}", tt.TocThenTic());
    }


    if(cfg::is_show_track)
        drawTrack(cur_img, ids, cur_pts, cur_right_pts, prev_left_map);

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



void FeatureTracker::rejectWithF()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
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
        int size_a = cur_pts.size();
        ReduceVector(prev_pts, status);
        ReduceVector(cur_pts, status);
        ReduceVector(cur_un_pts, status);
        ReduceVector(ids, status);
        ReduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.Toc());
    }
}


void FeatureTracker::ReadIntrinsicParameter(const vector<string> &calib_file)
{
    for (const auto & i : calib_file){
        vio_logger->info(fmt::format("readIntrinsicParameter() Reading parameter of camera:{}", i));
        camodocal::CameraPtr camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(i);
        vio_logger->info(camera->parametersToString());
        m_camera.push_back(camera);
    }

    insts_tracker->set_camera(m_camera[0]);
    if (calib_file.size() == 2){
        insts_tracker->set_is_stereo(true);
        insts_tracker->set_right_camera(m_camera[1]);
    }
}



void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(row + 600, col + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < col; i++){
        for (int j = 0; j < row; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera[0]->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.emplace_back(b.x() / b.z(), b.y() / b.z());
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    }

    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * kFocalLength + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * kFocalLength + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 &&
        pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600){
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.gray0.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else{
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    // turn the following code on if you need
    // cv::imshow(name, undistortedImg);
    // cv::waitKey(0);
}


vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (auto & pt : pts){
        Vec2d a(pt.x, pt.y);
        Vec3d b;
        cam->liftProjective(a, b);
        un_pts.emplace_back(b.x() / b.z(), b.y() / b.z());
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &id_vec,
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



void FeatureTracker::drawTrack(const SegImage &img,
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

    for (size_t j = 0; j < visual_new_pts.size(); j++){
        cv::circle(img_track_, visual_new_pts[j], 2, cv::Scalar(255, 255, 255), 2);
    }

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
            cv::arrowedLine(img_track_, curLeftPts[i], it->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }
}




void FeatureTracker::setPrediction(std::map<int, Eigen::Vector3d> &predictPts)
{
    predict_pts.clear();
    predict_pts_debug.clear();
    std::map<int, Eigen::Vector3d>::iterator itPredict;
    for (size_t i = 0; i < ids.size(); i++)
    {
        //printf("prevLeftId size %d prevLeftPts size %d\n",(int)prevLeftIds.size(), (int)prevLeftPts.size());
        int id = ids[i];
        itPredict = predictPts.find(id);
        if (itPredict != predictPts.end())
        {
            Eigen::Vector2d tmp_uv;
            m_camera[0]->spaceToPlane(itPredict->second, tmp_uv);
            predict_pts.emplace_back(tmp_uv.x(), tmp_uv.y());
            predict_pts_debug.emplace_back(tmp_uv.x(), tmp_uv.y());
        }
        else
            predict_pts.push_back(prev_pts[i]);
    }
}


void FeatureTracker::removeOutliers(std::set<int> &removePtsIds)
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
    Warnt("----------Time : {} ----------", img.time0);

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

    bool is_exist_inst = !cur_img.insts_info.empty();

    ///开启另一个线程检测动态特征点
    TicToc t_i;
    //std::thread t_inst_track(&InstsFeatManager::InstsTrack, insts_tracker.get(), img);
    //std::thread t_inst_track(&InstsFeatManager::InstsFlowTrack, insts_tracker.get(), img);

    if(is_exist_inst){
        Erode10Gpu(cur_img.inv_merge_mask_gpu,cur_img.inv_merge_mask_gpu);///形态学运算
        cur_img.inv_merge_mask_gpu.download(cur_img.inv_merge_mask);
    }

    if (!prev_pts.empty())
    {
        Debugt("trackImageNaive | prev_pts.size:{}", prev_pts.size());
        vector<uchar> status = FeatureTrackByLK(prev_img.gray0, cur_img.gray0, prev_pts, cur_pts);
        //vector<uchar> status=flowTrackGpu(lk_optical_flow,lk_optical_flow_back,prev_img.gray0_gpu,cur_img.gray0_gpu,prev_pts,cur_pts);
        if(!cur_img.inv_merge_mask.empty()){
            for(int i=0;i<(int)status.size();++i){
                if(status[i] && cur_img.inv_merge_mask.at<uchar>(cur_pts[i]) == 0 )
                    status[i]=0;
            }
        }
        else{
            tk_logger->error("cur_img.inv_merge_mask is empty");
        }
        ReduceVector(prev_pts, status);
        ReduceVector(cur_pts, status);
        ReduceVector(ids, status);
        ReduceVector(track_cnt, status);
        Debugt("trackImageNaive | cur_pts.size:{}", cur_pts.size());
    }
    Infot("trackImageNaive | flowTrack left:{}", tt.TocThenTic());

    for (auto &n : track_cnt) n++;

    //RejectWithF();
    TicToc t_m;
    TicToc t_t;

    SortPoints(cur_pts, track_cnt, ids);
    if(!cur_img.inv_merge_mask.empty())
        mask = cur_img.inv_merge_mask.clone();
    else
        mask = cv::Mat(cur_img.color0.rows,cur_img.color0.cols,CV_8UC1,cv::Scalar(255));

    for(const auto& pt : cur_pts) cv::circle(mask, pt, cfg::kMinDist, 0, -1);
    mask_gpu.upload(mask);

    if (int n_max_cnt = cfg::kMaxCnt - (int)cur_pts.size(); n_max_cnt > 10)
    {
        Warnt("trackImageNaive | n_max_cnt:{}", n_max_cnt);
        //if(cur_img.gray0.empty()) cur_img.gray0_gpu.download(cur_img.gray0);
        Debugt("trackImageNaive | semantic_mask size:{}x{} type:{} ", semantic_mask.rows, semantic_mask.cols,
               semantic_mask.type());
        Debugt("trackImageNaive | mask size:{}x{} type:{} ", mask.rows, mask.cols, mask.type());
        Debugt("trackImageNaive | mask_gpu size:{}x{} type:{} ", mask_gpu.rows, mask_gpu.cols, mask_gpu.type());
        Debugt("trackImageNaive | img.gray0_gpu size:{}x{} type:{} ", cur_img.gray0_gpu.rows, cur_img.gray0_gpu.cols,
               cur_img.gray0_gpu.type());
        //n_pts = detectNewFeaturesGPU(n_max_cnt,cur_img.gray0_gpu,mask_gpu);
        n_pts = DetectShiTomasiCorners(n_max_cnt, cur_img.gray0, mask);

        /*cv::imshow("cur_img.gray0",cur_img.gray0);
        cv::waitKey(1);*/
        visual_new_pts = n_pts;
        for (auto &p : n_pts){
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
        Debugt("trackImageNaive | cur_pts.size:{}", cur_pts.size());
    }
    else
        n_pts.clear();

    Infot("trackImageNaive | detect feature:{}", tt.TocThenTic());

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    Infot("trackImageNaive | vel&&un:{}", tt.TocThenTic());

    if(cfg::is_stereo && (!cur_img.gray1.empty() || !cur_img.gray1_gpu.empty()) && !cur_pts.empty())
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        Debugt("trackImageNaive | flowTrack right start");

        std::vector<uchar> status= FeatureTrackByLK(cur_img.gray0, cur_img.gray1, cur_pts, cur_right_pts);
        //vector<uchar> status=flowTrackGpu(lk_optical_flow,lk_optical_flow_back,cur_img.gray0_gpu,cur_img.gray1_gpu,cur_pts,cur_right_pts);
        Debugt("trackImageNaive | flowTrack right finish");
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
        cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
        right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        prev_un_right_pts_map = cur_un_right_pts_map;
        Debugt("trackImageNaive | cur_right_pts.size:{}", cur_right_pts.size());
        Infot("trackImageNaive | flow_track right:{}", tt.TocThenTic());
    }
    if(cfg::is_show_track)
        drawTrack(cur_img, ids, cur_pts, cur_right_pts, prev_left_map);

    //t_inst_track.join();

    /*static int cnt=0;
    cnt++;
    if(cnt==5){
    *//*cv::Mat inst_mask = cv::Mat(img.color0.rows,img.color0.cols,CV_8UC3,cv::Scalar(0));
    for(auto &[key,inst] : insts_tracker->instances){
        if(inst.lost_num>0)continue;
        debug_t("inst.mask_tensor.sizes():{}", dims2str(inst.mask_tensor.sizes()));
        cv::Mat m;
        torch::Tensor t = inst.mask_tensor.unsqueeze(0);
        debug_t("t.sizes:{}", dims2str(t.sizes()));
        t = t.expand({3,inst.mask_tensor.sizes()[0],inst.mask_tensor.sizes()[1]});
        debug_t("t.sizes:{}", DimsToStr(t.sizes()));
        int color_array[3]={(int)inst.color[0],(int)inst.color[1],(int)inst.color[2]};
        torch::Tensor color = torch::from_blob(color_array,{3,1,1}).expand({3,inst.mask_tensor.sizes()[0],inst.mask_tensor.sizes()[1]});
        debug_t("1");
        torch::Tensor result = color * t.to(torch::kCPU) ;
        DebugT("2");

        m = cv::Mat(img.color0.rows,img.color0.cols,CV_8UC3,result.data_ptr()).clone();
        debug_t("3");

        //cv::cvtColor(inst.mask_img,m,CV_GRAY2BGR);
        inst_mask += m;
        DebugT("4");

    }*//*
        cv::imwrite("inst_mask.png",img.merge_mask);
        cv::imwrite("background_track.png",img_track);
        cv::Mat inst_show = img.color0.clone();
        insts_tracker->drawInsts(inst_show);
        cv::imwrite("inst_track.png",inst_show);

    }*/

    Infot("TrackSemanticImage 动态检测线程总时间:{} ms", t_i.TocThenTic());
    if(cfg::is_show_track)
        insts_tracker->DrawInsts(img_track_);
    Infot("TrackSemanticImage drawInsts:{} ms", t_i.TocThenTic());

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

    /*
    TicToc t_r,t_all;
    cur_time = img.time0;
    row = img.color0.rows;
    col = img.color0.cols;
    cur_pts.clear();

    DebugT("trackSemanticImage insts_info.size:{}",img.insts_info.size());

    cur_img = img;
    bool isInstsEmpty = cur_img.insts_info.empty();
    semantic_mask = img.inv_merge_mask;

    if constexpr(false){
        static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(cur_img.gray0, cur_img.gray0);
        if(!cur_img.gray1.empty())
            clahe->apply(cur_img.gray1, cur_img.gray1);
    }


    ///开启另一个线程检测动态特征点
    TicToc t_i;
    //std::thread t_inst_track(&InstsFeatManager::instsTrack, insts_tracker.get(), img);

    InfoT("trackSemanticImage 设置语义mask:{} ms",t_r.toc_then_tic());


    /// 跟踪特征点
    if (!prev_pts.empty()){
        vector<uchar> status=flowTrackGpu(lk_optical_flow,lk_optical_flow_back,prev_img.gray0_gpu,cur_img.gray0_gpu,prev_pts,cur_pts);
        //vector<uchar> status=flowTrack(prev_img.gray0,cur_img.gray0,prev_pts,cur_pts);
        if(!semantic_mask.empty()){
            for(int i=0;i<cur_pts.size();i++){
                if(status[i] && semantic_mask.at<uchar>(cur_pts[i])==0)
                    status[i]=0;
            }
        }
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        ReduceVector(track_cnt, status);
    }
    for (auto &cnt : track_cnt) cnt++; //将各个特征点的跟踪次数++
    InfoT("TrackSemanticImage 跟踪特征点:{} ms",t_r.toc_then_tic());

    /// 检测新的角点
    //RejectWithF();
    if (int n_max_cnt = cfg::kMaxCnt - (int)cur_pts.size(); n_max_cnt > 10){
        /*if(!semantic_mask.empty()){
            static cv::Mat mask_element=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(10,10),cv::Point(-1,-1));///对semantic_mask进行腐蚀，扩大黑色区域,防止检测的特征点落在动态物体的边缘
            cv::erode(semantic_mask,semantic_mask,mask_element,cv::Point(-1,-1));
        }*/

    /*
        sortPoints(cur_pts,track_cnt,ids);
        if(!semantic_mask.empty())
            mask = semantic_mask.clone();
        else
            mask = cv::Mat(row,col,CV_8UC1,cv::Scalar(255));
        for(const auto& pt : cur_pts){
            cv::circle(mask, pt, cfg::kMinDist, 0, -1);
        }
        mask_gpu.upload(mask);

        debug_t("trackSemanticImage start detect n_max_cnt,size:{}",n_max_cnt);
        detector = cv::cuda::createGoodFeaturesToTrackDetector(CV_8UC1, n_max_cnt, 0.01, cfg::kMinDist);
        cv::cuda::GpuMat d_new_pts;
        detector->detect(cur_img.gray0_gpu,d_new_pts,mask_gpu);
        gpuMat2Points(d_new_pts,n_pts);

        DebugT("trackSemanticImage gpuMat2Points n_pts:{}",n_pts.size());
        //n_pts = detectNewFeaturesGPU(n_max_cnt,cur_img.gray0_gpu,mask_gpu);

        visual_new_pts = n_pts;
        for (auto &p : n_pts){
            cur_pts.push_back(p);
            ids.push_back(n_id++); //给每个角点分配一个ID
            track_cnt.push_back(1);
        }
    }
    else{
        n_pts.clear();
    }
    info_t("trackSemanticImage 检测新的角点:{} ms",t_r.toc_then_tic());

    ///对特征点进行去畸变,返回归一化特征点
    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    /// 速度跟踪
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);
    info_t("trackSemanticImage 去畸变&&速度跟踪:{} ms",t_r.toc_then_tic());

    /// 右边相机图像的跟踪
    if(!(cur_img.gray1_gpu.empty() || cur_img.gray1.empty() ) && stereo_cam){
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();

        //vector<uchar> status=flowTrack(cur_img.gray0,cur_img.gray1,cur_pts,cur_right_pts);
        vector<uchar> status = flowTrackGpu(lk_optical_flow,lk_optical_flow_back,cur_img.gray0_gpu,cur_img.gray1_gpu,cur_pts,cur_right_pts);

        if(cfg::dataset == DatasetType::kViode){
            for(int i=0;i<cur_pts.size();i++){
                if(status[i] && belongViodeDynamic(cur_right_pts[i], cur_img.seg1))
                    status[i]=0;
            }
        }
        ids_right = ids;
        reduceVector(cur_right_pts, status);
        ReduceVector(ids_right, status);
        cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
        right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        prev_un_right_pts_map = cur_un_right_pts_map;
        DebugT("trackSemanticImage | cur_right_pts.size:{}",cur_right_pts.size());
        info_t("trackSemanticImage | flow_track right:{}",t_i.toc_then_tic());
    }

    ///光流跟踪的可视化
    if(cfg::kShowTrack) drawTrack(cur_img, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
    info_t("TrackSemanticImage 光流跟踪的可视化");//0.9ms

    //t_inst_track.join();

    InfoT("trackSemanticImage 动态检测线程总时间:{} ms",t_i.toc_then_tic());

    if(cfg::kShowTrack)insts_tracker->drawInsts(img_track);
    InfoT("trackSemanticImage drawInsts:{} ms",t_i.toc_then_tic());


    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    //insts_tracker->VisualizeInst(img.color0);
    //cv::imshow("merge_mask",insts_tracker->mask_background);
    //cv::imshow("inst_mask",img_track);
    //cv::imshow("inst0",insts_tracker->instances.begin()->second.mask_img);
    //cv::waitKey(1);

    /// 将归一化坐标、像素坐标、速度组合在一起
    return setOutputFeats();
    */
}


}
