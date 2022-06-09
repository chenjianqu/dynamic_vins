/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/
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

#include "background_tracker.h"

#include <opencv2/cudaimgproc.hpp>

#include "front_end_parameters.h"
#include "utils/dataset/viode_utils.h"

namespace dynamic_vins{\


FeatureTracker::FeatureTracker(const string& config_path)
{
    fe_para::SetParameters(config_path);

    Debugt("init FeatureTracker");

    lk_optical_flow = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30);
    lk_optical_flow_back = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(21, 21), 3, 30, true);

    bg.id = -1;//表示相机
    bg.box2d = std::make_shared<Box2D>();
}



/**
 * 不对动态物体进行任何处理,而直接将所有的特征点当作静态区域
 * @param img
 * @return
 */
FeatureBackground FeatureTracker::TrackImage(SemanticImage &img)
{
    TicToc t_r,tt;
    cur_time = img.time0;
    cur_img = img;

    bg.box2d->mask_cv = cv::Mat(cur_img.gray0.rows,cur_img.gray0.cols,CV_8UC1,cv::Scalar(255));

    bg.curr_points.clear();
    if (!bg.last_points.empty()){
        vector<uchar> status = FeatureTrackByLK(prev_img.gray0, img.gray0, bg.last_points, bg.curr_points);
        ReduceVector(bg.last_points, status);
        ReduceVector(bg.curr_points, status);
        ReduceVector(bg.ids, status);
        ReduceVector(bg.track_cnt, status);
    }
    for (auto &n : bg.track_cnt)
        n++;

    Infot("TrackImage | flowTrack:{} ms", tt.TocThenTic());

    //RejectWithF();
    TicToc t_m;
    SortPoints(bg.curr_points, bg.track_cnt, bg.ids);
    for(const auto& pt : bg.curr_points) cv::circle(bg.box2d->mask_cv, pt, fe_para::kMinDist, 0, -1);
    TicToc t_t;
    int n_max_cnt = fe_para::kMaxCnt - static_cast<int>(bg.curr_points.size());
    vector<cv::Point2f> n_pts;
    if (n_max_cnt > 0)
        cv::goodFeaturesToTrack(cur_img.gray0, n_pts, fe_para::kMaxCnt - bg.curr_points.size(),
                                0.01, fe_para::kMinDist, bg.box2d->mask_cv);
    else
        n_pts.clear();

    for (auto &p : n_pts){
        bg.curr_points.push_back(p);
        bg.ids.push_back(InstFeat::global_id_count++);
        bg.track_cnt.push_back(1);
    }

    Infot("TrackImage | goodFeaturesToTrack:{} ms", tt.TocThenTic());

    bg.UndistortedPts(cam0);
    bg.PtsVelocity(cur_time-prev_time);
    //bg.curr_un_points = UndistortedPts(bg.curr_points, cam0);
    //bg.pts_velocity = PtsVelocity(bg.ids, bg.curr_un_points, bg.curr_id_pts,bg.prev_id_pts);

    Infot("TrackImage | un&&vel:{} ms", tt.TocThenTic());

    if(cfg::is_stereo && !cur_img.gray1.empty())
    {
        bg.right_ids.clear();
        bg.right_points.clear();
        bg.right_un_points.clear();
        bg.right_pts_velocity.clear();
        bg.right_curr_id_pts.clear();
        if(!bg.curr_points.empty())
        {
            vector<uchar> status= FeatureTrackByLK(cur_img.gray0, cur_img.gray1, bg.curr_points, bg.right_points);

            bg.right_ids = bg.ids;
            ReduceVector(bg.right_points, status);
            ReduceVector(bg.right_ids, status);
            // only keep left-right pts
            /*
            reduceVector(bg.curr_points, status);
            reduceVector(ids, status);
            reduceVector(bg.track_cnt, status);
            reduceVector(cur_un_pts, status);
            ReduceVector(pts_velocity, status);
            */
            //bg.right_un_points = UndistortedPts(bg.right_points, cam1);
            //bg.right_pts_velocity = PtsVelocity(bg.right_ids, bg.right_un_points, bg.right_curr_id_pts, bg.right_prev_id_pts);
            bg.RightUndistortedPts(cam1);
            bg.RightPtsVelocity(cur_time-prev_time);
        }
        bg.right_prev_id_pts = bg.right_curr_id_pts;

        Infot("TrackImage | flowTrack right:{} ms", tt.TocThenTic());
    }

    if(fe_para::is_show_track)
        DrawTrack(cur_img, bg.ids, bg.curr_points, bg.right_points, prev_left_map);

    Infot("TrackImage | DrawTrack right:{} ms", tt.TocThenTic());

    prev_img = cur_img;
    prev_time = cur_time;

    bg.PostProcess();

    prev_left_map.clear();
    for(size_t i = 0; i < bg.curr_points.size(); i++)
        prev_left_map[bg.ids[i]] = bg.curr_points[i];

    return SetOutputFeats();
}


FeatureBackground FeatureTracker::SetOutputFeats()
{
    FeatureBackground fm;
    //left cam
    for (size_t i = 0; i < bg.ids.size(); i++){
        constexpr int camera_id = 0;
        Vec7d xyz_uv_velocity;
        xyz_uv_velocity << bg.curr_un_points[i].x, bg.curr_un_points[i].y, 1,
        bg.curr_points[i].x, bg.curr_points[i].y,
        bg.pts_velocity[i].x, bg.pts_velocity[i].y;
        fm[bg.ids[i]].emplace_back(camera_id,  xyz_uv_velocity);
        //Debugt("id:{} cam:{}",ids[i],camera_id);
    }
    //stereo
    if (cfg::is_stereo && !cur_img.gray1.empty()){
        for (size_t i = 0; i < bg.right_ids.size(); i++){
            constexpr int camera_id = 1;
            Vec7d xyz_uv_velocity;
            xyz_uv_velocity << bg.right_un_points[i].x, bg.right_un_points[i].y, 1,
            bg.right_points[i].x, bg.right_points[i].y,
            bg.right_pts_velocity[i].x, bg.right_pts_velocity[i].y;
            fm[bg.right_ids[i]].emplace_back(camera_id,  xyz_uv_velocity);
            //Debugt("id:{} cam:{}",bg.right_ids[i],camera_id);
        }
    }
    return fm;
}


/**
 * 简单的去掉动态物体区域来提高物体估计的精度
 * @param img
 * @return
 */
FeatureBackground FeatureTracker::TrackImageNaive(SemanticImage &img)
{
    TicToc t_r,tt;
    cur_time = img.time0;
    cur_img = img;

    cur_img.gray0_gpu.download(cur_img.gray0);
    cur_img.gray1_gpu.download(cur_img.gray1);

    Debugt("TrackImageNaive | input mask:{}", DimsToStr(img.inv_merge_mask_gpu.size()));

    ///形态学运算
    if(cur_img.exist_inst){
        static auto erode_kernel = cv::getStructuringElement(
                cv::MORPH_RECT,cv::Size(15,15),cv::Point(-1,-1));
        static auto erode_filter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE,CV_8UC1,erode_kernel);
        erode_filter->apply(img.inv_merge_mask_gpu,img.inv_merge_mask_gpu);
        img.inv_merge_mask_gpu.download(img.inv_merge_mask);
    }

    if(cur_img.exist_inst)
        bg.box2d->mask_cv = cur_img.inv_merge_mask.clone();
    else
        bg.box2d->mask_cv = cv::Mat(cur_img.color0.rows,cur_img.color0.cols,CV_8UC1,cv::Scalar(255));

    ///特征点跟踪
    bg.TrackLeftGPU(img,prev_img,lk_optical_flow,lk_optical_flow_back);
    Infot("trackImageNaive | flowTrack left:{} ms", tt.TocThenTic());

    //RejectWithF();

    ///特征点检测
    //bg.SortPoints();

    bg.DetectNewFeature(img,true);
    Infot("trackImageNaive | detect feature:{} ms", tt.TocThenTic());

    ///特征点矫正和计算速度
    bg.UndistortedPts(cam0);
    bg.PtsVelocity(cur_time-prev_time);

    Infot("trackImageNaive | vel&&un:{} ms", tt.TocThenTic());

    ///右图像跟踪
    if(cfg::is_stereo && (!cur_img.gray1.empty() || !cur_img.gray1_gpu.empty()) && !bg.curr_points.empty()){
        bg.TrackRightGPU(img,lk_optical_flow,lk_optical_flow_back);
        bg.RightUndistortedPts(cam1);
        bg.RightPtsVelocity(cur_time-prev_time);

        Debugt("trackImageNaive | bg.right_points.size:{}", bg.right_points.size());
        Infot("trackImageNaive | flow_track right:{} ms", tt.TocThenTic());
    }

    if(fe_para::is_show_track)
        DrawTrack(cur_img, bg.ids, bg.curr_points, bg.right_points, prev_left_map);

    prev_img = cur_img;
    prev_time = cur_time;

    bg.PostProcess();

    prev_left_map.clear();
    for(size_t i = 0; i < bg.curr_points.size(); i++)
        prev_left_map[bg.ids[i]] = bg.curr_points[i];

    return SetOutputFeats();
}



void FeatureTracker::RejectWithF()
{
    /*    if (bg.curr_points.size() >= 8)
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
            ReduceVector(bg.track_cnt, status);
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




void FeatureTracker::DrawTrack(const SemanticImage &img,
                               vector<unsigned int> &curLeftIds,
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
        double len = std::min(1.0, 1.0 * bg.track_cnt[j] / 20);
        cv::circle(img_track_, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }
    for (auto & pt : bg.visual_new_points)
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




/**
 * 前端的主函数, 两个线程并行跟踪背景区域和动态区域
 * @param img
 * @return
 */
FeatureBackground FeatureTracker::TrackSemanticImage(SemanticImage &img)
{
    TicToc t_r,tt;
    cur_time = img.time0;
    cur_img = img;

    ///形态学运算
    if(img.exist_inst){
        ErodeMaskGpu(cur_img.inv_merge_mask_gpu, cur_img.inv_merge_mask_gpu);
        cur_img.inv_merge_mask_gpu.download(cur_img.inv_merge_mask);
    }

    if(cur_img.exist_inst)
        bg.box2d->mask_cv = cur_img.inv_merge_mask.clone();
    else
        bg.box2d->mask_cv = cv::Mat(cur_img.color0.rows,cur_img.color0.cols,CV_8UC1,cv::Scalar(255));

    ///特征点跟踪
    bg.TrackLeft(img,prev_img,cfg::use_background_flow);

    //RejectWithF();

    ///检测新的特征点
    //bg.SortPoints();
    bg.DetectNewFeature(img,false);
    Infot("TrackSemanticImage | detect feature:{} ms", tt.TocThenTic());

    ///矫正特征点,并计算特征点的速度
    bg.UndistortedPts(cam0);
    bg.PtsVelocity(cur_time-prev_time);

    ///跟踪右图像
    if(cfg::is_stereo && (!cur_img.gray1.empty() || !cur_img.gray1_gpu.empty()) && !bg.curr_points.empty()){
        bg.TrackRightGPU(img,lk_optical_flow,lk_optical_flow_back);

        bg.RightUndistortedPts(cam1);
        bg.RightPtsVelocity(cur_time-prev_time);
    }
    if(fe_para::is_show_track)
        DrawTrack(cur_img, bg.ids, bg.curr_points, bg.right_points, prev_left_map);

    prev_img = cur_img;
    prev_time = cur_time;

    bg.PostProcess();

    prev_left_map.clear();
    for(size_t i = 0; i < bg.curr_points.size(); i++)
        prev_left_map[bg.ids[i]] = bg.curr_points[i];

    return SetOutputFeats();
}


}
