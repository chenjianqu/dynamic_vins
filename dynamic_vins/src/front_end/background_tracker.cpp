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

    Debugv("init FeatureTracker");

    lk_optical_flow = cv::cuda::SparsePyrLKOpticalFlow::create(cv::Size(21, 21), 3, 30);
    lk_optical_flow_back = cv::cuda::SparsePyrLKOpticalFlow::create(
            cv::Size(21, 21), 3, 30, true);

    bg.id = 0;//表示相机
    bg.box2d = std::make_shared<Box2D>();
    line_detector = std::make_shared<LineDetector>(config_path);
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

    bg.roi->mask_cv = cv::Mat(cur_img.gray0.rows,cur_img.gray0.cols,CV_8UC1,cv::Scalar(255));

    bg.curr_points.clear();
    if (!bg.last_points.empty()){
        vector<uchar> status = FeatureTrackByLK(prev_img.gray0, img.gray0, bg.last_points, bg.curr_points,fe_para::is_flow_back);
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
    //根据跟踪次数进行排序
    SortPoints(bg.curr_points, bg.track_cnt, bg.ids);
    //设置mask，防止点聚集
    for(const auto& pt : bg.curr_points)
        cv::circle(bg.roi->mask_cv, pt, fe_para::kMinDist, 0, -1);

    int n_max_cnt = fe_para::kMaxCnt - static_cast<int>(bg.curr_points.size());
    vector<cv::Point2f> n_pts;
    if (n_max_cnt > 0){
        cv::goodFeaturesToTrack(cur_img.gray0, n_pts, fe_para::kMaxCnt - bg.curr_points.size(),
                                0.01, fe_para::kMinDist, bg.roi->mask_cv);
    }
    else{
        n_pts.clear();
    }

    for (auto &p : n_pts){
        bg.curr_points.push_back(p);
        bg.ids.push_back(InstFeat::global_id_count++);
        bg.track_cnt.push_back(1);
    }

    Infot("TrackImage | goodFeaturesToTrack:{} ms", tt.TocThenTic());

    bg.UndistortedPts(cam_t.cam0);
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

        if(!bg.curr_points.empty()){
            vector<uchar> status= FeatureTrackByLK(cur_img.gray0, cur_img.gray1, bg.curr_points, bg.right_points,fe_para::is_flow_back);

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
            bg.RightUndistortedPts(cam_t.cam1);
            bg.RightPtsVelocity(cur_time-prev_time);
        }
        bg.right_prev_id_pts = bg.right_curr_id_pts;

        Infot("TrackImage | flowTrack right:{} ms", tt.TocThenTic());
    }

    if(fe_para::is_show_track){
        //DrawTrack(cur_img, bg.ids, bg.curr_points, bg.right_points, prev_left_map);
        DrawTrack(cur_img.gray0, cur_img.gray1, bg.ids, bg.curr_points, bg.right_points, last_id_pts_map);
    }

    Infot("TrackImage | DrawTrack right:{} ms", tt.TocThenTic());

    prev_img = cur_img;
    prev_time = cur_time;

    bg.PostProcess();

    last_id_pts_map.clear();
    for(size_t i = 0; i < bg.curr_points.size(); i++)
        last_id_pts_map[bg.ids[i]] = bg.curr_points[i];

    return SetOutputFeats();
}



/**
 * 直线检测和跟踪
 * @param gray0
 * @param gray1
 */
void FeatureTracker::TrackLine(cv::Mat gray0, cv::Mat gray1,cv::Mat mask){
    ///线特征的提取和跟踪
    Debugt("TrackLine | start");
    bg.curr_lines = line_detector->Detect(gray0,mask);
    Debugt("TrackLine | detect new lines size:{}",bg.curr_lines->keylsd.size());

    line_detector->TrackLeftLine(bg.prev_lines, bg.curr_lines);
    Debugt("TrackLine | track lines size:{}",bg.curr_lines->keylsd.size());

    bg.curr_lines->SetLines();
    bg.curr_lines->UndistortedLineEndPoints(cam_t.cam0);

    if(cfg::is_stereo){
        Debugt("TrackLine | start track right image");

        bg.curr_lines_right = line_detector->Detect(gray1);
        line_detector->TrackRightLine(bg.curr_lines,bg.curr_lines_right);

        bg.curr_lines_right->SetLines();
        bg.curr_lines_right->UndistortedLineEndPoints(cam_t.cam1);
    }
    Debugt("TrackLine | finished");
}



/**
 * 不对动态物体进行任何处理,而直接将所有的特征点当作静态区域,跟踪点线特征
 * @param img
 * @return
 */
FeatureBackground FeatureTracker::TrackImageLine(SemanticImage &img)
{
    TicToc t_r,tt;
    cur_time = img.time0;
    cur_img = img;

    bg.roi->mask_cv = cv::Mat(cur_img.gray0.rows,cur_img.gray0.cols,CV_8UC1,cv::Scalar(255));

    std::thread line_thread;
    if(cfg::use_line){
        cv::Mat line_gray0 = img.gray0.clone();
        cv::Mat line_gray1 = img.gray1.clone();
        line_thread = std::thread(&FeatureTracker::TrackLine, this, line_gray0, line_gray1,cv::Mat());
    }

    ///跟踪左图像的点特征
    bg.curr_points.clear();
    if (!bg.last_points.empty()){
        vector<uchar> status = FeatureTrackByLK(prev_img.gray0, img.gray0, bg.last_points, bg.curr_points,fe_para::is_flow_back);
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
    for(const auto& pt : bg.curr_points)
        cv::circle(bg.roi->mask_cv, pt, fe_para::kMinDist, 0, -1);
    TicToc t_t;
    int n_max_cnt = fe_para::kMaxCnt - static_cast<int>(bg.curr_points.size());
    vector<cv::Point2f> n_pts;
    if (n_max_cnt > 0)
        cv::goodFeaturesToTrack(cur_img.gray0, n_pts, fe_para::kMaxCnt - bg.curr_points.size(),
                                0.01, fe_para::kMinDist, bg.roi->mask_cv);
    else
        n_pts.clear();

    for (auto &p : n_pts){
        bg.curr_points.push_back(p);
        bg.ids.push_back(InstFeat::global_id_count++);
        bg.track_cnt.push_back(1);
    }

    Infot("TrackImage | goodFeaturesToTrack:{} ms", tt.TocThenTic());

    bg.UndistortedPts(cam_t.cam0);
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
            vector<uchar> status= FeatureTrackByLK(cur_img.gray0, cur_img.gray1, bg.curr_points, bg.right_points,fe_para::is_flow_back);

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
            bg.RightUndistortedPts(cam_t.cam1);
            bg.RightPtsVelocity(cur_time-prev_time);
        }
        bg.right_prev_id_pts = bg.right_curr_id_pts;

        Infot("TrackImage | flowTrack right:{} ms", tt.TocThenTic());
    }

       if(fe_para::is_show_track){
            ///可视化点
            DrawTrack(cur_img, bg.ids, bg.curr_points, bg.right_points, last_id_pts_map);
       }


       if(cfg::use_line){
           line_thread.join();

           if(fe_para::is_show_track){

               //if(bg.prev_lines)
               //    line_detector->VisualizeLineMonoMatch(img_vis, bg.prev_lines, bg.curr_lines); //前后帧直线可视化

               ///可视化线
               line_detector->VisualizeLine(img_vis, bg.curr_lines);
               if(cfg::is_stereo){
                   if(cfg::dataset==DatasetType::kKitti){
                       line_detector->VisualizeRightLine(img_vis, bg.curr_lines_right, true);
                   }
                   else{
                       line_detector->VisualizeRightLine(img_vis, bg.curr_lines_right, false);
                   }


                   if(cfg::dataset==DatasetType::kKitti){
                       line_detector->VisualizeLineStereoMatch(img_vis, bg.curr_lines, bg.curr_lines_right,true);
                   }
                   else{
                       line_detector->VisualizeLineStereoMatch(img_vis, bg.curr_lines, bg.curr_lines_right,false);
                   }
               }



           }
       }

    Infot("TrackImage | DrawTrack right:{} ms", tt.TocThenTic());

    prev_img = cur_img;
    prev_time = cur_time;

    bg.PostProcess();

    last_id_pts_map.clear();
    for(size_t i = 0; i < bg.curr_points.size(); i++)
        last_id_pts_map[bg.ids[i]] = bg.curr_points[i];

    return SetOutputFeats();
}



FeatureBackground FeatureTracker::SetOutputFeats()
{
    FeatureBackground fm;

    std::map<unsigned int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> points;

    //left cam
    for (size_t i = 0; i < bg.ids.size(); i++){
        constexpr int camera_id = 0;
        Vec7d xyz_uv_velocity;
        xyz_uv_velocity << bg.curr_un_points[i].x, bg.curr_un_points[i].y, 1,
        bg.curr_points[i].x, bg.curr_points[i].y,
        bg.pts_velocity[i].x, bg.pts_velocity[i].y;

        points[bg.ids[i]].emplace_back(camera_id,  xyz_uv_velocity);
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

            points[bg.right_ids[i]].emplace_back(camera_id,  xyz_uv_velocity);
            //Debugt("id:{} cam:{}",bg.right_ids[i],camera_id);
        }
    }

    fm.points = points;

    if(cfg::use_line && bg.curr_lines){
        std::map<unsigned int, std::vector<std::pair<int,Line>>> lines;
        ///左图像的先特征
        for(Line& l:bg.curr_lines->un_lines){
            std::vector<std::pair<int,Line>> lv;
            lv.emplace_back(0,l);
            lines.insert({l.id,lv});
        }
        if(cfg::is_stereo && bg.curr_lines_right){
            ///右图像的线特征
            for(Line& l:bg.curr_lines_right->un_lines){
                lines[l.id].push_back({1,l});
            }
        }

        fm.lines = lines;
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
    if(cur_img.exist_inst && fe_para::use_mask_morphology){
        static auto erode_kernel = cv::getStructuringElement(
                cv::MORPH_RECT, cv::Size(fe_para::kMaskMorphologySize, fe_para::kMaskMorphologySize), cv::Point(-1, -1));
        static auto erode_filter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE,CV_8UC1,erode_kernel);
        erode_filter->apply(img.inv_merge_mask_gpu,img.inv_merge_mask_gpu);
        img.inv_merge_mask_gpu.download(img.inv_merge_mask);
    }

    if(cur_img.exist_inst)
        bg.roi->mask_cv = cur_img.inv_merge_mask.clone();
    else
        bg.roi->mask_cv = cv::Mat(cur_img.color0.rows,cur_img.color0.cols,CV_8UC1,cv::Scalar(255));

    std::thread line_thread;
    if(cfg::use_line){
        cv::Mat line_mask = bg.roi->mask_cv.clone();
        cv::Mat line_gray0 = img.gray0.clone();
        cv::Mat line_gray1 = img.gray1.clone();
        line_thread = std::thread(&FeatureTracker::TrackLine, this, line_gray0, line_gray1,line_mask);
        //line_thread.join();
        //TrackLine(line_gray0,line_gray1,cv::Mat());
    }

    ///特征点跟踪
    bg.TrackLeftGPU(img,prev_img,lk_optical_flow,lk_optical_flow_back);
    Infot("trackImageNaive | flowTrack left:{} ms", tt.TocThenTic());

    //RejectWithF();

    ///特征点检测
    //bg.SortPoints();

    bg.DetectNewFeature(img,true,fe_para::kMinDist,bg.roi->mask_cv);
    Infot("trackImageNaive | detect feature:{} ms", tt.TocThenTic());

    ///特征点矫正和计算速度
    bg.UndistortedPts(cam_t.cam0);
    bg.PtsVelocity(cur_time-prev_time);

    Infot("trackImageNaive | vel&&un:{} ms", tt.TocThenTic());

    ///右图像跟踪
    if(cfg::is_stereo && (!cur_img.gray1.empty() || !cur_img.gray1_gpu.empty()) && !bg.curr_points.empty()){
        bg.TrackRightGPU(img,lk_optical_flow,lk_optical_flow_back);
        bg.RightUndistortedPts(cam_t.cam1);
        bg.RightPtsVelocity(cur_time-prev_time);

        Debugt("trackImageNaive | bg.right_points.size:{}", bg.right_points.size());
        Infot("trackImageNaive | flow_track right:{} ms", tt.TocThenTic());
    }

    if(fe_para::is_show_track)
        DrawTrack(cur_img, bg.ids, bg.curr_points, bg.right_points, last_id_pts_map);

    if(cfg::use_line){
        line_thread.join();

        /*
         //可视化线匹配
         if(bg.prev_lines)
            line_detector->VisualizeLineMonoMatch(img_vis, bg.prev_lines, bg.curr_lines);*/

        if(fe_para::is_show_track){
            ///可视化线
            line_detector->VisualizeLine(img_vis, bg.curr_lines);
            if(cfg::is_stereo){
                if(cfg::dataset==DatasetType::kKitti){
                    line_detector->VisualizeRightLine(img_vis, bg.curr_lines_right, true);
                }
                else{
                    line_detector->VisualizeRightLine(img_vis, bg.curr_lines_right, false);
                }
            }

            /*
            //可视化双目线匹配
            if(cfg::dataset==DatasetType::kKitti){
                line_detector->VisualizeLineStereoMatch(img_vis, bg.curr_lines, bg.curr_lines_right,true);
            }
            else{
                line_detector->VisualizeLineStereoMatch(img_vis, bg.curr_lines, bg.curr_lines_right,false);
            }*/

        }
    }

    prev_img = cur_img;
    prev_time = cur_time;

    bg.PostProcess();

    last_id_pts_map.clear();
    for(size_t i = 0; i < bg.curr_points.size(); i++)
        last_id_pts_map[bg.ids[i]] = bg.curr_points[i];

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
                               std::map<unsigned int, cv::Point2f> &prevLeftPts){

    if(cfg::slam==SLAM::kNaive && !img.inv_merge_mask_gpu.empty()){
        cv::cuda::GpuMat img_show_gpu;
        cv::cuda::cvtColor(img.inv_merge_mask_gpu,img_show_gpu,CV_GRAY2BGR);
        //cv::cuda::cvtColor(mask_gpu,img_show_gpu,CV_GRAY2BGR);
        cv::cuda::scaleAdd(img_show_gpu,0.5,img.color0_gpu,img_show_gpu);
        img_show_gpu.download(img_vis);
    }
    else if(cfg::slam==SLAM::kDynamic && !img.inv_merge_mask_gpu.empty()){
        cv::cuda::GpuMat img_show_gpu;
        cv::cuda::bitwise_not(img.inv_merge_mask_gpu,img_show_gpu);//二值化反转
        cv::cuda::cvtColor(img_show_gpu,img_show_gpu,CV_GRAY2BGR);
        cv::cuda::scaleAdd(img_show_gpu,0.5,img.color0_gpu,img_show_gpu);
        img_show_gpu.download(img_vis);
    }
    else{
        img_vis = img.color0;
    }

    if (cfg::is_stereo && !img.color1.empty()){
        if(cfg::dataset == DatasetType::kKitti){
            cv::vconcat(img_vis, img.color1, img_vis);
        }else{
            cv::hconcat(img_vis, img.color1, img_vis);
        }
    }

    //DEBUG
    //return;

    //cv::cvtColor(img_track, img_track, CV_GRAY2RGB);
    for (size_t j = 0; j < curLeftPts.size(); j++){
        double len = std::min(1.0, 1.0 * bg.track_cnt[j] / 20);
        cv::circle(img_vis, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }

    for (auto & pt : bg.visual_new_points)
        cv::circle(img_vis, pt, 2, cv::Scalar(255, 255, 255), 2);

    if (cfg::is_stereo && !img.color1.empty() ){
        if(cfg::dataset == DatasetType::kKitti){
            for (auto &rightPt : curRightPts){
                rightPt.y += (float)img.color0.rows;
                cv::circle(img_vis, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            }
        }
        else{
            for (auto &rightPt : curRightPts){
                rightPt.x += (float)img.color0.cols;
                cv::circle(img_vis, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    /*for (size_t i = 0; i < curLeftIds.size(); i++){
        if(auto it = prevLeftPts.find(curLeftIds[i]); it != prevLeftPts.end()){
            cv::arrowedLine(img_track_, curLeftPts[i], it->second, cv::Scalar(0, 255, 0),
                            1, 8, 0, 0.2);
        }
    }  */

}


void FeatureTracker::DrawTrack(const cv::Mat &imLeft, const cv::Mat &imRight,
                               vector<unsigned int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts,
                               vector<cv::Point2f> &curRightPts,
                               map<unsigned int, cv::Point2f> &prevLeftPtsMap){
    //int rows = imLeft.rows;
    int cols = imLeft.cols;
    if (!imRight.empty() && cfg::is_stereo){
        if(cfg::dataset == DatasetType::kKitti){
            cv::vconcat(imLeft, imRight, img_vis);
        }else{
            cv::hconcat(imLeft, imRight, img_vis);
        }
    }
    else{
        img_vis = imLeft.clone();
    }


    cv::cvtColor(img_vis, img_vis, CV_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++){
        double len = std::min(1.0, 1.0 * bg.track_cnt[j] / 20);
        cv::circle(img_vis, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }

    if (cfg::is_stereo && !imRight.empty() ){
        if(cfg::dataset == DatasetType::kKitti){
            for (auto &rightPt : curRightPts){
                rightPt.y += (float)imLeft.rows;
                cv::circle(img_vis, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            }
        }
        else{
            for (auto &rightPt : curRightPts){
                rightPt.x += (float)imLeft.cols;
                cv::circle(img_vis, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    map<unsigned int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++){
        unsigned int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end()){
            cv::arrowedLine(img_vis, curLeftPts[i], mapIt->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }

    //draw prediction
    /*
    for(size_t i = 0; i < predict_pts_debug.size(); i++)
    {
        cv::circle(imTrack, predict_pts_debug[i], 2, cv::Scalar(0, 170, 255), 2);
    }
    */
    //printf("predict pts size %d \n", (int)predict_pts_debug.size());

    //cv::Mat imCur2Compress;
    //cv::resize(imCur2, imCur2Compress, cv::Size(cols, rows / 2));
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
    if(img.exist_inst && fe_para::use_mask_morphology){
        ErodeMaskGpu(cur_img.inv_merge_mask_gpu, cur_img.inv_merge_mask_gpu,fe_para::kMaskMorphologySize);
        cur_img.inv_merge_mask_gpu.download(cur_img.inv_merge_mask);
    }

    if(cur_img.exist_inst)
        bg.roi->mask_cv = cur_img.inv_merge_mask.clone();
    else
        bg.roi->mask_cv = cv::Mat(cur_img.color0.rows,cur_img.color0.cols,CV_8UC1,cv::Scalar(255));

    ///特征点跟踪
    bg.TrackLeft(img.gray0,prev_img.gray0,bg.roi->mask_cv);

    //RejectWithF();

    ///检测新的特征点
    //bg.SortPoints();
    bg.DetectNewFeature(img,false,fe_para::kMinDist,bg.roi->mask_cv);
    Infot("TrackSemanticImage | detect feature:{} ms", tt.TocThenTic());

    ///矫正特征点,并计算特征点的速度
    bg.UndistortedPts(cam_t.cam0);
    bg.PtsVelocity(cur_time-prev_time);

    ///跟踪右图像
    if(cfg::is_stereo && (!cur_img.gray1.empty() || !cur_img.gray1_gpu.empty()) && !bg.curr_points.empty()){
        bg.TrackRightGPU(img,lk_optical_flow,lk_optical_flow_back);

        bg.RightUndistortedPts(cam_t.cam1);
        bg.RightPtsVelocity(cur_time-prev_time);
        Debugt("TrackSemanticImage | track right finished");
    }

    if(fe_para::is_show_track){
        DrawTrack(cur_img, bg.ids, bg.curr_points, bg.right_points, last_id_pts_map);
    }

    /*if(img.seq%10==0){
        string save_name = cfg::kDatasetSequence+"_"+std::to_string(img.seq)+"_bg.png";
        cv::imwrite(save_name,img_track_);
    }*/

    prev_img = cur_img;
    prev_time = cur_time;

    bg.PostProcess();

    last_id_pts_map.clear();
    for(size_t i = 0; i < bg.curr_points.size(); i++)
        last_id_pts_map[bg.ids[i]] = bg.curr_points[i];

    return SetOutputFeats();
}


}
