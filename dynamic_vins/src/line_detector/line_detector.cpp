/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <vector>
#include <random>

#include <opencv2/core/types.hpp>
#include "line_detector.h"
#include "utils/log_utils.h"

using std::vector;

namespace dynamic_vins{

    LineDetector::LineDetector(const std::string &config_file) {
        int image_width,image_height;

        cv::FileStorage fs(config_file, cv::FileStorage::READ);
        if(!fs.isOpened()){
            throw std::runtime_error(std::string("ERROR: Wrong path to settings:"+config_file));
        }
        fs["image_width"] >> image_width;
        fs["image_height"] >> image_height;
        fs.release();

        lsd_detector = cv::line_descriptor::LSDDetectorC::createLSDDetectorC();
        // lsd parameters

        lsd_opts.refine       = 1;     //1     	The way found lines will be refined
        lsd_opts.scale        = 0.5;   //0.8   	The scale of the image that will be used to find the lines. Range (0..1].
        lsd_opts.sigma_scale  = 0.6;	//0.6  	Sigma for Gaussian filter. It is computed as sigma = _sigma_scale/_scale.
        lsd_opts.quant        = 2.0;	//2.0   Bound to the quantization error on the gradient norm
        lsd_opts.ang_th       = 22.5;	//22.5	Gradient angle tolerance in degrees
        lsd_opts.log_eps      = 1.0;	//0		Detection threshold: -log10(NFA) > log_eps. Used only when advance refinement is chosen
        lsd_opts.density_th   = 0.6;	//0.7	Minimal density of aligned region points in the enclosing rectangle.
        lsd_opts.n_bins       = 1024;	//1024 	Number of bins in pseudo-ordering of gradient modulus.
        double min_line_length = 0.125;  // Line segments shorter than that are rejected
        // lsd_opts.refine       = 1;
        // lsd_opts.scale        = 0.5;
        // lsd_opts.sigma_scale  = 0.6;
        // lsd_opts.quant        = 2.0;
        // lsd_opts.ang_th       = 22.5;
        // lsd_opts.log_eps      = 1.0;
        // lsd_opts.density_th   = 0.6;
        // lsd_opts.n_bins       = 1024;
        // double min_line_length = 0.125;
        lsd_opts.min_length   = min_line_length*(std::min(image_width,image_height));
        lbd_descriptor = cv::line_descriptor::BinaryDescriptor::createBinaryDescriptor();
        line_matcher = cv::line_descriptor::BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
    }



    FrameLines::Ptr LineDetector::Detect(cv::Mat &img,const cv::Mat &mask) {
        std::vector<cv::line_descriptor::KeyLine> lsd_temp;

        lsd_detector->detect(img,lsd_temp,2, 1, lsd_opts,mask);

        cv::Mat lsd_descr_temp;
        lbd_descriptor->compute( img, lsd_temp, lsd_descr_temp);

        FrameLines::Ptr lines = std::make_shared<FrameLines>();

        for ( int i = 0; i < (int) lsd_temp.size(); i++ ){
            if( lsd_temp[i].octave == 0 && lsd_temp[i].lineLength >= 60){
                lines->keylsd.push_back( lsd_temp[i]);
                lines->lbd_descr.push_back( lsd_descr_temp.row(i) );
            }
        }

        return lines;
    }


    void LineDetector::TrackLine(vector<unsigned int> &curr_line_ids,
                                 vector<cv::line_descriptor::KeyLine> &curr_keylsd,
                                 cv::Mat &curr_lbd,
                                 std::map<unsigned int,int> &curr_track_cnt,
                                 vector<unsigned int> &prev_line_ids,
                                 vector<cv::line_descriptor::KeyLine> &prev_keylsd,
                                 cv::Mat &prev_lbd,
                                 std::map<unsigned int,int> &prev_track_cnt){
        ///描述子匹配
        std::vector<cv::DMatch> lsd_matches;
        line_matcher->match(curr_lbd, prev_lbd, lsd_matches);

        /* select best matches */
        std::vector<cv::DMatch> good_matches;
        for ( int i = 0; i < (int) lsd_matches.size(); i++ ){
            if( lsd_matches[i].distance < 30 ){
                cv::DMatch mt = lsd_matches[i];
                cv::line_descriptor::KeyLine line1 =  curr_keylsd[mt.queryIdx] ;
                cv::line_descriptor::KeyLine line2 =  prev_keylsd[mt.trainIdx] ;
                cv::Point2f serr = line1.getStartPoint() - line2.getEndPoint();
                cv::Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
                if((serr.dot(serr) < 200 * 200) && (eerr.dot(eerr) < 200 * 200)&&abs(line1.angle-line2.angle)<0.1)   // 线段在图像里不会跑得特别远
                    good_matches.push_back( lsd_matches[i] );
            }
        }

        std::map<unsigned int,int> track_cnt_tmp;//每条线的跟踪次数

        std::vector<unsigned int > success_id;
        for (int k = 0; k < good_matches.size(); ++k) {
            cv::DMatch mt = good_matches[k];
            unsigned int line_id = prev_line_ids[mt.trainIdx];
            curr_line_ids[mt.queryIdx] = line_id;
            success_id.push_back(line_id);
            track_cnt_tmp.insert({line_id,prev_track_cnt[line_id]+1});//添加新的跟踪次数
        }

        curr_track_cnt = track_cnt_tmp;

        ///将所有的线段划分为成功跟踪和未成功跟踪的线
        vector<cv::line_descriptor::KeyLine> vecLine_tracked, vecLine_new;
        vector<unsigned int> lineID_tracked, lineID_new;
        cv::Mat descr_tracked, descr_new;
        // 将跟踪的线和没跟踪上的线进行区分
        for (size_t i = 0; i < curr_keylsd.size(); ++i){
            if(curr_line_ids[i] == -1){
                curr_line_ids[i] = global_line_id++;
                vecLine_new.push_back(curr_keylsd[i]);
                lineID_new.push_back(curr_line_ids[i]);
                descr_new.push_back(curr_lbd.row(i ) );
            }
            else{
                vecLine_tracked.push_back(curr_keylsd[i]);
                lineID_tracked.push_back(curr_line_ids[i]);
                descr_tracked.push_back(curr_lbd.row(i ) );
            }
        }

        ///将未跟踪的线划分为垂线或水平线
        vector<cv::line_descriptor::KeyLine> h_line_new, v_line_new;
        vector<unsigned int> h_lineID_new,v_lineID_new;
        cv::Mat h_descr_new,v_descr_new;
        for (size_t i = 0; i < vecLine_new.size(); ++i){
            if((((vecLine_new[i].angle >= 3.14/4 && vecLine_new[i].angle <= 3*3.14/4)) ||
            (vecLine_new[i].angle <= -3.14/4 && vecLine_new[i].angle >= -3*3.14/4))){
                h_line_new.push_back(vecLine_new[i]);
                h_lineID_new.push_back(lineID_new[i]);
                h_descr_new.push_back(descr_new.row(i ));
            }
            else{
                v_line_new.push_back(vecLine_new[i]);
                v_lineID_new.push_back(lineID_new[i]);
                v_descr_new.push_back(descr_new.row(i ));
            }
        }

        ///统计已跟踪直线的垂线或水平线的数量
        int h_line=0,v_line=0;
        for (auto & l : vecLine_tracked){
            if((((l.angle >= 3.14 / 4 && l.angle <= 3 * 3.14 / 4)) ||
            (l.angle <= -3.14 / 4 && l.angle >= -3 * 3.14 / 4))){
                h_line ++;
            }
            else{
                v_line ++;
            }
        }

        int diff_h = 35 - h_line;
        int diff_v = 35 - v_line;
        ///补充水平线的线条
        if( diff_h > 0) {   // 补充线条
            int kkk = 1;
            if(diff_h > h_line_new.size())
                diff_h = h_line_new.size();
            else
                kkk = int(h_line_new.size() / diff_h);

            for (int k = 0; k < diff_h; ++k){
                vecLine_tracked.push_back(h_line_new[k]);
                lineID_tracked.push_back(h_lineID_new[k]);
                descr_tracked.push_back(h_descr_new.row(k));
            }
        }

        ///补充垂线的线条
        if( diff_v > 0){    // 补充线条
            int kkk = 1;
            if(diff_v > v_line_new.size())
                diff_v = v_line_new.size();
            else
                kkk = int(v_line_new.size() / diff_v);

            for (int k = 0; k < diff_v; ++k){
                vecLine_tracked.push_back(v_line_new[k]);
                lineID_tracked.push_back(v_lineID_new[k]);
                descr_tracked.push_back(v_descr_new.row(k));
                curr_track_cnt.insert({v_lineID_new[k],1});//添加该直线的跟踪次数
            }
        }

        curr_keylsd = vecLine_tracked;
        curr_line_ids = lineID_tracked;
        curr_lbd = descr_tracked;
    }



    void LineDetector::TrackLeftLine(const FrameLines::Ptr& prevLines, const FrameLines::Ptr& currLines){
        ///第一帧，设置线ID
        if(!prevLines){
            for (size_t i = 0; i < currLines->keylsd.size(); ++i) {
                currLines->line_ids.push_back(global_line_id++);
            }
            return;
        }
        else{
            for (size_t i = 0; i < currLines->keylsd.size(); ++i) {
                currLines->line_ids.push_back(-1);   // give a negative id
            }
        }

        TrackLine(currLines->line_ids,currLines->keylsd,currLines->lbd_descr,currLines->track_cnt,
                  prevLines->line_ids,prevLines->keylsd,prevLines->lbd_descr,prevLines->track_cnt);

    }


    void LineDetector::TrackRightLine(const FrameLines::Ptr& left_lines, const FrameLines::Ptr& right_lines){
        for (size_t i = 0; i < right_lines->keylsd.size(); ++i) {
            right_lines->line_ids.push_back(-1);   // give a negative id
        }

        ///描述子匹配
        std::vector<cv::DMatch> lsd_matches;
        line_matcher->match(right_lines->lbd_descr, left_lines->lbd_descr, lsd_matches);

        /* select best matches */
        std::vector<cv::DMatch> good_matches;
        std::vector<cv::line_descriptor::KeyLine> good_Keylines;
        for ( int i = 0; i < (int) lsd_matches.size(); i++ ){
            if( lsd_matches[i].distance < 30 ){
                cv::DMatch mt = lsd_matches[i];
                cv::line_descriptor::KeyLine line1 =  right_lines->keylsd[mt.queryIdx] ;
                cv::line_descriptor::KeyLine line2 =  left_lines->keylsd[mt.trainIdx] ;
                cv::Point2f serr = line1.getStartPoint() - line2.getEndPoint();
                cv::Point2f eerr = line1.getEndPoint() - line2.getEndPoint();
                if((serr.dot(serr) < 200 * 200) && (eerr.dot(eerr) < 200 * 200)&&abs(line1.angle-line2.angle)<0.1)   // 线段在图像里不会跑得特别远
                    good_matches.push_back( lsd_matches[i] );
            }
        }

        std::map<unsigned int,int> track_cnt_tmp;//每条线的跟踪次数

        std::vector<unsigned int > success_id;
        for (int k = 0; k < good_matches.size(); ++k) {
            cv::DMatch mt = good_matches[k];
            unsigned int line_id = left_lines->line_ids[mt.trainIdx];
            right_lines->line_ids[mt.queryIdx] = line_id;
            success_id.push_back(line_id);
        }

        //visualize_line_match(forwframe_->img.clone(), curframe_->img.clone(), forwframe_->keylsd, curframe_->keylsd, good_matches);

        ///将所有的线段划分为成功跟踪和未成功跟踪的线
        vector<cv::line_descriptor::KeyLine> vecLine_tracked, vecLine_new;
        vector<unsigned int> lineID_tracked, lineID_new;
        cv::Mat DEscr_tracked, Descr_new;
        // 将跟踪的线和没跟踪上的线进行区分
        for (size_t i = 0; i < right_lines->keylsd.size(); ++i){
            if(right_lines->line_ids[i] != -1){
                vecLine_tracked.push_back(right_lines->keylsd[i]);
                lineID_tracked.push_back(right_lines->line_ids[i]);
                DEscr_tracked.push_back(right_lines->lbd_descr.row( i ) );
            }
        }

        right_lines->keylsd = vecLine_tracked;
        right_lines->line_ids = lineID_tracked;
        right_lines->lbd_descr = DEscr_tracked;
    }


    void LineDetector::NearbyLineTracking(const std::vector<Line> &forw_lines, const std::vector<Line> &cur_lines,
                                          std::vector<std::pair<int, int> > &lineMatches) {

        float th = 3.1415926/9;
        float dth = 30 * 30;
        for (size_t i = 0; i < forw_lines.size(); ++i) {
            Line lf = forw_lines.at(i);
            Line best_match;
            size_t best_j = 100000;
            size_t best_i = 100000;
            float grad_err_min_j = 100000;
            float grad_err_min_i = 100000;
            vector<Line> candidate;

            // 从 forw --> cur 查找
            for(size_t j = 0; j < cur_lines.size(); ++j) {
                Line lc = cur_lines.at(j);
                // condition 1
                cv::Point2f d = lf.Center - lc.Center;
                float dist = d.dot(d);
                if( dist > dth) continue;  //
                // condition 2
                float delta_theta1 = fabs(lf.theta - lc.theta);
                float delta_theta2 = 3.1415926 - delta_theta1;
                if( delta_theta1 < th || delta_theta2 < th)
                {
                    //std::cout << "theta: "<< lf.theta * 180 / 3.14259 <<" "<< lc.theta * 180 / 3.14259<<" "<<delta_theta1<<" "<<delta_theta2<<std::endl;
                    candidate.push_back(lc);
                    //float cost = fabs(lf.image_dx - lc.image_dx) + fabs( lf.image_dy - lc.image_dy) + 0.1 * dist;
                    float cost = fabs(lf.line_grad_avg - lc.line_grad_avg) + dist/10.0;

                    //std::cout<< "line match cost: "<< cost <<" "<< cost - sqrt( dist )<<" "<< sqrt( dist ) <<"\n\n";
                    if(cost < grad_err_min_j)
                    {
                        best_match = lc;
                        grad_err_min_j = cost;
                        best_j = j;
                    }
                }

            }
            if(grad_err_min_j > 50) continue;  // 没找到

            // 如果 forw --> cur 找到了 best, 那我们反过来再验证下
            if(best_j < cur_lines.size())
            {
                // 反过来，从 cur --> forw 查找
                Line lc = cur_lines.at(best_j);
                for (int k = 0; k < forw_lines.size(); ++k)
                {
                    Line lk = forw_lines.at(k);

                    // condition 1
                    cv::Point2f d = lk.Center - lc.Center;
                    float dist = d.dot(d);
                    if( dist > dth) continue;  //
                    // condition 2
                    float delta_theta1 = fabs(lk.theta - lc.theta);
                    float delta_theta2 = 3.1415926 - delta_theta1;
                    if( delta_theta1 < th || delta_theta2 < th)
                    {
                        //std::cout << "theta: "<< lf.theta * 180 / 3.14259 <<" "<< lc.theta * 180 / 3.14259<<" "<<delta_theta1<<" "<<delta_theta2<<std::endl;
                        //candidate.push_back(lk);
                        //float cost = fabs(lk.image_dx - lc.image_dx) + fabs( lk.image_dy - lc.image_dy) + dist;
                        float cost = fabs(lk.line_grad_avg - lc.line_grad_avg) + dist/10.0;

                        if(cost < grad_err_min_i)
                        {
                            grad_err_min_i = cost;
                            best_i = k;
                        }
                    }

                }
            }

            if( grad_err_min_i < 50 && best_i == i){

                //std::cout<< "line match cost: "<<grad_err_min_j<<" "<<grad_err_min_i <<"\n\n";
                lineMatches.emplace_back(best_j,i);
            }
            /*
            vector<Line> l;
            l.push_back(lf);
            vector<Line> best;
            best.push_back(best_match);
            visualizeLineTrackCandidate(l,forwframe_->img,"forwframe_");
            visualizeLineTrackCandidate(best,curframe_->img,"curframe_best");
            visualizeLineTrackCandidate(candidate,curframe_->img,"curframe_");
            cv::waitKey(0);
            */
        }

    }




    void LineDetector::VisualizeLineMatch(cv::Mat imageMat1, cv::Mat imageMat2,
                                            std::vector<KeyLine> octave0_1, std::vector<KeyLine>octave0_2,
                                            std::vector<bool> good_matches){
        //	Mat img_1;
        cv::Mat img1,img2;
        if (imageMat1.channels() != 3){
            cv::cvtColor(imageMat1, img1, cv::COLOR_GRAY2BGR);
        }
        else{
            img1 = imageMat1;
        }

        if (imageMat2.channels() != 3){
            cv::cvtColor(imageMat2, img2, cv::COLOR_GRAY2BGR);
        }
        else{
            img2 = imageMat2;
        }

        //    srand(time(NULL));
        int lowest = 0, highest = 255;
        int range = (highest - lowest) + 1;
        for (int k = 0; k < good_matches.size(); ++k) {

            if(!good_matches[k]) continue;

            KeyLine line1 = octave0_1[k];  // trainIdx
            KeyLine line2 = octave0_2[k];  //queryIdx

            unsigned int r = lowest + int(rand() % range);
            unsigned int g = lowest + int(rand() % range);
            unsigned int b = lowest + int(rand() % range);
            cv::Point startPoint = cv::Point(int(line1.startPointX), int(line1.startPointY));
            cv::Point endPoint = cv::Point(int(line1.endPointX), int(line1.endPointY));
            cv::line(img1, startPoint, endPoint, cv::Scalar(r, g, b),2 ,8);

            cv::Point startPoint2 = cv::Point(int(line2.startPointX), int(line2.startPointY));
            cv::Point endPoint2 = cv::Point(int(line2.endPointX), int(line2.endPointY));
            cv::line(img2, startPoint2, endPoint2, cv::Scalar(r, g, b),2, 8);
            cv::line(img2, startPoint, startPoint2, cv::Scalar(0, 0, 255),1, 8);
            cv::line(img2, endPoint, endPoint2, cv::Scalar(0, 0, 255),1, 8);

        }
        /* plot matches */
        /*
        cv::Mat lsd_outImg;
        std::vector<char> lsd_mask( lsd_matches.size(), 1 );
        drawLineMatches( imageMat1, octave0_1, imageMat2, octave0_2, good_matches, lsd_outImg, Scalar::all( -1 ), Scalar::all( -1 ), lsd_mask,
        DrawLinesMatchesFlags::DEFAULT );

        imshow( "LSD matches", lsd_outImg );
        */
        cv::namedWindow("LSD matches1", CV_WINDOW_NORMAL);
        cv::namedWindow("LSD matches2", CV_WINDOW_NORMAL);
        imshow("LSD matches1", img1);
        imshow("LSD matches2", img2);
        cv::waitKey(1);
    }





    vector<cv::Scalar> getColorMap(){
        vector<cv::Scalar> colors;

        std::default_random_engine randomEngine;
        std::uniform_int_distribution<unsigned int> color_dist(0,255);

        for(int i=0;i<255;++i){
            cv::Scalar c(color_dist(randomEngine),color_dist(randomEngine),color_dist(randomEngine));
            colors.push_back(c);
        }
        return colors;
    }


    void LineDetector::VisualizeLine(cv::Mat &img,const FrameLines::Ptr &lines){
        if (img.channels() == 1){
            cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
        }

        //static vector<cv::Scalar> colors = getColorMap();

        int lowest = 0, highest = 255;
        int range = (highest - lowest) + 1;

        auto &keylsd = lines->keylsd;

        for (int k = 0; k < keylsd.size(); ++k) {
            cv::Point startPoint = cv::Point(int(keylsd[k].startPointX), int(keylsd[k].startPointY));
            cv::Point endPoint = cv::Point(int(keylsd[k].endPointX), int(keylsd[k].endPointY));
            //int idx = lines->line_ids[k] % 255;
            //cv::line(img, startPoint, endPoint,colors[idx],2 ,8);
            if(lines->track_cnt[lines->line_ids[k]] > 1){ //跟踪超过1次
                cv::line(img, startPoint, endPoint,cv::Scalar(255,0,0),2 ,8);
            }
            else{
                cv::line(img, startPoint, endPoint,cv::Scalar(0,0,255),2 ,8);
            }
        }

    }

    void LineDetector::VisualizeLineStereoMatch(cv::Mat &img, const FrameLines::Ptr &left_lines, const FrameLines::Ptr &right_lines){
        std::map<unsigned int,vector<KeyLine>> map_keylsd;
        int offset_y = img.rows / 2;

        int right_size = right_lines->keylsd.size();
        for(int i=0;i<right_size;++i){
            map_keylsd.insert({right_lines->line_ids[i],{}});
            map_keylsd[right_lines->line_ids[i]].push_back(right_lines->keylsd[i]);
        }
        int left_size = left_lines->keylsd.size();
        for(int i=0;i<left_size;++i){
            auto it=map_keylsd.find(left_lines->line_ids[i]);
            if(it!=map_keylsd.end()){
                it->second.push_back(left_lines->keylsd[i]);
            }
        }

        Debugt("LineDetector::VisualizeLineMatch map_keylsd size:{}",map_keylsd.size());

        for(auto &[line_id,vec_line]:map_keylsd){
            cv::Point left_start = cv::Point(int(vec_line[1].startPointX), int(vec_line[1].startPointY));
            cv::Point right_start = cv::Point(int(vec_line[0].startPointX), int(vec_line[0].startPointY) + offset_y);
            cv::line(img, left_start, right_start,cv::Scalar(0,128,0),1 ,8);

            cv::Point left_end = cv::Point(int(vec_line[1].endPointX), int(vec_line[1].endPointY));
            cv::Point right_end = cv::Point(int(vec_line[0].endPointX), int(vec_line[0].endPointY) + offset_y);
            cv::line(img, left_end, right_end,cv::Scalar(0,128,0),1 ,8);
        }

    }


    void LineDetector::VisualizeLineMonoMatch(cv::Mat &img, const FrameLines::Ptr &prev_lines, const FrameLines::Ptr &curr_lines){
        if(!prev_lines || !curr_lines){
            return;
        }

        std::map<unsigned int,vector<KeyLine>> map_keylsd;

        int right_size = curr_lines->keylsd.size();
        for(int i=0;i<right_size;++i){
            map_keylsd.insert({curr_lines->line_ids[i],{}});
            map_keylsd[curr_lines->line_ids[i]].push_back(curr_lines->keylsd[i]);
        }
        int left_size = prev_lines->keylsd.size();
        for(int i=0;i<left_size;++i){
            auto it=map_keylsd.find(prev_lines->line_ids[i]);
            if(it!=map_keylsd.end()){
                it->second.push_back(prev_lines->keylsd[i]);
            }
        }

        Debugt("LineDetector::VisualizeLineMatch map_keylsd size:{}",map_keylsd.size());

        for(auto &[line_id,vec_line]:map_keylsd){
            if(vec_line.size()<2){
                continue;
            }
            cv::Point left_start = cv::Point(int(vec_line[1].startPointX), int(vec_line[1].startPointY));
            cv::Point right_start = cv::Point(int(vec_line[0].startPointX), int(vec_line[0].startPointY));
            cv::line(img, left_start, right_start,cv::Scalar(0,128,0),1 ,8);

            cv::Point left_end = cv::Point(int(vec_line[1].endPointX), int(vec_line[1].endPointY));
            cv::Point right_end = cv::Point(int(vec_line[0].endPointX), int(vec_line[0].endPointY));
            cv::line(img, left_end, right_end,cv::Scalar(0,128,0),1 ,8);
        }

    }


    void LineDetector::VisualizeRightLine(cv::Mat &img,const FrameLines::Ptr &lines,bool vertical){
        //static vector<cv::Scalar> colors = getColorMap();
        if(vertical){
            int lowest = 0, highest = 255;
            int range = (highest - lowest) + 1;
            int offset_rows = img.rows / 2;

            auto &keylsd = lines->keylsd;

            for (int k = 0; k < keylsd.size(); ++k) {
                cv::Point startPoint = cv::Point(int(keylsd[k].startPointX), int(keylsd[k].startPointY)+offset_rows);
                cv::Point endPoint = cv::Point(int(keylsd[k].endPointX), int(keylsd[k].endPointY)+offset_rows);
                //int idx = lines->line_ids[k] % 255;
                //cv::line(img, startPoint, endPoint,colors[idx],2 ,8);
                if(lines->track_cnt[lines->line_ids[k]] > 1){ //跟踪超过1次
                    cv::line(img, startPoint, endPoint,cv::Scalar(255,0,0),2 ,8);
                }
                else{
                    cv::line(img, startPoint, endPoint,cv::Scalar(0,0,255),2 ,8);
                }
            }
        }
        else{
            int lowest = 0, highest = 255;
            int range = (highest - lowest) + 1;
            int offset_cols = img.cols / 2;

            auto &keylsd = lines->keylsd;

            for (int k = 0; k < keylsd.size(); ++k) {
                cv::Point startPoint = cv::Point(int(keylsd[k].startPointX+offset_cols), int(keylsd[k].startPointY));
                cv::Point endPoint = cv::Point(int(keylsd[k].endPointX+offset_cols), int(keylsd[k].endPointY));
                //int idx = lines->line_ids[k] % 255;
                //cv::line(img, startPoint, endPoint,colors[idx],2 ,8);
                if(lines->track_cnt[lines->line_ids[k]] > 1){ //跟踪超过1次
                    cv::line(img, startPoint, endPoint,cv::Scalar(255,0,0),2 ,8);
                }
                else{
                    cv::line(img, startPoint, endPoint,cv::Scalar(0,0,255),2 ,8);
                }
            }
        }



    }


}


