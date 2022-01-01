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

#include "../featureTracker/feature_tracker.h"




FeatureTracker::FeatureTracker()
{
    stereo_cam = false;
    n_id = 0;
    try{
        myLogger->info("init FeatureTracker");
        insts_tracker.reset(new InstsFeatManager);
    }
    catch(std::runtime_error &e){
        throw e;
    }
}


void FeatureTracker::setMask()
{
    mask = cv::Mat(row, col, CV_8UC1, cv::Scalar(255));

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.emplace_back(track_cnt[i], make_pair(cur_pts[i], ids[i]));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(),
         [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b){
            return a.first > b.first;
    });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &[t_cnt,pt_id] : cnt_pts_id){
        auto &[pt,id]=pt_id;
        if (mask.at<uchar>(pt) == 255){
            cur_pts.push_back(pt);
            ids.push_back(id);
            track_cnt.push_back(t_cnt);
            cv::circle(mask, pt, Config::MIN_DIST, 0, -1);
        }
    }

}




map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
FeatureTracker::trackImage(SegImage &img)
{
    TicToc t_r;
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
        TicToc t_o;
        vector<uchar> status = flowTrack(prev_img.gray0,img.gray0,prev_pts, cur_pts);
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        //printf("track cnt %d\n", (int)ids.size());
    }

    for (auto &n : track_cnt)
        n++;

        //rejectWithF();
        ROS_DEBUG("set mask_img begins");
        TicToc t_m;
        setMask();
        ROS_DEBUG("set mask_img costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t;
        int n_max_cnt = Config::MAX_CNT - static_cast<int>(cur_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask.empty())
                cout << "mask_img is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask_img type wrong " << endl;
            cv::goodFeaturesToTrack(cur_img.gray0, n_pts, Config::MAX_CNT - cur_pts.size(), 0.01, Config::MIN_DIST, mask);
        }
        else
            n_pts.clear();
        ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

        for (auto &p : n_pts){
            cur_pts.push_back(p);
            ids.push_back(n_id++);
            track_cnt.push_back(1);
        }
        //printf("feature cnt after add %d\n", (int)ids.size());

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    if(!cur_img.gray1.empty() && stereo_cam)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if(!cur_pts.empty())
        {
            vector<uchar> status= flowTrack(cur_img.gray0,cur_img.gray1,cur_pts, cur_right_pts);

            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }


    if(Config::SHOW_TRACK)
        drawTrack(cur_img, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    return setOutputFeats();
}


map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
FeatureTracker::setOutputFeats()
{
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> featureFrame;
    for (size_t i = 0; i < ids.size(); i++){
        int feature_id = ids[i];
        double x = cur_un_pts[i].x;
        double y = cur_un_pts[i].y;
        constexpr double z = 1;
        double p_u = cur_pts[i].x;
        double p_v = cur_pts[i].y;
        constexpr int camera_id = 0;
        double velocity_x = pts_velocity[i].x;
        double velocity_y = pts_velocity[i].y;
        Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
        xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
        featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
    }

    if (!cur_img.gray1.empty() && stereo_cam)
    {
        for (size_t i = 0; i < ids_right.size(); i++)
        {
            int feature_id = ids_right[i];
            double x = cur_un_right_pts[i].x;
            double y = cur_un_right_pts[i].y;
            constexpr double z = 1;
            double p_u = cur_right_pts[i].x;
            double p_v = cur_right_pts[i].y;
            constexpr int camera_id = 1;
            double velocity_x = right_pts_velocity[i].x;
            double velocity_y = right_pts_velocity[i].y;
            Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
            xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
            featureFrame[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
        }
    }
    //printf("feature track whole time %f\n", t_r.toc());
    return featureFrame;
}




map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>
FeatureTracker::trackImageNaive(SegImage &img)
{
    TicToc t_r;
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
    semantic_mask.release();

    TicToc tt;

    if(!Config::isInputSeg){
        cout<<"insts_info size:"<<img.insts_info.size()<<endl;
        if(! img.insts_info.empty()){
            semantic_mask = ~img.merger_mask;
        }
    }
    else if(Config::Dataset == DatasetType::VIODE){
        insts_tracker->mergeViodeSemanticMask(img.seg0, semantic_mask);
    }

    cout<<"infer time:"<<tt.toc_then_tic()<<endl;

    if (!prev_pts.empty())
    {
        TicToc t_o;
        cout<<"prev_pts.size():"<<prev_pts.size()<<endl;
        vector<uchar> status = flowTrack(prev_img.gray0,img.gray0,prev_pts, cur_pts);

        if(!semantic_mask.empty()){
            for(int i=0;i<status.size();++i){
                if(status[i] && semantic_mask.at<uchar>(cur_pts[i]) <128 )
                    status[i]=0;
            }
        }

        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
        //printf("track cnt %d\n", (int)ids.size());
    }

    cout<<"flowTrack left:"<<tt.toc_then_tic()<<endl;


    for (auto &n : track_cnt)
        n++;

    //rejectWithF();
    ROS_DEBUG("set mask_img begins");
    TicToc t_m;
    ROS_DEBUG("set mask_img costs %fms", t_m.toc());

    ROS_DEBUG("detect feature begins");
    TicToc t_t;
    int n_max_cnt = Config::MAX_CNT - static_cast<int>(cur_pts.size());
    if (n_max_cnt > 10)
    {
        setMask();

        if(mask.empty())
            cout << "mask_img is empty " << endl;
        if (mask.type() != CV_8UC1)
            cout << "mask_img type wrong " << endl;

        if(!semantic_mask.empty()){
            superpositionMask(mask,semantic_mask);
/*            cv::imshow("semantic_mask",semantic_mask);
            cv::imshow("mask_img",mask_img);
            cv::waitKey(1);*/
        }

        cv::goodFeaturesToTrack(cur_img.gray0, n_pts, (int)Config::MAX_CNT - cur_pts.size(), 0.01, Config::MIN_DIST, mask);
    }
    else
        n_pts.clear();
    ROS_DEBUG("detect feature costs: %f ms", t_t.toc());

    for (auto &p : n_pts)
    {
        cur_pts.push_back(p);
        ids.push_back(n_id++);
        track_cnt.push_back(1);
    }
    //printf("feature cnt after add %d\n", (int)ids.size());

    cout<<"detect feature:"<<tt.toc_then_tic()<<endl;

    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    cout<<"vel un:"<<tt.toc_then_tic()<<endl;

    if(!cur_img.gray1.empty() && stereo_cam)
    {
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if(!cur_pts.empty())
        {
            vector<uchar> status= flowTrack(cur_img.gray0,cur_img.gray1,cur_pts, cur_right_pts);
            if(Config::Dataset == DatasetType::VIODE){
                for(int i=0;i<status.size();++i){
                    if(status[i] && belongViodeDynamicOject(cur_right_pts[i], img.seg1))
                        status[i]=0;
                }
            }
            else if(!img.seg1.empty()){
                for(int i=0;i<status.size();++i){
                    if(status[i] && img.seg1.at<uchar>(cur_right_pts[i]) >0 )
                        status[i]=0;
                }
            }

            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            // only keep left-right pts
            /*
            reduceVector(cur_pts, status);
            reduceVector(ids, status);
            reduceVector(track_cnt, status);
            reduceVector(cur_un_pts, status);
            reduceVector(pts_velocity, status);
            */
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        prev_un_right_pts_map = cur_un_right_pts_map;
    }

    cout<<"flow_track right:"<<tt.toc_then_tic()<<endl;


    if(Config::SHOW_TRACK)
        drawTrack(cur_img, ids, cur_pts, cur_right_pts, prevLeftPtsMap);

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    return setOutputFeats();
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
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera[0]->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + col / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + row / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_prev_pts, cv::FM_RANSAC, Config::F_THRESHOLD, 0.99, status);
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, cur_pts.size(), 1.0 * cur_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}



void FeatureTracker::readIntrinsicParameter(const vector<string> &calib_file)
{
    for (const auto & i : calib_file){
        myLogger->info(fmt::format("readIntrinsicParameter() Reading parameter of camera:{}",i));
        camodocal::CameraPtr camera = CameraFactory::instance()->generateCameraFromYamlFile(i);
        myLogger->info(camera->parametersToString());
        m_camera.push_back(camera);
    }

    insts_tracker->camera=m_camera[0];
    if (calib_file.size() == 2){
        stereo_cam = true;
        insts_tracker->isStereo=true;
        insts_tracker->right_camera=m_camera[1];
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
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + col / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + row / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < row + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < col + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.gray0.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
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
    for (auto & pt : pts)
    {
        Eigen::Vector2d a(pt.x, pt.y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.emplace_back(b.x() / b.z(), b.y() / b.z());
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &id_vec, vector<cv::Point2f> &pts,
                                                map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts) const
{
    vector<cv::Point2f> pts_vel;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < id_vec.size(); i++){
        cur_id_pts.insert(make_pair(id_vec[i], pts[i]));
    }
    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(id_vec[i]);
            if (it != prev_id_pts.end()){
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_vel.emplace_back(v_x, v_y);
            }
            else
                pts_vel.emplace_back(0, 0);
        }
    }
    else
    {
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
                               map<int, cv::Point2f> &prevLeftPts){

    if(!insts_tracker->mask_background.empty()){
        cv::cvtColor(insts_tracker->mask_background,imTrack,CV_GRAY2BGR);
        cv::add(imTrack*0.5,img.color0,imTrack);
        //imTrack=img.color0;
    }
    else{
        imTrack = img.color0;
    }

    if (!img.color1.empty() && stereo_cam){
        if(Config::Dataset == DatasetType::KITTI){
            cv::vconcat(imTrack, img.color1, imTrack);
        }else{
            cv::hconcat(imTrack, img.color1, imTrack);
        }
    }
    //cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++){
        double len = std::min(1.0, 1.0 * track_cnt[j] / 20);
        cv::circle(imTrack, curLeftPts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
    }

    if (!img.color1.empty() && stereo_cam){
        if(Config::Dataset == DatasetType::KITTI){
            for (auto &rightPt : curRightPts){
                rightPt.y += (float)img.color0.rows;
                cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            }
        }
        else{
            for (auto &rightPt : curRightPts){
                rightPt.x += (float)img.color0.cols;
                cv::circle(imTrack, rightPt, 2, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    for (size_t i = 0; i < curLeftIds.size(); i++){
        if(auto it = prevLeftPts.find(curLeftIds[i]); it != prevLeftPts.end()){
            cv::arrowedLine(imTrack, curLeftPts[i], it->second, cv::Scalar(0, 255, 0), 1, 8, 0, 0.2);
        }
    }
}




void FeatureTracker::setPrediction(map<int, Eigen::Vector3d> &predictPts)
{
    predict_pts.clear();
    predict_pts_debug.clear();
    map<int, Eigen::Vector3d>::iterator itPredict;
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


void FeatureTracker::removeOutliers(set<int> &removePtsIds)
{
    std::set<int>::iterator itSet;
    vector<uchar> status;
    for (int & id : ids){
        itSet = removePtsIds.find(id);
        if(itSet != removePtsIds.end())
            status.push_back(0);
        else
            status.push_back(1);
    }

    reduceVector(prev_pts, status);
    reduceVector(ids, status);
    reduceVector(track_cnt, status);
}



/**
 * 叠加两个mask，结果写入到第一个maks中
 * @param mask1
 * @param mask2
 */
void FeatureTracker::superpositionMask(cv::Mat &mask1, const cv::Mat &mask2)
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


FeatureMap FeatureTracker::trackSemanticImage(SegImage &img)
{
    TicToc t_r,t_all;
    cur_time = img.time0;
    cur_img=img;
    row = cur_img.gray0.rows;
    col = cur_img.gray0.cols;
    cur_pts.clear();

    if constexpr(false){
        static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(img.gray0, img.gray0);
        if(!img.gray1.empty())
            clahe->apply(img.gray1, img.gray1);
    }

    if(!Config::isInputSeg){
        TicToc tt;
        tkLogger->debug("trackSemanticImage insts_info.size:{}",img.insts_info.size());
        semantic_mask = insts_tracker->addInstances(img);
    }
    else if(Config::Dataset == DatasetType::VIODE){
        semantic_mask=insts_tracker->addViodeInstances(cur_img.seg0);//根据实例分割结果创建实例，并设置语义mask
    }
    tkLogger->info("trackSemanticImage 设置语义mask:{} ms",t_r.toc_then_tic());

    ///开启另一个线程检测动态特征点
    TicToc t_i;
    std::thread t_inst_track(&InstsFeatManager::instsTrack, insts_tracker.get(), cur_img, prev_img);

    /// 跟踪特征点
    if (!prev_pts.empty()){
        vector<uchar> status=flowTrack(prev_img.gray0,cur_img.gray0,prev_pts,cur_pts);
        if(!semantic_mask.empty()){
            for(int i=0;i<cur_pts.size();i++){
                if(status[i] && semantic_mask.at<uchar>(cur_pts[i])==0)
                    status[i]=0;
            }
        }
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        for (auto &cnt : track_cnt) cnt++; //将各个特征点的跟踪次数++
    }
    tkLogger->info("trackSemanticImage 跟踪特征点:{} ms",t_r.toc_then_tic());

    /// 检测新的角点
    //rejectWithF();
    int n_max_cnt = Config::MAX_CNT - static_cast<int>(cur_pts.size());
    if (n_max_cnt > 10){
        setMask(); //设置mask，用于将检测的特征点分散开来
        if(!semantic_mask.empty()){
            ///对semantic_mask进行腐蚀，扩大黑色区域,防止检测的特征点落在动态物体的边缘
            cv::Mat erode_mask;
            static cv::Mat mask_element=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(10,10),cv::Point(-1,-1));
            cv::erode(semantic_mask,erode_mask,mask_element,cv::Point(-1,-1));
            ///mask叠加
            superpositionMask(mask,erode_mask);
            //setSemanticMask(mask_img,cur_img.seg0);
            //测试
            //cv::imshow("mask_img",semantic_mask);
            //cv::waitKey(0);
            //cv::Mat dst;
            //imageTranslate(img.gray0,dst,-100,-100);
            //cv::imshow("bg_mask",dst);
            //setMaskBaseSemanticMask();
        }

        cv::goodFeaturesToTrack(cur_img.gray0, n_pts, n_max_cnt, 0.01, Config::MIN_DIST, mask); //检测新的角点
        for (auto &p : n_pts){
            cur_pts.push_back(p);//
            ids.push_back(n_id++); //给每个角点分配一个ID
            track_cnt.push_back(1);
        }
    }
    else{
        n_pts.clear();
    }
    tkLogger->info("trackSemanticImage 检测新的角点:{} ms",t_r.toc_then_tic());

    ///对特征点进行去畸变,返回归一化特征点
    cur_un_pts = undistortedPts(cur_pts, m_camera[0]);
    /// 速度跟踪
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);
    tkLogger->info("trackSemanticImage 去畸变&&速度跟踪:{} ms",t_r.toc_then_tic());

    /// 右边相机图像的跟踪
    if(!cur_img.color1.empty() && stereo_cam){
        ids_right.clear();
        cur_right_pts.clear();
        cur_un_right_pts.clear();
        right_pts_velocity.clear();
        cur_un_right_pts_map.clear();
        if(!cur_pts.empty()){
            vector<uchar> status=flowTrack(cur_img.gray0,cur_img.gray1,cur_pts,cur_right_pts);
            if(Config::Dataset == DatasetType::VIODE){
                for(int i=0;i<cur_pts.size();i++){
                    if(status[i]){
                        auto key= pixel2label(cur_right_pts[i],img.seg1);
                        if(Config::VIODE_DynamicLabelID.count(Config::VIODE_RGB2Label[key]) != 0){
                            status[i]=0;
                        }
                    }
                }
            }
            ids_right = ids;
            reduceVector(cur_right_pts, status);
            reduceVector(ids_right, status);
            cur_un_right_pts = undistortedPts(cur_right_pts, m_camera[1]);
            right_pts_velocity = ptsVelocity(ids_right, cur_un_right_pts, cur_un_right_pts_map, prev_un_right_pts_map);
        }
        //printf("cur_right_pts.size:%d\n",cur_right_pts.size());
        prev_un_right_pts_map = cur_un_right_pts_map;
    }
    tkLogger->info("trackSemanticImage 右边相机图像的跟踪:{} ms",t_r.toc_then_tic());

    ///光流跟踪的可视化
    if(Config::SHOW_TRACK)
        drawTrack(cur_img, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
    //t_r.toc_print_tic("trackSemanticImage 光流跟踪的可视化");//0.9ms


    t_inst_track.join();

    tkLogger->info("trackSemanticImage 动态检测线程总时间:{} ms",t_i.toc_then_tic());

    if(Config::SHOW_TRACK)
        insts_tracker->drawInsts(imTrack);

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

    //insts_tracker->visualizeInst(img.color0);
    //cv::imshow("inst_mask",imTrack);
    //cv::waitKey(1);

    /// 将归一化坐标、像素坐标、速度组合在一起
    return setOutputFeats();
}

