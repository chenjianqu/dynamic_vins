/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "instance.h"
#include "estimator.h"



void Instance::GetBoxVertex(EigenContainer<Eigen::Vector3d> &vertex) {
    Eigen::Vector3d minPt,maxPt;
    minPt = - box;
    maxPt = box;

    vertex.resize(8);

    vertex[0]=minPt;

    vertex[1].x()=maxPt.x();vertex[1].y()=minPt.y();vertex[1].z()=minPt.z();
    vertex[2].x()=maxPt.x();vertex[2].y()=minPt.y();vertex[2].z()=maxPt.z();
    vertex[3].x()=minPt.x();vertex[3].y()=minPt.y();vertex[3].z()=maxPt.z();
    vertex[4].x()=minPt.x();vertex[4].y()=maxPt.y();vertex[4].z()=maxPt.z();

    vertex[5] = maxPt;

    vertex[6].x()=maxPt.x();vertex[6].y()=maxPt.y();vertex[6].z()=minPt.z();
    vertex[7].x()=minPt.x();vertex[7].y()=maxPt.y();vertex[7].z()=minPt.z();

    for(int i=0;i<8;++i){
        vertex[i] = state[kWindowSize].R * vertex[i] + state[kWindowSize].P;
    }
}


/**
 * 根据速度推导滑动窗口中各个物体的位姿
 * @param e
 */
void Instance::SetWindowPose()
{
    //DebugV("SetWindowPose Inst:{} 起始位姿:<{}> 速度:v<{}> a{}>",id,vec2str(state[0].P),vec2str(vel.v),VecToStr(vel.a) );
    for(int i=1; i <= kWindowSize; i++){
        state[i].time=e->headers[i];
        double time_ij=state[i].time - state[0].time;
        Eigen::Matrix3d Roioj=Sophus::SO3d::exp(vel.a*time_ij).matrix();
        Eigen::Vector3d Poioj=vel.v*time_ij;
        state[i].R = Roioj * state[0].R;
        state[i].P = Roioj * state[0].P + Poioj;
    }
}


/**
 * 物体的位姿初始化,并将之前的观测删去
 * 由于动态物体的运动，因此选择某一帧中所有三角化的路标点的世界坐标作为初始P，初始R为单位阵
 * @param estimator
 */
void Instance::InitialPose()
{
    if(is_initial) return;

    ///初始化的思路是找到某一帧，该帧拥有已经三角化的特征点的开始帧数量最多。
    int cnt[kWindowSize + 1]={0};
    for(auto &lm : landmarks){
        if(lm.depth > 0){
            cnt[lm.feats[0].frame]++;
        }
    }
    int frame_cnt=-1;//将作为初始化位姿的帧号
    int cnt_max=0;
    int cnt_sum=0;
    for(int i=0; i <= kWindowSize; ++i){
        if(cnt[i]>cnt_max){
            cnt_max=cnt[i];
            frame_cnt=i;
        }
        cnt_sum+=cnt[i];
    }

    //太少了
    if(cnt_sum<2){
        return;
    }

    if(frame_cnt >= 0)
    {
        ///计算初始位姿
        Eigen::Vector3d center=Eigen::Vector3d::Zero(),minPt=center,maxPt=center;
        int index=0;
        for(auto &landmark : landmarks){
            if(landmark.depth > 0 && landmark.feats[0].frame == frame_cnt){
                Eigen::Vector3d point_cam= landmark.feats[0].point * landmark.depth;//相机坐标
                auto point_imu=e->ric[0] * point_cam + e->tic[0];//IMU坐标
                Eigen::Vector3d p=e->Rs[frame_cnt] * point_imu + e->Ps[frame_cnt];//世界坐标
                center+=p;
                index++;
                if(p.x() < minPt.x()) minPt.x()=p.x();
                if(p.x() > maxPt.x()) maxPt.x()=p.x();
                if(p.y() < minPt.y()) minPt.y()=p.y();
                if(p.y() > maxPt.y()) maxPt.y()=p.y();
                if(p.z() < minPt.z()) minPt.z()=p.z();
                if(p.z() > maxPt.z()) maxPt.z()=p.z();
            }
        }
        vel.SetZero();

        state[0].P=center/index;
        state[0].R=Eigen::Matrix3d::Identity();
        state[0].time=e->headers[0];
        SetWindowPose();


        /*box.x()=(box_max_pt.x()-box_min_pt.x())/2.0;
        box.y()=(box_max_pt.y()-box_min_pt.y())/2.0;
        box.z()=(box_max_pt.z()-box_min_pt.z())/2.0;*/
        box = Vec3d::Ones();
        is_initial=true;

        DebugV("Instance:{} 初始化成功,cnt_max:{} init_frame:{} 初始位姿:P<{}> 初始box:<{}>",
               id, cnt_max, frame_cnt, VecToStr(center), VecToStr(box));

        ///删去初始化之前的观测
        for(auto it=landmarks.begin(),it_next=it;it!=landmarks.end();it=it_next){
            it_next++;
            if(it->feats.size() == 1 && it->feats[0].frame < frame_cnt)
                landmarks.erase(it);
        }
        for(auto &lm:landmarks){
            if(lm.feats[0].frame < frame_cnt){
                for(auto it=lm.feats.begin(),it_next=it; it != lm.feats.end(); it=it_next){
                    it_next++;
                    if(it->frame < frame_cnt) lm.feats.erase(it); ///删掉掉前面的观测
                }
                if(lm.depth > 0) lm.depth=-1.0;///需要重新进行三角化
            }
        }
    }


}




/**
 * 滑动窗口去除最老帧的位姿和点，并将估计的深度转移到次老帧,这个函数应该先于estimator的滑动窗口函数
 */
int Instance::SlideWindowOld()
{
    ///为了加快速度，先计算用于求坐标系变换的临时矩阵.使用的公式来自附录的第12条，temp_x表示公式的第x项
    Mat3d R_margin;
    Vec3d t_margin;
    for(auto &landmark : landmarks){
        if(landmark.feats[0].frame == 0 && landmark.feats.size() > 1 && landmark.feats[1].frame == 1){ //此时才有必要计算
            auto &R_bc = e->ric[0];
            auto R_cb = R_bc.transpose();
            auto &P_bc = e->tic[0];
            auto temp_5 = -R_cb * P_bc;
            auto temp_RcbRbiw = R_cb * e->Rs[1].transpose();
            auto temp_4 = temp_RcbRbiw * (state[1].P - e->Ps[1]);
            auto temp_RcbRbiwRwoiRojw=temp_RcbRbiw * state[1].R * state[0].R.transpose();
            auto temp_3 = temp_RcbRbiwRwoiRojw * (e->Ps[0] - state[0].P);
            auto temp_2 = temp_RcbRbiwRwoiRojw * e->Rs[0] * P_bc;
            auto temp_1 = temp_RcbRbiwRwoiRojw * e->Rs[0] * R_bc;

            R_margin=temp_1;
            t_margin=temp_2+temp_3+temp_4+temp_5;
            break;
        }
    }

    int debug_num=0;

    ///改变所有路标点的start_frame ,并保留最老帧中的特征点，寻找一个新的归宿
    for(auto it=landmarks.begin(),it_next=it;it!=landmarks.end();it=it_next)
    {
        it_next++;

        ///测试
        //printf("(");
        //for(auto& feat:it->feats)
        //    printf("%d ",feat.frame);
        //printf(" | ");

        if(it->feats[0].frame != 0)
        {
            for(auto &feat : it->feats) feat.frame--;
        }
        //该路标点只有刚开始这个观测，故将其删掉
        else if(it->feats.size() <= 1)
        {
            landmarks.erase(it);
            debug_num++;
            continue;
        }
        //将估计的逆深度值转移到新的开始帧
        else
        {
            Eigen::Vector3d point_old=it->feats[0].point;
            //首先删除该观测
            it->feats.erase(it->feats.begin());

            //计算新的深度
            if(it->depth > 0){
                auto pts_cam_j= point_old * it->depth;
                Eigen::Vector3d pts_cam_i;
                if(it->feats[0].frame == 1){ //满足了使用预计算的矩阵的条件，可减少计算量
                    pts_cam_i = R_margin * pts_cam_j + t_margin;
                }
                else{
                    int frame=it->feats[0].frame;
                    auto &R_bc = e->ric[0];
                    auto R_cb = R_bc.transpose();
                    auto &P_bc = e->tic[0];
                    auto temp_5 = -R_cb * P_bc;
                    auto temp_RcbRbiw = R_cb * e->Rs[frame].transpose();
                    auto temp_4 = temp_RcbRbiw * (state[frame].P - e->Ps[frame]);
                    auto temp_RcbRbiwRwoiRojw=temp_RcbRbiw * state[frame].R * state[0].R.transpose();
                    auto temp_3 = temp_RcbRbiwRwoiRojw * (e->Ps[0] - state[0].P);
                    auto temp_2 = temp_RcbRbiwRwoiRojw * e->Rs[0] * P_bc;
                    auto temp_1 = temp_RcbRbiwRwoiRojw * e->Rs[0] * R_bc;
                    R_margin=temp_1;
                    t_margin=temp_2+temp_3+temp_4+temp_5;
                    pts_cam_i = R_margin * pts_cam_j + t_margin;
                }
                if(pts_cam_i.z() > 0)
                    it->depth=pts_cam_i.z();
            }

            //将观测的frame--
            for(auto & feature_point : it->feats)
                feature_point.frame--;
        }


        ///测试
        //for(auto& feat:it->feats)
        //    printf("%d ",feat.frame);
        //printf(")\n");
    }

    ///将最老帧的相关变量去掉
    for (int i = 0; i < kWindowSize; i++){
        state[i].swap(state[i+1]);
    }
    state[kWindowSize]=state[kWindowSize - 1];

    //state[0]=state[1];
    //SetWindowPose(e);

    return debug_num;
}


/**
 * 滑动窗口去除次新帧的位姿和观测，并将最新帧移到次新帧
 */
int Instance::SlideWindowNew()
{
    int debug_num=0;

    for (auto it = landmarks.begin(), it_next = it; it != landmarks.end(); it = it_next)
    {
        it_next++;

        ///测试
        //printf("(");
        //for(auto& feat:it->feats)printf("%d ",feat.frame);
        //printf(" | ");


        if(it->feats.empty()){
            landmarks.erase(it);
            debug_num++;
            continue;
        }

        int index=-1;
        for(auto& feat:it->feats){
            index++;
            if(feat.frame == e->frame_count-1)
                it->feats.erase(it->feats.begin() + index);
        }
        for(auto& feat:it->feats){
            if(feat.frame == e->frame_count)
                feat.frame--;
        }


        ///测试
        //for(auto& feat:it->feats)printf("%d ",feat.frame);
        //printf(")\n");
    }


    state[kWindowSize - 1] = state[kWindowSize];

    return debug_num;
}


void Instance::SetCurrentPoint3d()
{
    point3d_curr.clear();


    for(auto &landmark : landmarks){
        if(landmark.depth > 0){
            bool isPresent=false;
            for(auto &feat : landmark.feats){
                if(feat.frame == e->frame_count-1){//因为实在slidewindows函数后面，所以需要-1
                    isPresent=true;
                    break;
                }
            }
            if(!isPresent)
                continue;

            int frame_j=landmark.feats[0].frame;
            int frame_i=e->frame_count;
            Eigen::Vector3d pts_cam_j = landmark.feats[0].point * landmark.depth;//k点在j时刻的相机坐标
            Eigen::Vector3d pts_imu_j = e->ric[0] * pts_cam_j + e->tic[0];//k点在j时刻的IMU坐标
            Eigen::Vector3d pts_w_j=e->Rs[frame_j] * pts_imu_j + e->Ps[frame_j];//k点在j时刻的世界坐标

            Eigen::Vector3d pts_obj_j=state[frame_j].R.transpose() * (pts_w_j - state[frame_j].P);//k点在j时刻的物体坐标
            Eigen::Vector3d pts_w_i=state[frame_i].R * pts_obj_j + state[frame_i].P;//k点在i时刻的世界坐标
            point3d_curr.push_back(pts_w_i);
            //point3d_curr.push_back(pts_w_j);

            ///测试
            /*
            if(id==114119232){
                printf("%d:(%.2lf,%.2lf,d:%.2lf,%d) to (%.3lf,%.3lf,%.3lf)\n",landmark.id,landmark.feats[0].point.x(),
                       landmark.feats[0].point.y(),landmark.depth,landmark.start_frame,
                       pts_w_j.x(),pts_w_j.y(),pts_w_j.z());
            }
             */

        }
    }


}


/**
 * 两帧中的重投影误差
 * @param feat_j
 * @param feat_i
 * @param e
 * @param depth
 * @param isStereo
 * @return
 */
double Instance::ReprojectTwoFrameError(FeaturePoint &feat_j, FeaturePoint &feat_i, double depth, bool isStereo)
{
    Eigen::Vector2d delta_j((e->td - feat_j.td) * feat_j.vel);
    //Eigen::Vector3d pts_j_td = feat_j.point - Eigen::Vector3d(delta_j.x(),delta_j.y(),0);
    Eigen::Vector3d pts_j_td = feat_j.point;

    Eigen::Vector2d delta_i;
    if(isStereo)
        delta_i= (e->td - feat_i.td) * feat_i.vel_right;
    else
        delta_i= (e->td - feat_i.td) * feat_i.vel;
    //Eigen::Vector3d pts_i_td = feat_i.point - Eigen::Vector3d(delta_i.x(),delta_i.y(),0);
    Eigen::Vector3d pts_i_td = feat_i.point;

    Eigen::Vector3d pts_imu_j=e->ric[0] * (pts_j_td / depth) + e->tic[0];//k点在j时刻的IMU坐标
    Eigen::Vector3d pts_w_j=e->Rs[feat_j.frame]*pts_imu_j + e->Ps[feat_j.frame];//k点在j时刻的世界坐标
    //Eigen::Vector3d pts_obj_j=Q_woj.inverse()*(pts_w_j-P_woj);//k点在j时刻的物体坐标
    //Eigen::Vector3d pts_w_i=Q_woi*pts_obj_j+P_woi;//k点在i时刻的世界坐标
    Eigen::Vector3d pts_imu_i=e->Rs[feat_i.frame].transpose()*(pts_w_j- e->Ps[feat_i.frame]);//k点在i时刻的IMU坐标
    Eigen::Vector3d pts_cam_i;
    if(isStereo)
        pts_cam_i=e->ric[1].transpose()*(pts_imu_i - e->tic[1]);
    else
        pts_cam_i=e->ric[0].transpose()*(pts_imu_i - e->tic[0]);

    Eigen::Vector2d residual = (pts_cam_i / pts_cam_i.z()).head<2>() - pts_i_td.head<2>();

    return std::sqrt(residual.x() * residual.x() + residual.y() * residual.y());
}


void Instance::OutlierRejection()
{
    if(!is_initial || !is_tracking)
        return;
    int num_delete=0,index=0;
    DebugV("Inst:{} landmark_num:{} box:<{}>", id, landmarks.size(), VecToStr(box));

    std::string debug_msg;
    string lm_msg;

    for(auto it=landmarks.begin(),it_next=it;it!=landmarks.end();it=it_next)
    {
        it_next++;
        auto &lm=*it;
        if(lm.feats.empty() || lm.depth <= 0)
            continue;
        if(int frame=lm.feats[0].frame; !IsInBox(
                e->Rs[frame], e->Ps[frame], e->ric[0], e->tic[0], state[frame].R,
                state[frame].P, lm.depth, lm.feats[0].point, box)){
            debug_msg += fmt::format("lid:{} ", lm.id);
            landmarks.erase(it);
            num_delete++;
            continue;
        }

        lm_msg += fmt::format("\nlid:{} depth:{:.2f} ", lm.id,lm.depth);

        double err = 0;
        int err_cnt = 0;
        ///单目重投影误差
        for(int i=1;i<(int)lm.feats.size(); ++i){
            //double repro_e= ReprojectTwoFrameError(lm.feats[0],lm.feats[i],e,lm.depth,false);
            /*double repro_e=ReprojectError(e->Rs[lm.feats[0].frame], e->Ps[lm.feats[0].frame],e->ric[0], e->tic[0],
                                             e->Rs[lm.feats[i].frame], e->Ps[lm.feats[i].frame],e->ric[0], e->tic[0],
                                             lm.depth,lm.feats[0].point,lm.feats[i].point);*/
            int imu_i = lm.feats[0].frame;
            int imu_j = lm.feats[i].frame;

            Eigen::Vector3d pts_w = e->Rs[imu_i] * (e->ric[0] * (lm.depth * lm.feats[0].point) + e->tic[0]) + e->Ps[imu_i];
            Eigen::Vector3d pts_oi=state[imu_i].R.transpose() * ( pts_w-state[imu_i].P);
            Eigen::Vector3d pts_wj=state[imu_j].R * pts_oi + state[imu_j].P;
            Eigen::Vector3d pts_cj = e->ric[0].transpose() * (e->Rs[imu_j].transpose() * (pts_wj - e->Ps[imu_j]) - e->tic[0]);
            Eigen::Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - lm.feats[i].point.head<2>();
            double re = residual.norm();

            err+=re;
            err_cnt++;
            lm_msg += fmt::format("M({},{},{:.2f}) ", lm.feats[0].frame, lm.feats[i].frame, re * kFocalLength);
        }
        if(lm.feats.size()>5) lm_msg+="\n";
        ///双目重投影误差
        for(int i=0;i<(int)lm.feats.size(); ++i){
            if(lm.feats[i].is_stereo){
                //double repro_e= ReprojectTwoFrameError(lm.feats[0],lm.feats[i],e,lm.depth,true);
                /*double repro_e=ReprojectError(e->Rs[lm.feats[0].frame], e->Ps[lm.feats[0].frame],e->ric[0], e->tic[0],
                                                 e->Rs[lm.feats[i].frame], e->Ps[lm.feats[i].frame],e->ric[1], e->tic[1],
                                                 lm.depth,lm.feats[0].point,lm.feats[i].point_right);*/
                /*double re= ReprojectDynamicError(
                        e->Rs[lm.feats[0].frame], e->Ps[lm.feats[0].frame], e->ric[0], e->tic[0],
                        state[lm.feats[0].frame].R, state[lm.feats[0].frame].P, e->Rs[lm.feats[i].frame],
                        e->Ps[lm.feats[i].frame], e->ric[1], e->tic[1], state[lm.feats[i].frame].R,
                        state[lm.feats[i].frame].P, lm.depth, lm.feats[0].point, lm.feats[i].point_right);*/
                int imu_i = lm.feats[0].frame;
                int imu_j = lm.feats[i].frame;
                Eigen::Vector3d pts_w = e->Rs[imu_i] * (e->ric[0] * (lm.depth * lm.feats[0].point) + e->tic[0]) + e->Ps[imu_i];
                Eigen::Vector3d pts_oi=state[imu_i].R.transpose() * ( pts_w-state[imu_i].P);
                Eigen::Vector3d pts_wj=state[imu_j].R * pts_oi + state[imu_j].P;
                Eigen::Vector3d pts_cj = e->ric[1].transpose() * (e->Rs[imu_j].transpose() * (pts_wj - e->Ps[imu_j]) - e->tic[1]);
                Eigen::Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - lm.feats[i].point.head<2>();
                double re = residual.norm();

                err+=re;
                err_cnt++;
                lm_msg += fmt::format("S({},{},{:.2f}) ", lm.feats[0].frame, lm.feats[i].frame, re * kFocalLength);
            }
        }
        double ave_err = err / err_cnt * kFocalLength;
        index++;
        if(ave_err > 10){
            DebugV("del lid:{} ,avg:{:.2f},d:{:.2f}> ", lm.id, ave_err, lm.depth);
            landmarks.erase(it);
            num_delete++;
        }
    }

    DebugV(lm_msg);
    DebugV("outbox:{}", debug_msg);
    DebugV("Inst:{} delete num:{}", id, num_delete);
}




void Instance::SetOptimizeParameters()
{
    para_speed[0][0] = vel.v.x();
    para_speed[0][1] = vel.v.y();
    para_speed[0][2] = vel.v.z();
    para_speed[0][3] = vel.a.x();
    para_speed[0][4] = vel.a.y();
    para_speed[0][5] = vel.a.z();
    para_box[0][0]=box.x();
    para_box[0][1]=box.y();
    para_box[0][2]=box.z();

    for(int i=0; i <= kWindowSize; ++i){
        para_state[i][0]=state[i].P.x();
        para_state[i][1]=state[i].P.y();
        para_state[i][2]=state[i].P.z();
        Eigen::Quaterniond q(state[i].R);
        para_state[i][3]=q.x();
        para_state[i][4]=q.y();
        para_state[i][5]=q.z();
        para_state[i][6]=q.w();
    }

    int index=-1;
    for(auto &landmark : landmarks){
        if(landmark.depth > 0){
            index++;
            para_inv_depth[index][0]= 1.0 / landmark.depth;
        }
    }
}




void Instance::GetOptimizationParameters()
{
    last_vel=vel;

    if(opt_vel){
        vel.v.x()=para_speed[0][0];
        vel.v.y()=para_speed[0][1];
        vel.v.z()=para_speed[0][2];
        vel.a.x()=para_speed[0][3];
        vel.a.y()=para_speed[0][4];
        vel.a.z()=para_speed[0][5];
    }
    box.x()=para_box[0][0];
    box.y()=para_box[0][1];
    box.z()=para_box[0][2];


    /*for(int i=0;i<=kWindowSize;++i){
        state[i].P.x()=para_state[i][0];
        state[i].P.y()=para_state[i][1];
        state[i].P.z()=para_state[i][2];
        Eigen::Quaterniond q(para_state[i][6],para_state[i][3],para_state[i][4],para_state[i][5]);
        q.normalize();
        state[i].R=q.toRotationMatrix();
    }*/
    state[0].P.x()=para_state[0][0];
    state[0].P.y()=para_state[0][1];
    state[0].P.z()=para_state[0][2];
    Eigen::Quaterniond q(para_state[0][6], para_state[0][3], para_state[0][4], para_state[0][5]);
    q.normalize();
    state[0].R=q.toRotationMatrix();
    SetWindowPose();

    int index=-1;
    for(auto &landmark : landmarks){
        if(landmark.depth > 0){
            index++;
            landmark.depth= 1.0 / para_inv_depth[index][0];
        }
    }
}










