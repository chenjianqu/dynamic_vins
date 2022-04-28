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

namespace dynamic_vins{\


void Instance::GetBoxVertex(EigenContainer<Vec3d> &vertex) {
    Vec3d minPt,maxPt;
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
        vertex[i] = state[kWinSize].R * vertex[i] + state[kWinSize].P;
    }
}


/**
 * 根据速度推导滑动窗口中各个物体的位姿
 */
void Instance::SetWindowPose()
{
    for(int i=1; i <= kWinSize; i++){
        state[i].time=e->headers[i];
        double time_ij=state[i].time - state[0].time;
        Mat3d Roioj=Sophus::SO3d::exp(vel.a*time_ij).matrix();
        Vec3d Poioj=vel.v*time_ij;
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
    if(is_initial || !is_tracking)
        return;

    ///初始化的思路是找到某一帧，该帧拥有已经三角化的特征点的开始帧数量最多。
    int win_cnt[kWinSize + 1]={0};
    for(auto &lm : landmarks){
        if(lm.depth > 0){
            win_cnt[lm.feats[0].frame]++;
        }
    }
    int frame_index=-1;//将作为初始化位姿的帧号
    int cnt_max=0, cnt_sum=0;
    for(int i=0; i <= kWinSize; ++i){
        if(win_cnt[i] > cnt_max){
            cnt_max=win_cnt[i];
            frame_index=i;
        }
        cnt_sum+=win_cnt[i];
    }
    if(cnt_sum < kInstanceInitMinNum)
        return;

    ///计算初始位姿
    Vec3d center=Vec3d::Zero(),minPt=center,maxPt=center;
    for(auto &landmark : landmarks){
        if(landmark.depth > 0 && landmark.feats[0].frame == frame_index){
            Vec3d point_cam= landmark.feats[0].point * landmark.depth;//相机坐标
            auto point_imu=e->ric[0] * point_cam + e->tic[0];//IMU坐标
            Vec3d p= e->Rs[frame_index] * point_imu + e->Ps[frame_index];//世界坐标
            center+=p;

            if(p.x() < minPt.x()) minPt.x()=p.x();
            if(p.x() > maxPt.x()) maxPt.x()=p.x();
            if(p.y() < minPt.y()) minPt.y()=p.y();
            if(p.y() > maxPt.y()) maxPt.y()=p.y();
            if(p.z() < minPt.z()) minPt.z()=p.z();
            if(p.z() > maxPt.z()) maxPt.z()=p.z();
        }
    }
    vel.SetZero();

    state[0].P=center/double(cnt_sum);
    state[0].R=Mat3d::Identity();
    state[0].time=e->headers[0];
    SetWindowPose();

    /*box.x()=(box_max_pt.x()-box_min_pt.x())/2.0;
    box.y()=(box_max_pt.y()-box_min_pt.y())/2.0;
    box.z()=(box_max_pt.z()-box_min_pt.z())/2.0;*/
    box = Vec3d::Ones();
    is_initial=true;

    Debugv("Instance:{} 初始化成功,cnt_max:{} init_frame:{} 初始位姿:P<{}> 初始box:<{}>",
           id, cnt_max, frame_index, VecToStr(center), VecToStr(box));

    ///删去初始化之前的观测
    for(auto it=landmarks.begin(),it_next=it;it!=landmarks.end();it=it_next){
        it_next++;
        if(it->feats.size() == 1 && it->feats[0].frame < frame_index){
            landmarks.erase(it);
        }
    }
    for(auto &lm:landmarks){
        if(lm.feats[0].frame < frame_index){
            for(auto it=lm.feats.begin(),it_next=it; it != lm.feats.end(); it=it_next){
                it_next++;
                if(it->frame < frame_index) {
                    lm.feats.erase(it); ///删掉掉前面的观测
                }
            }
            if(lm.depth > 0){
                lm.depth=-1.0;///需要重新进行三角化
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
        if(it->feats[0].frame != 0){
            for(auto &feat : it->feats)
                feat.frame--;
        }
        else if(it->feats.size() <= 1){ ///该路标点只有刚开始这个观测，故将其删掉
            if(it->depth>0){
                triangle_num--;
                depth_sum -= it->depth;
            }
            landmarks.erase(it);
            debug_num++;
            continue;
        }
        else{ ///将估计的逆深度值转移到新的开始帧
            Vec3d point_old=it->feats[0].point;
            it->feats.erase(it->feats.begin());//首先删除该观测
            ///计算新的深度
            if(it->depth > 0){
                depth_sum -= it->depth;
                auto pts_cam_j= point_old * it->depth;
                Vec3d pts_cam_i;
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
                //设置深度
                if(pts_cam_i.z() > 0){
                    it->depth=pts_cam_i.z();
                    depth_sum += it->depth;
                }
                else{
                    it->depth = -1;
                    triangle_num--;
                }
            }

            //将观测的frame--
            for(auto & feature_point : it->feats)
                feature_point.frame--;
        }
    }

    ///将最老帧的相关变量去掉
    for (int i = 0; i < kWinSize; i++){
        state[i].swap(state[i+1]);
    }
    state[kWinSize]=state[kWinSize - 1];

    return debug_num;
}


/**
 * 滑动窗口去除次新帧的位姿和观测，并将最新帧移到次新帧
 */
int Instance::SlideWindowNew()
{
    int debug_num=0;

    for (auto it = landmarks.begin(), it_next = it; it != landmarks.end(); it = it_next){
        it_next++;

        if(it->feats.empty()){
            landmarks.erase(it);
            debug_num++;
            continue;
        }

        int index=-1;
        for(auto& feat:it->feats){
            index++;
            if(feat.frame == e->frame - 1){
                it->feats.erase(it->feats.begin() + index);
            }
        }
        for(auto& feat:it->feats){
            if(feat.frame == e->frame)
                feat.frame--;
        }
    }

    state[kWinSize - 1] = state[kWinSize];
    return debug_num;
}

/**
 * 将路标点转换为到世界坐标系
 */
void Instance::SetCurrentPoint3d()
{
    point3d_curr.clear();
    for(auto &lm : landmarks){
        if(lm.depth <= 0)
            continue;
        int frame_j=lm.feats[0].frame;
        int frame_i=e->frame;
        Vec3d pts_cam_j = lm.feats[0].point * lm.depth;//k点在j时刻的相机坐标
        Vec3d pts_imu_j = e->ric[0] * pts_cam_j + e->tic[0];//k点在j时刻的IMU坐标
        Vec3d pts_w_j=e->Rs[frame_j] * pts_imu_j + e->Ps[frame_j];//k点在j时刻的世界坐标
        Vec3d pts_obj_j=state[frame_j].R.transpose() * (pts_w_j - state[frame_j].P);//k点在j时刻的物体坐标
        Vec3d pts_w_i=state[frame_i].R * pts_obj_j + state[frame_i].P;//k点在i时刻的世界坐标
        point3d_curr.push_back(pts_w_i);
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
    Vec2d delta_j((e->td - feat_j.td) * feat_j.vel);
    //Vec3d pts_j_td = feat_j.point - Vec3d(delta_j.x(),delta_j.y(),0);
    Vec3d pts_j_td = feat_j.point;

    Vec2d delta_i;
    if(isStereo)
        delta_i= (e->td - feat_i.td) * feat_i.vel_right;
    else
        delta_i= (e->td - feat_i.td) * feat_i.vel;
    //Vec3d pts_i_td = feat_i.point - Vec3d(delta_i.x(),delta_i.y(),0);
    Vec3d pts_i_td = feat_i.point;

    Vec3d pts_imu_j=e->ric[0] * (pts_j_td / depth) + e->tic[0];//k点在j时刻的IMU坐标
    Vec3d pts_w_j=e->Rs[feat_j.frame]*pts_imu_j + e->Ps[feat_j.frame];//k点在j时刻的世界坐标
    //Vec3d pts_obj_j=Q_woj.inverse()*(pts_w_j-P_woj);//k点在j时刻的物体坐标
    //Vec3d pts_w_i=Q_woi*pts_obj_j+P_woi;//k点在i时刻的世界坐标
    Vec3d pts_imu_i=e->Rs[feat_i.frame].transpose()*(pts_w_j- e->Ps[feat_i.frame]);//k点在i时刻的IMU坐标
    Vec3d pts_cam_i;
    if(isStereo)
        pts_cam_i=e->ric[1].transpose()*(pts_imu_i - e->tic[1]);
    else
        pts_cam_i=e->ric[0].transpose()*(pts_imu_i - e->tic[0]);

    Vec2d residual = (pts_cam_i / pts_cam_i.z()).head<2>() - pts_i_td.head<2>();

    return std::sqrt(residual.x() * residual.x() + residual.y() * residual.y());
}

/**
 * 外点剔除
 */
void Instance::OutlierRejection()
{
    if(!is_initial || !is_tracking)
        return;
    int num_delete=0,index=0;
    Debugv("Inst:{} landmark_num:{} box:<{}>", id, landmarks.size(), VecToStr(box));

    std::string debug_msg;
    string lm_msg;

    for(auto it=landmarks.begin(),it_next=it;it!=landmarks.end();it=it_next){
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
            Vec3d pts_w = e->Rs[imu_i] * (e->ric[0] * (lm.depth * lm.feats[0].point) + e->tic[0]) + e->Ps[imu_i];
            Vec3d pts_oi=state[imu_i].R.transpose() * ( pts_w-state[imu_i].P);
            Vec3d pts_wj=state[imu_j].R * pts_oi + state[imu_j].P;
            Vec3d pts_cj = e->ric[0].transpose() * (e->Rs[imu_j].transpose() * (pts_wj - e->Ps[imu_j]) - e->tic[0]);
            Vec2d residual = (pts_cj / pts_cj.z()).head<2>() - lm.feats[i].point.head<2>();
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
                Vec3d pts_w = e->Rs[imu_i] * (e->ric[0] * (lm.depth * lm.feats[0].point) + e->tic[0]) + e->Ps[imu_i];
                Vec3d pts_oi=state[imu_i].R.transpose() * ( pts_w-state[imu_i].P);
                Vec3d pts_wj=state[imu_j].R * pts_oi + state[imu_j].P;
                Vec3d pts_cj = e->ric[1].transpose() * (e->Rs[imu_j].transpose() * (pts_wj - e->Ps[imu_j]) - e->tic[1]);
                Vec2d residual = (pts_cj / pts_cj.z()).head<2>() - lm.feats[i].point.head<2>();
                double re = residual.norm();
                err+=re;
                err_cnt++;
                lm_msg += fmt::format("S({},{},{:.2f}) ", lm.feats[0].frame, lm.feats[i].frame, re * kFocalLength);
            }
        }
        double ave_err = err / err_cnt * kFocalLength;
        index++;
        if(ave_err > 10){
            Debugv("del lid:{} ,avg:{:.2f},d:{:.2f}> ", lm.id, ave_err, lm.depth);
            landmarks.erase(it);
            num_delete++;
        }
    }

    Debugv(lm_msg);
    Debugv("outbox:{}", debug_msg);
    Debugv("Inst:{} delete num:{}", id, num_delete);
}



/**
 * 将需要优化的参数转换到double数组中，这是因为ceres库接受double数组作为优化变量的参数形式
 */
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

    for(int i=0; i <= kWinSize; ++i){
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

/**
 * 将优化完成的参数从double数组转移到各数据结构中
 */
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




}

