/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "instance.h"
#include "estimator.h"

namespace dynamic_vins{\





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
 * 滑动窗口去除最老帧的位姿和点，并将估计的深度转移到次老帧,这个函数应该先于estimator的滑动窗口函数
 */
int Instance::SlideWindowOld()
{
    ///为了加快速度，先计算用于求坐标系变换的临时矩阵.使用的公式来自附录的第12条，temp_x表示公式的第x项
    Mat3d R_margin;
    Vec3d t_margin;
    for(auto &landmark : landmarks){
        if(landmark.feats.front().frame == 0 && landmark.feats.size() > 1 && (++landmark.feats.begin())->frame == 1){ //此时才有必要计算
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
        if(it->feats.front().frame != 0){ ///起始观测不在0帧, 则将所有的观测的frame--
            for(auto &feat : it->feats){
                feat.frame--;
            }
            continue;
        }
        else if(it->feats.size() <= 1){ ///起始观测在0帧,但是该路标点只有刚开始这个观测，故将其删掉
            landmarks.erase(it);
            debug_num++;
            continue;
        }
        else{ ///起始观测在0帧,且有多个观测, 将估计的逆深度值转移到新的开始帧
            Vec3d point_old=it->feats.front().point;
            it->feats.erase(it->feats.begin());//首先删除该观测
            ///计算新的深度
            if(it->depth > 0){
                auto pts_cam_j= point_old * it->depth;
                Vec3d pts_cam_i;
                if(it->feats.front().frame == 1){ ///满足了使用预计算的矩阵的条件，可减少计算量
                    pts_cam_i = R_margin * pts_cam_j + t_margin;
                }
                else{ ///从头计算,比较费计算量
                    int frame=it->feats.front().frame;
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

                ///设置深度
                if(pts_cam_i.z() > 0){
                    it->depth=pts_cam_i.z();
                }
                else{
                    it->depth = -1;
                }
            }

            ///将观测的frame--
            for(auto & feature_point : it->feats){
                feature_point.frame--;
            }
        }
    }

    ///将最老帧的轨迹保存到历史轨迹中
    if(is_initial && is_init_velocity){
        history_pose.push_back(state[0]);
        if(history_pose.size()>100){
            history_pose.erase(history_pose.begin());
        }
    }

    ///将最老帧的相关变量去掉
    for (int i = 0; i < kWinSize; i++){
        state[i].swap(state[i+1]);
    }
    state[kWinSize]=state[kWinSize - 1];

    for (int i = 0; i < kWinSize; i++){
        boxes3d[i].swap(boxes3d[i+1]);
    }
    boxes3d[kWinSize].reset();

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
        //删除空路标
        if(it->feats.empty()){
            landmarks.erase(it);
            debug_num++;
            continue;
        }
        //上一帧中开始观测到的,但是只被观测了一次,删除
        if(it->feats.size()==1 && it->feats.front().frame == e->frame-1){
            landmarks.erase(it);
            debug_num++;
            continue;
        }
        //删除上一帧中观测
        for(auto feat_it=it->feats.begin();feat_it!=it->feats.end();++feat_it){
            if(feat_it->frame == e->frame-1){
                it->feats.erase(feat_it);
                break;
            }
        }
        //对当前帧的观测的frame--
        for(auto& feat:it->feats){
            if(feat.frame == e->frame){
                feat.frame--;
                break;
            }
        }
    }

    boxes3d[kWinSize-1] = boxes3d[kWinSize];
    boxes3d[kWinSize].reset();

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
        int frame_j=lm.feats.front().frame;
        int frame_i=e->frame;
        Vec3d pts_cam_j = lm.feats.front().point * lm.depth;//k点在j时刻的相机坐标
        Vec3d pts_imu_j = e->ric[0] * pts_cam_j + e->tic[0];//k点在j时刻的IMU坐标
        Vec3d pts_w_j=e->Rs[frame_j] * pts_imu_j + e->Ps[frame_j];//k点在j时刻的世界坐标
        Vec3d pts_obj_j=state[frame_j].R.transpose() * (pts_w_j - state[frame_j].P);//k点在j时刻的物体坐标
        Vec3d pts_w_i=state[frame_i].R * pts_obj_j + state[frame_i].P;//k点在i时刻的世界坐标
        point3d_curr.push_back(pts_w_i);
    }
}

/**
 * 所有三角化的点在当前帧下的平均深度
 * @return
 */
double Instance::AverageDepth() const{
    double depth_sum=0;
    int cnt=0;

    for(auto &lm : landmarks){
        if(lm.depth <=0)
            continue;
        cnt++;
        int imu_i = lm.feats.front().frame;
        int imu_j = e->frame;
        Vec3d pts_w = e->Rs[imu_i] * (e->ric[0] * (lm.depth * lm.feats.front().point) + e->tic[0]) + e->Ps[imu_i];
        Vec3d pts_oi=state[imu_i].R.transpose() * ( pts_w-state[imu_i].P);
        Vec3d pts_wj=state[imu_j].R * pts_oi + state[imu_j].P;
        Vec3d pts_cj = e->ric[0].transpose() * (e->Rs[imu_j].transpose() * (pts_wj - e->Ps[imu_j]) - e->tic[0]);
        depth_sum += pts_cj.z();
    }

    if(cnt>0){
        return depth_sum/cnt;
    }
    else{
        return 0;
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
double Instance::ReprojectTwoFrameError(FeaturePoint &feat_j, FeaturePoint &feat_i, double depth, bool isStereo) const
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
    string log_text = fmt::format("OutlierRejection Inst:{} landmark_num:{} box:{}\n", id, landmarks.size(),
                                  VecToStr(box3d->dims));

    for(auto it=landmarks.begin(),it_next=it;it!=landmarks.end();it=it_next){
        it_next++;
        auto &lm=*it;
        if(lm.feats.empty() || lm.depth <= 0)
            continue;

        if(std::isfinite(lm.depth)){

        ///根据包围框去除外点
        /*int frame=lm.feats[0].frame;
        Vec3d pts_w = e->Rs[frame] * (e->ric[0] * (lm.feats[0].point * lm.depth) + e->tic[0]) + e->Ps[frame];
        Vec3d pts_oi=state[frame].R.transpose() * ( pts_w - state[frame].P);
        constexpr double factor=4.;
        bool is_in_box = (std::abs(pts_oi.x()) < factor*box.x()) && (std::abs(pts_oi.y())<factor*box.y() ) &&
                (std::abs(pts_oi.z()) < factor*box.z());
        if(!is_in_box){
            log_text += fmt::format("del outbox lid:{},d:{:.2f},pts_oi:{} \n", lm.id, lm.depth, VecToStr(pts_oi));
            it->feats.erase(it->feats.begin());//删除第一个观测
            it->depth=-1;
            if(it->feats.empty()){
                landmarks.erase(it);
            }
            num_delete++;
            continue;
        }*/

        double err = 0;
        int err_cnt = 0;
        ///单目重投影误差
        auto feat_it=lm.feats.begin();
        int imu_i = feat_it->frame;
        Vec3d &start_observe = feat_it->point;
        for(++feat_it;feat_it!=lm.feats.end();++feat_it){
            int imu_j = feat_it->frame;
            Vec3d pts_w = e->Rs[imu_i] * (e->ric[0] * (lm.depth * start_observe) + e->tic[0]) + e->Ps[imu_i];
            Vec3d pts_oi=state[imu_i].R.transpose() * ( pts_w-state[imu_i].P);
            Vec3d pts_wj=state[imu_j].R * pts_oi + state[imu_j].P;
            Vec3d pts_cj = e->ric[0].transpose() * (e->Rs[imu_j].transpose() * (pts_wj - e->Ps[imu_j]) - e->tic[0]);
            Vec2d residual = (pts_cj / pts_cj.z()).head<2>() - feat_it->point.head<2>();
            double re = residual.norm();
            err+=re;
            err_cnt++;
        }
        ///双目重投影误差
        feat_it=lm.feats.begin();
        for(++feat_it;feat_it!=lm.feats.end();++feat_it){
            if(feat_it->is_stereo){
                int imu_j = feat_it->frame;
                Vec3d pts_w = e->Rs[imu_i] * (e->ric[0] * (lm.depth * start_observe) + e->tic[0]) + e->Ps[imu_i];
                Vec3d pts_oi=state[imu_i].R.transpose() * ( pts_w-state[imu_i].P);
                Vec3d pts_wj=state[imu_j].R * pts_oi + state[imu_j].P;
                Vec3d pts_cj = e->ric[1].transpose() * (e->Rs[imu_j].transpose() * (pts_wj - e->Ps[imu_j]) - e->tic[1]);
                Vec2d residual = (pts_cj / pts_cj.z()).head<2>() - feat_it->point.head<2>();
                double re = residual.norm();
                err+=re;
                err_cnt++;
            }
        }

        double ave_err = err / err_cnt * kFocalLength;
        index++;
        if(ave_err > 30){
            log_text += fmt::format("del lid:{},d:{:.2f},avg:{:.2f} \n", lm.id, lm.depth, ave_err);
            it->feats.erase(it->feats.begin());//删除第一个观测
            it->depth=-1;
            if(it->feats.empty()){
                landmarks.erase(it);
            }
            num_delete++;
        }

        }
        else{
            log_text += fmt::format("del lid:{} not_finite,d:{:.2f} \n", lm.id, lm.depth);
            it->feats.erase(it->feats.begin());//删除第一个观测
            it->depth=-1;
            if(it->feats.empty()){
                landmarks.erase(it);
            }
            num_delete++;
        }
    }

    log_text +=fmt::format("Inst:{} delete num:{}", id, num_delete);
    Debugv(log_text);
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
    para_box[0][0]=box3d->dims.x();
    para_box[0][1]=box3d->dims.y();
    para_box[0][2]=box3d->dims.z();

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
    for(auto &lm : landmarks){
        if(lm.depth<0.2)
            continue;
        index++;
        para_inv_depth[index][0]= 1.0 / lm.depth;
    }
}

/**
 * 将优化完成的参数从double数组转移到各数据结构中
 */
void Instance::GetOptimizationParameters()
{
    last_vel=vel;

    vel.v.x()=para_speed[0][0];
    vel.v.y()=para_speed[0][1];
    vel.v.z()=para_speed[0][2];
    vel.a.x()=para_speed[0][3];
    vel.a.y()=para_speed[0][4];
    vel.a.z()=para_speed[0][5];
    box3d->dims.x()=para_box[0][0];
    box3d->dims.y()=para_box[0][1];
    box3d->dims.z()=para_box[0][2];

    for(int i=0;i<=kWinSize;++i){
        ///限制其单帧位移不能太大
        if((state[i].P - Vec3d(para_state[i][0],para_state[i][1],para_state[i][2])).norm() > 10){
            continue;
        }


        state[i].P.x()=para_state[i][0];
        state[i].P.y()=para_state[i][1];
        state[i].P.z()=para_state[i][2];
        Eigen::Quaterniond q(para_state[i][6],para_state[i][3],para_state[i][4],para_state[i][5]);
        q.normalize();
        state[i].R=q.toRotationMatrix();
    }
    /*state[0].P.x()=para_state[0][0];
    state[0].P.y()=para_state[0][1];
    state[0].P.z()=para_state[0][2];
    Eigen::Quaterniond q(para_state[0][6], para_state[0][3], para_state[0][4], para_state[0][5]);
    q.normalize();
    state[0].R=q.toRotationMatrix();
    SetWindowPose();*/

    int index=-1;
    for(auto &lm : landmarks){
        if(lm.depth<0.2)
            continue;
        index++;
        lm.depth= 1.0 / para_inv_depth[index][0];
    }
}





}

