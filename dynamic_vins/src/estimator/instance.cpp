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

namespace dynamic_vins{ \


/**
 * 根据速度推导滑动窗口中各个物体的位姿
 */
void Instance::SetWindowPose()
{
    for(int i=1; i <= kWinSize; i++){
        state[i].time=body.headers[i];
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
    for(auto &lm : landmarks){
        if(lm.bad)
            continue;
        if(lm.frame() == 0 && lm.size() > 1 && lm[1]->frame == 1){ //此时才有必要计算
            auto &R_bc = body.ric[0];
            auto R_cb = R_bc.transpose();
            auto &P_bc = body.tic[0];
            auto temp_5 = -R_cb * P_bc;
            auto temp_RcbRbiw = R_cb * body.Rs[1].transpose();
            auto temp_4 = temp_RcbRbiw * (state[1].P - body.Ps[1]);
            auto temp_RcbRbiwRwoiRojw=temp_RcbRbiw * state[1].R * state[0].R.transpose();
            auto temp_3 = temp_RcbRbiwRwoiRojw * (body.Ps[0] - state[0].P);
            auto temp_2 = temp_RcbRbiwRwoiRojw * body.Rs[0] * P_bc;
            auto temp_1 = temp_RcbRbiwRwoiRojw * body.Rs[0] * R_bc;

            R_margin=temp_1;
            t_margin=temp_2+temp_3+temp_4+temp_5;
            break;
        }
    }

    //Debugv("Instance::SlideWindowOld() inst:{} prepare",id);

    int debug_num=0;

    ///改变所有路标点的start_frame ,并保留最老帧中的特征点，寻找一个新的归宿
    for(auto &lm:landmarks){
        if(lm.bad){
            continue;
        }
        else if(lm.frame() != 0){ ///起始观测不在0帧, 则将所有的观测的frame--
            for(auto &feat : lm.feats){
                feat->frame--;
            }
            continue;
        }
        else if(lm.size() <= 1){ ///起始观测在0帧,但是该路标点只有刚开始这个观测，故将其删掉
            lm.bad=true;
            debug_num++;
            continue;
        }
        else{ ///起始观测在0帧,且有多个观测, 将估计的逆深度值转移到新的开始帧
            Vec3d point_old=lm.front()->point;
            lm.EraseBegin();
            ///计算新的深度
            if(lm.depth > 0){
                auto pts_cam_j= point_old * lm.depth;
                Vec3d pts_cam_i;
                if(lm.frame() == 1){ ///满足了使用预计算的矩阵的条件，可减少计算量
                    pts_cam_i = R_margin * pts_cam_j + t_margin;
                }
                else{ ///从头计算,比较费计算量
                    int frame=lm.frame();
                    auto &R_bc = body.ric[0];
                    auto R_cb = R_bc.transpose();
                    auto &P_bc = body.tic[0];
                    auto temp_5 = -R_cb * P_bc;
                    auto temp_RcbRbiw = R_cb * body.Rs[frame].transpose();
                    auto temp_4 = temp_RcbRbiw * (state[frame].P - body.Ps[frame]);
                    auto temp_RcbRbiwRwoiRojw=temp_RcbRbiw * state[frame].R * state[0].R.transpose();
                    auto temp_3 = temp_RcbRbiwRwoiRojw * (body.Ps[0] - state[0].P);
                    auto temp_2 = temp_RcbRbiwRwoiRojw * body.Rs[0] * P_bc;
                    auto temp_1 = temp_RcbRbiwRwoiRojw * body.Rs[0] * R_bc;
                    R_margin=temp_1;
                    t_margin=temp_2+temp_3+temp_4+temp_5;
                    pts_cam_i = R_margin * pts_cam_j + t_margin;
                }

                ///设置深度
                if(pts_cam_i.z() > 0){
                    lm.depth=pts_cam_i.z();
                }
                else{
                    lm.depth = -1;
                }
            }

            ///将观测的frame--
            for(auto & feat : lm.feats){
                feat->frame--;
            }
        }
    }

    ///将最老帧的相关变量去掉
    for (int i = 0; i < kWinSize; i++){
        state[i].swap(state[i+1]);
        boxes3d[i].swap(boxes3d[i+1]);
        points_extra[i] = points_extra[i+1];
        points_extra_pcl[i] = points_extra_pcl[i+1];
    }

    state[kWinSize]=state[kWinSize - 1];
    boxes3d[kWinSize].reset();
    points_extra[kWinSize].clear();
    points_extra_pcl[kWinSize].reset();

    return debug_num;
}


/**
 * 滑动窗口去除次新帧的位姿和观测，并将最新帧移到次新帧
 */
int Instance::SlideWindowNew()
{
    int debug_num=0;

    for (auto &lm:landmarks){
        if(lm.bad) continue;
        //删除空路标
        if(lm.feats.empty()){
            lm.bad=true;
            debug_num++;
            continue;
        }
        //上一帧中开始观测到的,但是只被观测了一次,删除
        if(lm.size()==1 && lm.frame() == body.frame-1){
            lm.bad=true;
            debug_num++;
            continue;
        }
        //删除上一帧中观测
        for(auto feat_it=lm.feats.begin();feat_it!=lm.feats.end();++feat_it){
            if((*feat_it)->frame == body.frame-1){
                lm.erase(feat_it);
                break;
            }
        }
        //对当前帧的观测的frame--
        for(auto& feat:lm.feats){
            if(feat->frame == body.frame){
                feat->frame--;
                break;
            }
        }
    }

    boxes3d[kWinSize-1] = boxes3d[kWinSize];
    boxes3d[kWinSize].reset();

    points_extra[kWinSize-1] = points_extra[kWinSize];
    points_extra[kWinSize].clear();
    points_extra_pcl[kWinSize-1] = points_extra_pcl[kWinSize];
    points_extra_pcl[kWinSize].reset();

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
        if(lm.depth <= 0 || lm.bad)
            continue;
        int frame_j=lm.frame();
        int frame_i=body.frame;
        point3d_curr.emplace_back( ObjectToWorld( CamToObject(lm.front()->point * lm.depth,frame_j),frame_i));
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
        if(lm.depth <=0 || lm.bad)
            continue;
        cnt++;
        int imu_i = lm.frame();
        int imu_j = body.frame;
        Vec3d pts_cj = ObjectToCam(CamToObject(lm.depth * lm.feats.front()->point,imu_i),imu_j);
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
 * 外点剔除
 */
void Instance::OutlierRejection()
{
    if(!is_initial || !is_tracking)
        return;
    int num_delete=0,index=0;
    string log_text = fmt::format("OutlierRejection Inst:{} landmark_num:{} box:{}\n", id, landmarks.size(),
                                  VecToStr(box3d->dims));

    for(auto &lm:landmarks){
        if(lm.bad)
            continue;

        if(std::isfinite(lm.depth)){
            ///根据包围框去除外点
            if(!IsInBoxPc(lm.front()->point * lm.depth,lm.frame())){
               // log_text += fmt::format("del outbox lid:{},d:{:.2f},pts_oi:{} \n", lm.id, lm.depth, VecToStr(pts_oi));
                lm.EraseBegin();
                lm.depth=-1;
                if(lm.feats.empty()){
                    lm.bad=true;
                }
                num_delete++;
                continue;
            }

            double err = 0;
            int err_cnt = 0;
            ///单目重投影误差
            auto feat_it=lm.feats.begin();
            int imu_i = (*feat_it)->frame;
            Vec3d &start_observe = (*feat_it)->point;
            for(++feat_it;feat_it!=lm.feats.end();++feat_it){
                int imu_j = (*feat_it)->frame;
                Vec3d pts_cj = ObjectToCam(CamToObject(lm.depth * start_observe,imu_i),imu_j);
                Vec2d residual = (pts_cj / pts_cj.z()).head<2>() - (*feat_it)->point.head<2>();
                double re = residual.norm();
                err+=re;
                err_cnt++;
            }
            ///双目重投影误差
            feat_it=lm.feats.begin();
            for(++feat_it;feat_it!=lm.feats.end();++feat_it){
                if((*feat_it)->is_stereo){
                    int imu_j = (*feat_it)->frame;
                    Vec3d pts_cj = ObjectToCam(CamToObject(lm.depth * start_observe,imu_i,0),imu_j,1);
                    Vec2d residual = (pts_cj / pts_cj.z()).head<2>() - (*feat_it)->point.head<2>();
                    double re = residual.norm();
                    err+=re;
                    err_cnt++;
                }
            }

            double ave_err = err / err_cnt * kFocalLength;
            index++;
            if(ave_err > 30){
                //log_text += fmt::format("del lid:{},d:{:.2f},avg:{:.2f} \n", lm.id, lm.depth, ave_err);
                lm.EraseBegin();
                lm.depth=-1;
                if(lm.feats.empty()){
                    lm.bad=true;
                }
                num_delete++;
            }

        }
        else{
            //log_text += fmt::format("del lid:{} not_finite,d:{:.2f} \n", lm.id, lm.depth);
            lm.EraseBegin();
            lm.depth=-1;
            if(lm.feats.empty()){
                lm.bad=true;
            }
            num_delete++;
        }
    }

    log_text +=fmt::format("Inst:{} delete num:{}", id, num_delete);
    Debugv(log_text);
}


/**
 * 判断之前三角化得到的点是否在包围框内
 * @return
 */
int Instance::OutlierRejectionByBox3d(){
    int del_num=0;

    double box_norm = box3d->dims.norm();

    for(auto &lm:landmarks){
        if(lm.bad)
            continue;

        ///剔除额外点
        if(lm.is_extra()){
            if(lm.depth>0){
                Vec3d point_obj = WorldToObject(lm.front()->p_w,lm.frame());
                if( (std::abs(point_obj.x()) >= 3*box3d->dims.x() ||
                std::abs(point_obj.y()) > 2*box3d->dims.y() ||
                std::abs(point_obj.z()) > 2*box3d->dims.z() ) ||
                (point_obj.norm() > 3*box_norm)){
                    lm.bad = true;
                }
            }
            continue;
        }

        ///根据包围框剔除双目3D点
        for(auto &feat: lm.feats){
            if(feat->is_triangulated &&feat->frame != body.frame){ //这里不剔除当前帧
                bool is_outbox= false;
                Vec3d point_obj = WorldToObject(feat->p_w,feat->frame);

                if( (std::abs(point_obj.x()) >= 3*box3d->dims.x() ||
                std::abs(point_obj.y()) > 3*box3d->dims.y() ||
                std::abs(point_obj.z()) > 3*box3d->dims.z() ) ||
                (point_obj.norm() > 3*box_norm)){
                    is_outbox=true;
                }

                if(is_outbox && boxes3d[feat->frame]){
                    Vec3d pts_cam = body.WorldToCam(feat->p_w,body.frame);
                    if((pts_cam - boxes3d[feat->frame]->center_pt).norm() >
                    3 * boxes3d[feat->frame]->dims.norm()){
                        is_outbox = false;
                    }
                }

                if(is_outbox){
                    feat->is_triangulated=false;
                    feat->is_stereo = false;
                    del_num++;
                }
            }
        }

        ///根据包围框剔除单目三角化得到的点
        if(lm.depth>0){
            auto feat = lm.front();
            bool is_outbox= false;
            Vec3d point_obj = CamToObject(feat->point * lm.depth,body.frame);

            if( (std::abs(point_obj.x()) >= 3*box3d->dims.x() ||
            std::abs(point_obj.y()) > 3*box3d->dims.y() ||
            std::abs(point_obj.z()) > 3*box3d->dims.z() ) ||
            (point_obj.norm() > 3*box_norm)){
                is_outbox=true;
            }

            if(is_outbox){
                lm.EraseBegin();
                lm.depth = -1;
                del_num++;
            }
        }
    }
    return del_num;

}



/**
 * 删除被标记为bad的路标点
 * @return 删除的路标点的数量
 */
int Instance::DeleteBadLandmarks(){
    int cnt=0;
    for(auto it=landmarks.begin(),it_next=it;it!=landmarks.end();it=it_next){
        it_next++;
        if(it->bad){
            landmarks.erase(it);
            cnt++;
        }
    }
    return cnt;
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
        if(lm.bad
        || lm.depth < 0.2
        || lm.is_extra())
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
        Vec3d step = Vec3d(para_state[i][0],para_state[i][1],para_state[i][2]) - state[i].P;

        ///限制其单帧位移不能太大
        if(step.norm() > 10){
            Vec3d step_constraint = step.normalized() * 10.;
            Warnv("To much step in inst:{} frame:{}, raw step:{}, used step:{}",
                  id,i,VecToStr(step), VecToStr(step_constraint));
            step = step_constraint;
        }

        state[i].P = state[i].P + step;

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
        if(lm.bad
        || lm.depth < 0.2
        || lm.is_extra())
            continue;
        index++;
        lm.depth= 1.0 / para_inv_depth[index][0];
    }
}

/**
 * 删除某一帧之前的所有路标
 * @param critical_frame 临界帧
 */
void Instance::DeleteOutdatedLandmarks(int critical_frame){
    for(auto &lm:landmarks){
        if(lm.bad || lm.frame()==critical_frame)
            continue;

        if(lm.size() == 1){ //只有一个观测,直接删除
            lm.bad=true;
            continue;
        }
        else{
            for(auto it=lm.feats.begin(),it_next=it; it != lm.feats.end(); it=it_next){
                it_next++;
                if((*it)->frame < critical_frame) {
                    lm.erase(it); //删掉掉前面的观测
                }
            }
            if(lm.feats.empty()){
                lm.bad=true;
            }
            if(lm.depth > 0){
                lm.depth=-1.0;//需要重新进行三角化
            }
        }

    }
}


}

