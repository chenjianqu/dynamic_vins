/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "estimator_insts.h"

#include <algorithm>
#include <filesystem>

#include "estimator.h"
 #include "utils/def.h"
#include "vio_parameters.h"

#include "estimator/factor/pose_local_parameterization.h"
#include "estimator/factor/project_instance_factor.h"
#include "estimator/factor/speed_factor.h"
#include "estimator/factor/box_factor.h"
#include "utils/dataset/kitti_utils.h"
#include "utils/io/io_parameters.h"
#include "utils/io/io_utils.h"

namespace dynamic_vins{\


void InstanceManager::set_estimator(Estimator* estimator){
    e=estimator;
}



/**
 * 添加动态物体的特征点,创建新的Instance, 更新box3d. 三角化输入的双目点,以得到场景流
 * @param frame_id
 * @param instance_id
 * @param input_insts
 */
void InstanceManager::PushBack(unsigned int frame_id, std::map<unsigned int,FeatureInstance> &input_insts)
{
    frame = frame_id;

    if(e->solver_flag == SolverFlag::kInitial || input_insts.empty()){
        return;
    }
    Debugv("PushBack input_inst_size:{},feat_size:{}",input_insts.size(),
           std::accumulate(input_insts.begin(), input_insts.end(), 0,
                           [](int sum, auto &p) { return sum + p.second.size(); }));
    string log_text;

    for(auto &inst_pair : instances){
        inst_pair.second.lost_number++;
        inst_pair.second.is_curr_visible=false;
    }

    for(auto &[instance_id , inst_feat] : input_insts){
        Debugv("PushBack 跟踪实例id:{} 特征点数:{}",instance_id,inst_feat.size());

        ///创建物体
        auto inst_iter = instances.find(instance_id);
        if(inst_iter == instances.end()){
            Instance new_inst(frame, instance_id, e);
            auto [it,is_insert] = instances.insert({instance_id, new_inst});
            it->second.is_initial=false;
            it->second.color = inst_feat.color;
            it->second.is_curr_visible=true;
            it->second.box2d = inst_feat.box2d;
            Debugv("PushBack | box2d:{}", inst_feat.box2d->class_name);
            it->second.box3d = std::make_shared<Box3D>();
            if(inst_feat.box3d){
                it->second.boxes3d[frame] = inst_feat.box3d;
                it->second.box3d->class_id = inst_feat.box3d->class_id;
                it->second.box3d->class_name = inst_feat.box3d->class_name;
                it->second.box3d->score = inst_feat.box3d->score;
                Debugv("PushBack input box3d center:{},yaw:{} score:{}",
                       VecToStr(inst_feat.box3d->center_pt), inst_feat.box3d->yaw, inst_feat.box3d->score);
            }

            tracking_number_++;

            for(auto &[feat_id,feat_vector] : inst_feat){
                LandmarkPoint lm(feat_id);//创建Landmark
                lm.feats.emplace_back(feat_vector, frame, e->td);//添加第一个观测
                it->second.landmarks.push_back(lm);
            }
            Debugv("PushBack | 创建实例:{}", instance_id);

        }
        else{ ///将特征添加到物体中
            auto &landmarks = inst_iter->second.landmarks;
            inst_iter->second.box2d = inst_feat.box2d;
            if(inst_feat.box3d){
                inst_iter->second.boxes3d[frame] = inst_feat.box3d;
                inst_iter->second.box3d->class_id = inst_feat.box3d->class_id;
                inst_iter->second.box3d->class_name = inst_feat.box3d->class_name;
                inst_iter->second.box3d->score = inst_feat.box3d->score;

                Debugv("PushBack input box3d center:{},yaw:{} score:{}",
                       VecToStr(inst_feat.box3d->center_pt), inst_feat.box3d->yaw, inst_feat.box3d->score);
            }

            inst_iter->second.lost_number=0;
            inst_iter->second.is_curr_visible=true;

            if(!inst_iter->second.is_tracking){
                inst_iter->second.is_tracking=true;
                Debugv("重新发现目标 id:{}",inst_iter->second.id);
                tracking_number_++;
            }

            for(auto &[feat_id,feat_vector] : inst_feat){
                //若路标不存在，则创建路标
                auto it = std::find_if(landmarks.begin(),landmarks.end(),
                                       [id=feat_id](const LandmarkPoint &it){ return it.id == id;});
                if (it ==landmarks.end()){
                    landmarks.emplace_back(feat_id);//创建Landmarks
                    it = std::prev(landmarks.end());
                }
                it->feats.emplace_back(feat_vector, frame, e->td);//添加观测
            }

        }
    }

    ///根据观测次数,对特征点进行排序
    for(auto &[inst_id,inst] : instances){
        if(inst.lost_number==0){
            inst.landmarks.sort([](const LandmarkPoint &lp1,const LandmarkPoint &lp2){
                return lp1.feats.size() > lp2.feats.size();});
        }
    }

}



/**
 * 三角化动态特征点,并限制三角化的点的数量在50以内
 * @param frame_cnt
 */
void InstanceManager::Triangulate()
{
    if(tracking_number_ < 1)
        return;
    auto getCamPose=[this](int index,int cam_id){
        assert(cam_id == 0 || cam_id == 1);
        Mat34d pose;
        Vec3d t0 = e->Ps[index] + e->Rs[index] * e->tic[cam_id];
        Mat3d R0 = e->Rs[index] * e->ric[cam_id];
        pose.leftCols<3>() = R0.transpose();
        pose.rightCols<1>() = -R0.transpose() * t0;
        return pose;
    };

    string log_text = "InstanceManager::Triangulate\n";

    int num_triangle=0,num_failed=0,num_delete_landmark=0,num_mono=0;
    for(auto &[key,inst] : instances){
        if(!inst.is_tracking )
            continue;

        string log_inst_text;

        ///对双目点进行三角化
        int num_stereo=0;
        for(auto &lm : inst.landmarks){

            for(auto it=lm.feats.begin(),it_next=it;it!=lm.feats.end();it=it_next){
                it_next++;
                auto &feat = *it;

                if(feat.is_stereo && !feat.is_triangulated){
                    auto leftPose = getCamPose(feat.frame,0);
                    auto rightPose= getCamPose(feat.frame,1);
                    Vec2d point0 = feat.point.head(2);
                    Vec2d point1 = feat.point_right.head(2);
                    Eigen::Vector3d point3d_w;
                    TriangulatePoint(leftPose, rightPose, point0, point1, point3d_w);
                    Eigen::Vector3d localPoint = leftPose.leftCols<3>() * point3d_w + leftPose.rightCols<1>();//变换到相机坐标系下
                    double depth = localPoint.z();

                    if (depth > kDynamicDepthMin && depth<kDynamicDepthMax){//如果深度有效
                        feat.is_triangulated = true;
                        feat.p_w = point3d_w;
                        num_stereo++;
                        if(lm.depth <=0 ){//这里的情况更加复杂一些,只有当该路标点未初始化时,才会进行初始化
                            ///清空该点之前的观测
                            lm.feats.erase(lm.feats.begin(),it);
                            lm.depth = depth;
                        }
                    }
                    else{
                        feat.is_stereo=false;
                    }
                }
            }

        }

        log_inst_text += fmt::format("add num_stereo:{} \n",num_stereo);


        ///下面初始化每个路标的深度值

        int inst_add_num=0;
        //double avg_depth= inst.AverageDepth();//在当前帧平均深度

        for(auto it=inst.landmarks.begin(),it_next=it;it!=inst.landmarks.end();it=it_next){
            it_next++;
            auto &lm=*it;
            if(lm.depth > 0 || lm.feats.empty())
                continue;
            int imu_i = lm.feats.front().frame;
            Eigen::Vector3d point3d_w;
            ///双目三角化,由于其它函数可能会将lm.depth 设置为-1,因此下面的分支可能会执行
            if(cfg::is_stereo && lm.feats.front().is_stereo && lm.feats.front().is_triangulated){
                point3d_w = lm.feats.front().p_w;
                log_inst_text += fmt::format("lid:{} S ,point3d_w:{}\n",lm.id, VecToStr(point3d_w));
            }
            ///单目三角化，以开始帧和开始帧的后一帧的观测作为三角化的点
            else if(lm.feats.size() >= 2){
                Mat34d leftPose = getCamPose(imu_i,0);
                auto feat_j = (++lm.feats.begin());
                int imu_j=feat_j->frame;
                Mat34d rightPose = getCamPose(imu_j,0);
                Vec2d point0 = lm.feats.front().point.head(2);
                Vec2d point1 = feat_j->point.head(2);
                if(inst.is_initial){
                    TriangulateDynamicPoint(leftPose, rightPose, point0, point1,
                                            inst.state[imu_i].R, inst.state[imu_i].P,
                                            inst.state[imu_j].R, inst.state[imu_j].P, point3d_w);
                    log_inst_text += fmt::format("lid:{} M init ,point3d_w:{}\n",lm.id, VecToStr(point3d_w));
                }
                else{
                    TriangulatePoint(leftPose, rightPose, point0, point1, point3d_w);
                    log_inst_text += fmt::format("lid:{} M not_init ,point3d_w:{}\n",lm.id, VecToStr(point3d_w));
                }
            }
            else{
                continue;
            }


            //将点投影当前帧
            Vec3d pts_cj = e->ric[0].transpose() * (e->Rs[frame].transpose() * (point3d_w - e->Ps[frame]) - e->tic[0]);
            double depth = pts_cj.z();

            if (depth > kDynamicDepthMin && depth<kDynamicDepthMax){ //判断深度值是否符合
                if(!inst.is_initial || inst.triangle_num < 5){ //若未初始化或路标点太少，则加入
                    lm.depth = depth;
                    inst_add_num++;
                    log_inst_text+=fmt::format("lid:{} NotInit d:{:.2f} p:{}\n", lm.id, depth, VecToStr(point3d_w));
                }
                else{ //判断是否在包围框附近
                    Eigen::Vector3d pts_obj_j=inst.state[imu_i].R.transpose() * (point3d_w - inst.state[imu_i].P);
                    if(pts_obj_j.norm() < inst.box3d->dims.norm()*3){
                        lm.depth = depth;
                        inst_add_num++;
                        log_inst_text+=fmt::format("lid:{} d:{:.2f} p:{}\n", lm.id, depth, VecToStr(point3d_w));
                    }
                    else{
                        log_inst_text+=fmt::format("lid:{} outbox d:{:.2f} p:{}\n", lm.id, depth, VecToStr(point3d_w));
                    }
                }
            }
            else{///深度值太大或太小
                num_failed++;
                if(lm.feats.size() == 1){//删除该路标
                    inst.landmarks.erase(it);
                    num_delete_landmark++;
                }
                else{//删除该观测
                    lm.feats.erase(lm.feats.begin());
                }
            }
        }

        log_text += fmt::format("inst:{} landmarks.size:{} depth.size:{} new_add:{}\n",
                                inst.id, inst.landmarks.size(), inst.triangle_num, inst_add_num);
        log_text += log_inst_text;
        num_triangle += inst_add_num;
    }

    if(num_triangle>0 || num_delete_landmark>0){
        log_text += fmt::format("InstanceManager::Triangulate: 增加:{}=(M:{},S:{}) 失败:{} 删除:{}",
               num_triangle, num_mono, num_triangle - num_mono,
               num_failed, num_delete_landmark);
    }
    Debugv(log_text);


}


void InstanceManager::ManageTriangulatePoint()
{
    string log_text="InstanceManager::ManageTriangulatePoint\n";

    for(auto &[key,inst] : instances){
        if(!inst.is_tracking)
            continue;
        ///路标太少,不进行删除
        if(inst.landmarks.size()<10){
            continue;
        }

        double box_norm = inst.box3d->dims.norm();

        std::array<int,11> statistics{0};//统计
        for(auto &lm:inst.landmarks){
            for(auto &feat:lm.feats){
                statistics[feat.frame]++;
            }
        }
        //若上一帧中点太少,可能上一帧未不存在特征点,则不进行剔除操作
        if(statistics[kWinSize-1]<=2){
            continue;
        }
        log_text += fmt::format("inst:{} \n",inst.id);

        for(auto &lm:inst.landmarks){

            ///判断之前三角化得到的点是否在包围框内
            if(inst.is_initial){
                string s;

                ///根据包围框剔除双目3D点

                for(auto &feat: lm.feats){
                    if(feat.is_triangulated &&feat.frame != e->frame){ //这里不剔除当前帧
                        bool is_outbox= false;
                        Vec3d point_obj = inst.state[feat.frame].R.transpose() * (feat.p_w -inst.state[feat.frame].P);

                        if( (std::abs(point_obj.x()) >= 3*inst.box3d->dims.x() ||
                        std::abs(point_obj.y()) > 3*inst.box3d->dims.y() ||
                        std::abs(point_obj.z()) > 3*inst.box3d->dims.z() ) ||
                        (point_obj.norm() > 3*box_norm)){
                            is_outbox=true;
                        }

                        if(is_outbox && inst.boxes3d[feat.frame]){
                            Vec3d pts_cam = e->ric[0].transpose() *
                                    (e->Rs[feat.frame].transpose() * (feat.p_w - e->Ps[feat.frame]) - e->tic[0]);

                            if((pts_cam - inst.boxes3d[feat.frame]->center_pt).norm() >
                            3 * inst.boxes3d[feat.frame]->dims.norm()){
                                is_outbox = false;
                            }
                        }

                        if(is_outbox){
                            feat.is_triangulated=false;
                            feat.is_stereo = false;
                            s += fmt::format("del S {}-{} ",feat.frame, VecToStr(feat.p_w));
                        }
                    }
                }
                if(!s.empty()){
                    log_text += fmt::format("del lid:{} {}\n",lm.id,s);
                }


                ///根据包围框剔除单目三角化得到的点
                if(lm.depth>0){
                    auto &feat = lm.feats.front();
                    bool is_outbox= false;
                    Vec3d point_w = e->Rs[feat.frame] * ( e->ric[0] * (feat.point * lm.depth) + e->tic[0])
                            + e->Ps[feat.frame];
                    Vec3d point_obj = inst.state[feat.frame].R.transpose() * (point_w -inst.state[feat.frame].P);

                    if( (std::abs(point_obj.x()) >= 3*inst.box3d->dims.x() ||
                    std::abs(point_obj.y()) > 3*inst.box3d->dims.y() ||
                    std::abs(point_obj.z()) > 3*inst.box3d->dims.z() ) ||
                    (point_obj.norm() > 3*box_norm)){
                        is_outbox=true;
                    }

                    if(is_outbox){
                        lm.feats.erase(lm.feats.begin());
                        lm.depth = -1;
                        s += fmt::format("del T {}-d:{} ",feat.frame, lm.depth);
                    }
                }

            }
        }

    }

    Debugv(log_text);


    for(auto &[key,inst] : instances){
        if(!inst.is_tracking)
            continue;

        inst.triangle_num=0;
        for(auto &lm:inst.landmarks){
            if(lm.depth>0){
                inst.triangle_num++;
            }
        }

        if(inst.triangle_num < 50)
            continue;

        for(auto it=inst.landmarks.begin(),it_next=it;it!=inst.landmarks.end();it=it_next){
            it_next++;
            if(it->feats.size()<=1 && it->feats.front().frame < frame && it->depth<=0){ //只有一个观测,且该观测不在当前帧
                inst.landmarks.erase(it);
            }
            if(inst.triangle_num <10){
                break;
            }
        }

    }

}


/**
 * 根据速度和上一帧的位姿预测动态物体在当前帧的位姿
 */
void InstanceManager::PropagatePose()
{
    if(tracking_number_ < 1)
        return;

    string log_text = "InstanceManager::PropagatePose \n";

    int last_frame= frame - 1;
    double time_ij= e->headers[frame] - e->headers[last_frame];

    InstExec([&](int key,Instance& inst){

        //inst.vel = inst.point_vel;

        inst.state[frame].time = e->headers[frame];

        /*inst.state[frame].R = inst.state[frame].R;
        inst.state[frame].P = inst.state[frame].P;*/

        /*
        if(!inst.is_init_velocity || inst.is_static){
            inst.state[frame].R = inst.state[last_frame].R;
            inst.state[frame].P = inst.state[last_frame].P;
            Debugv("InstanceManager::PropagatePose id:{} same",inst.id);
        }
        */

        /*Mat3d Roioj=Sophus::SO3d::exp(inst.point_vel.a*time_ij).matrix();
        Vec3d Poioj=inst.point_vel.v*time_ij;
        inst.state[frame].R = Roioj * inst.state[last_frame].R;
        inst.state[frame].P = Roioj* inst.state[last_frame].P + Poioj;
        Debugv("InstanceManager::PropagatePose id:{} Poioj:{}",inst.id, VecToStr(Poioj));*/

        if(inst.is_static){
            inst.state[frame].R = inst.state[last_frame].R;
            inst.state[frame].P = inst.state[last_frame].P;
        }
        ///距离太远,计算得到的速度不准确
        else if((inst.state[frame].P - e->Ps[frame]).norm() > 80){
            inst.state[frame].R = inst.state[last_frame].R;
            inst.state[frame].P = inst.state[last_frame].P;
        }
        else if(!inst.is_init_velocity){
            Mat3d Roioj=Sophus::SO3d::exp(inst.point_vel.a*time_ij).matrix();
            Vec3d Poioj=inst.point_vel.v*time_ij;
            inst.state[frame].R = Roioj * inst.state[last_frame].R;
            inst.state[frame].P = Roioj* inst.state[last_frame].P + Poioj;
            log_text += fmt::format("InstanceManager::PropagatePose id:{} Poioj:{} \n",inst.id, VecToStr(Poioj));
        }
        else{
            /*Mat3d Roioj=Sophus::SO3d::exp(inst.vel.a*time_ij).matrix();
            Vec3d Poioj=inst.vel.v*time_ij;
            inst.state[frame].R = Roioj * inst.state[last_frame].R;
            inst.state[frame].P = Roioj* inst.state[last_frame].P + Poioj;*/


            Mat3d Roioj=Sophus::SO3d::exp(inst.point_vel.a*time_ij).matrix();
            Vec3d Poioj=inst.point_vel.v*time_ij;
            inst.state[frame].R = Roioj * inst.state[last_frame].R;
            inst.state[frame].P = Roioj* inst.state[last_frame].P + Poioj;

            log_text += fmt::format("InstanceManager::PropagatePose id:{} Poioj:{} \n",inst.id, VecToStr(Poioj));
        }

        inst.vel = inst.point_vel;

    },true);

    Debugv(log_text);
}


void InstanceManager::SlideWindow()
{
    if(frame != kWinSize)
        return;

    if(e->margin_flag == MarginFlag::kMarginOld)
        Debugv("InstanceManager::SlideWindow margin_flag = kMarginOld");
    else
        Debugv("InstanceManager::SlideWindow margin_flag = kMarginSecondNew | ");

    string log_text="InstanceManager::SlideWindow\n";

    for(auto &[key,inst] : instances){
        if(!inst.is_tracking)
            continue;

        int debug_num=0;
        if (e->margin_flag == MarginFlag::kMarginOld)/// 边缘化最老的帧
            debug_num= inst.SlideWindowOld();
        else/// 去掉次新帧
            debug_num= inst.SlideWindowNew();

        inst.triangle_num=0;
        for(auto &lm:inst.landmarks){
            if(lm.depth>0){
                inst.triangle_num++;
            }
        }

        if(debug_num>0){
            log_text+=fmt::format("Inst:{},del:{} \n", inst.id, debug_num);
        }

        ///当物体没有正在跟踪的特征点时，将其设置为不在跟踪状态
        if(inst.landmarks.empty()){
            inst.ClearState();
            tracking_number_--;
            log_text+=fmt::format("inst_id:{} ClearState\n", inst.id);
        }
        else if(inst.is_tracking && inst.triangle_num==0){
            inst.is_initial=false;
            log_text+=fmt::format("inst_id:{} set is_initial=false\n", inst.id);
        }

    }

    Debugv(log_text);
}


void InstanceManager::InitialInstanceVelocity(){
    for(auto &[inst_id,inst] : instances){
        if(!inst.is_initial || inst.is_init_velocity){
            continue;
        }

        Vec3d avg_t = Vec3d::Zero();
        int cnt_t = 0;
        for(auto &lm: inst.landmarks){
            std::list<FeaturePoint>::iterator first_point;
            bool found_first=false;
            for(auto it=lm.feats.begin();it!=lm.feats.end();++it){
                if(it->is_triangulated){
                    if(!found_first){
                        first_point=it;
                        found_first=true;
                    }
                    else{
                        double time_ij = e->headers[it->frame] - e->headers[first_point->frame];
                        avg_t += (it->p_w - first_point->p_w) / time_ij;
                        cnt_t ++;
                        break;
                    }
                }
            }
        }
        if(cnt_t>10){
            Velocity v;
            v.v=avg_t/cnt_t;
            inst.history_vel.push_back(v);

            inst.vel.SetZero();
            for(auto &v:inst.history_vel){
                inst.vel.v += v.v;
            }
            inst.vel.v /= (double)inst.history_vel.size();

            inst.is_init_velocity = true;

            ///根据速度,和当前帧位姿,重新设置前面的物体位姿
            for(int i=0;i<frame;++i){
                double time_ij = e->headers[frame] - e->headers[i];
                Vec3d P_oioj = time_ij*inst.vel.v;
                Mat3d R_oioj = Sophus::SO3d::exp(time_ij * inst.vel.a).matrix();
                inst.state[i].R = R_oioj.transpose() * inst.state[frame].R;
                inst.state[i].P = R_oioj.transpose() * (inst.state[frame].P - P_oioj);
            }

            Debugv("InstanceManager::InitialInstanceVelocity modify inst:{}",inst.id);
        }

    }
}


/**
* 进行物体的位姿初始化
*/
void InstanceManager::InitialInstance(std::map<unsigned int,FeatureInstance> &input_insts){
    auto cam_to_world = [this](const Vec3d &p){
        return e->Rs[frame] * (e->ric[0] * p + e->tic[0]) + e->Ps[frame];
    };
    auto world_to_cam = [this](const Vec3d &p_w){
        return e->ric[0].transpose() * (e->Rs[frame].transpose() * (p_w - e->Ps[frame]) - e->tic[0]);
    };

    string log_text="InstanceManager::InitialInstance \n";

    for(auto &[inst_id,inst] : instances){
        if(inst.is_initial){
            inst.age++;
        }

        if(inst.is_initial || !inst.is_tracking)
            continue;

        ///寻找当前帧三角化的路标点
        vector<Vec3d> points3d_cam;
        for(auto &lm : inst.landmarks){
            if(lm.feats.empty())
                continue;
            auto &back_p = lm.feats.back();
            if(back_p.frame == frame && back_p.is_triangulated){
                points3d_cam.emplace_back(world_to_cam(back_p.p_w));
            }
        }

        if(points3d_cam.size() <= para::kInstanceInitMinNum){ //路标数量太少了
            log_text += fmt::format("inst:{} have not enough features,points3d_cam.size():{}\n",inst.id,points3d_cam.size());
            continue;
        }

        State init_state;
        ///根据3d box初始化物体
        if(cfg::use_det3d){
            if(!inst.boxes3d[frame]){
                //未检测到box3d
                log_text += fmt::format("inst:{} have enough features,but not associate box3d\n",inst.id);
                continue;
            }
            ///只初始化刚体
            if(cfg::dataset == DatasetType::kKitti && !(
                    inst.boxes3d[frame]->class_name=="Car" ||
                    inst.boxes3d[frame]->class_name=="Van" ||
                    inst.boxes3d[frame]->class_name=="Truck" ||
                    inst.boxes3d[frame]->class_name=="Tram")){
                continue;
            }

            inst.box3d->dims = inst.boxes3d[frame]->dims;

            //auto init_cam_pt = FitBox3DSimple(points3d_cam,inst.box3d->dims);
            auto init_cam_pt = FitBox3DFromCameraFrame(points3d_cam,inst.box3d->dims);
            if(init_cam_pt){
                init_state.P = cam_to_world(*init_cam_pt);
            }
            else{
                log_text += fmt::format("FitBox3D() inst:{} failed,points3d_cam.size():{}\n",inst.id,points3d_cam.size());
                init_state.P.setZero();
                for(auto &p:points3d_cam){
                    init_state.P += p;
                }
                init_state.P /= points3d_cam.size();
            }
            /*if(cfg::use_plane_constraint){
                if(cfg::is_use_imu){
                    init_state.P.z()=0;
                }
                else{
                    init_state.P.y()=0;
                }
            }*/
            ///根据box初始化物体的位姿和包围框
            /*
            //将包围框的8个顶点转换到世界坐标系下
            Vec3d corner_sum=Vec3d::Zero();
            for(int i=0;i<8;++i){
                corner_sum +=  cam_to_world( inst.boxes3d[frame]->corners.col(i));
            }
            init_state.P = corner_sum/8.;*/

            //init_state.R = Box3D::GetCoordinateRotationFromCorners(corners);//在box中构建坐标系
            init_state.R = e->Rs[frame] * e->ric[0] * inst.boxes3d[frame]->R_cioi();//设置包围框的旋转为初始旋转
        }
        ///无3D框初始化物体
        else{
            inst.box3d->dims = Vec3d(2,4,1.5);

            //auto init_cam_pt = FitBox3DSimple(points3d_cam,inst.box3d->dims);
            auto init_cam_pt = FitBox3DFromCameraFrame(points3d_cam,inst.box3d->dims);

            if(!init_cam_pt){
                log_text += fmt::format("FitBox3D() inst:{} failed,points3d_cam.size():{}\n",inst.id,points3d_cam.size());
                continue;
            }
            init_state.P = cam_to_world(*init_cam_pt);
            init_state.R.setIdentity();
        }


        inst.state[frame].time=e->headers[0];
        inst.vel.SetZero();

        //设置滑动窗口内所有时刻的位姿
        for(int i=0; i <= kWinSize; i++){
            inst.state[i] = init_state;
            inst.state[i].time = e->headers[i];
        }

        inst.is_initial=true;
        inst.history_vel.clear();

        log_text += fmt::format("Initialized id:{},type:{},cnt:{},初始位姿:P:{},R:{} 初始box:{}\n",
                                inst.id, inst.box3d->class_name,points3d_cam.size(), VecToStr(init_state.P),
                                VecToStr(init_state.R.eulerAngles(2,1,0)),
                                VecToStr(inst.box3d->dims));

        ///删去初始化之前的观测
        for(auto it=inst.landmarks.begin(),it_next=it;it!=inst.landmarks.end();it=it_next){
            it_next++;
            if(it->feats.size() == 1 && it->feats.front().frame < frame){
                inst.landmarks.erase(it);
            }
        }
        for(auto &lm:inst.landmarks){
            if(lm.feats.front().frame < frame){
                for(auto it=lm.feats.begin(),it_next=it; it != lm.feats.end(); it=it_next){
                    it_next++;
                    if(it->frame < frame) {
                        lm.feats.erase(it); //删掉掉前面的观测
                    }
                }
                if(lm.depth > 0){
                    lm.depth=-1.0;//需要重新进行三角化
                }
            }
        }

    }

    Debugv(log_text);

}



void InstanceManager::SetDynamicOrStatic(){
    string log_text="InstanceManager::SetDynamicOrStatic\n";

    InstExec([&log_text,this](int key,Instance& inst){

        if(!inst.is_tracking || inst.triangle_num<5){
            return;
        }
        /*
        ///下面根据重投影误差来判断物体的运动
        double err = 0;
        int err_cnt = 0;

        for(auto it=landmarks.begin(),it_next=it;it!=landmarks.end();it=it_next){
            it_next++;
            auto &lm=*it;
            if(lm.feats.empty() || lm.depth <= 0)
                continue;

            ///单目重投影误差
            for(int i=1;i<(int)lm.feats.size(); ++i){
                int imu_i = lm.feats[0].frame;
                int imu_j = lm.feats[i].frame;
                Vec3d pts_w = e->Rs[imu_i] * (e->ric[0] * (lm.depth * lm.feats[0].point) + e->tic[0]) + e->Ps[imu_i];
                Vec3d pts_cj = e->ric[0].transpose() * (e->Rs[imu_j].transpose() * (pts_w - e->Ps[imu_j]) - e->tic[0]);
                Vec2d residual = (pts_cj / pts_cj.z()).head<2>() - lm.feats[i].point.head<2>();
                double re = residual.norm();
                err+=re;
                err_cnt++;
            }
            ///双目重投影误差
            for(int i=0;i<(int)lm.feats.size(); ++i){
                if(lm.feats[i].is_stereo){
                    int imu_i = lm.feats[0].frame;
                    int imu_j = lm.feats[i].frame;
                    Vec3d pts_w = e->Rs[imu_i] * (e->ric[0] * (lm.depth * lm.feats[0].point) + e->tic[0]) + e->Ps[imu_i];
                    Vec3d pts_cj = e->ric[1].transpose() * (e->Rs[imu_j].transpose() * (pts_w - e->Ps[imu_j]) - e->tic[1]);
                    Vec2d residual = (pts_cj / pts_cj.z()).head<2>() - lm.feats[i].point.head<2>();
                    double re = residual.norm();
                    err+=re;
                    err_cnt++;
                }
            }
        }

        if(err_cnt>0){
            double avg_err = err / err_cnt * kFocalLength;
            if(avg_err < para::kInstanceStaticErrThreshold){
                if(!is_static){
                    if(static_cnt >=3 ){
                        static_cnt=0;
                        is_static=true;
                    }
                    else{
                        static_cnt++;
                    }
                }

            }
            Debugv("SetDynamicOrStatic id:{} triangle_num:{} avg_err:{} is_static:{}",id,triangle_num,avg_err,is_static);
        }*/

        /*if(is_init_velocity && vel.v.norm() > 2){
            is_static = false;
            static_frame=0;
            return;
        }*/

        ///下面根据场景流判断物体是否运动
        int cnt=0;
        Vec3d scene_vec=Vec3d::Zero();
        Vec3d point_v=Vec3d::Zero();

        for(auto &lm : inst.landmarks){
            if(lm.depth<=0)
                continue;
            if(lm.feats.size() <= 1)
                continue;
            //计算第一个观测所在的世界坐标
            Vec3d ref_vec;
            double ref_time;
            if(lm.feats.front().is_triangulated){
                ref_vec = lm.feats.front().p_w;
            }
            else{
                //将深度转换到世界坐标系
                ref_vec =  e->Rs[lm.feats.front().frame] * (e->ric[0] * (lm.feats.front().point * lm.depth) + e->tic[0]) +
                        e->Ps[lm.feats.front().frame];
            }
            ref_time= e->headers[lm.feats.front().frame];

            //计算其它观测的世界坐标
            int feat_index=1;
            for(auto feat_it = (++lm.feats.begin());feat_it!=lm.feats.end();++feat_it){
                if(feat_it->is_triangulated){//计算i观测时点的3D位置
                    scene_vec += ( feat_it->p_w - ref_vec ) / feat_index; //加上平均距离向量
                    point_v   += ( feat_it->p_w - ref_vec ) / (e->headers[feat_it->frame] -ref_time);
                    cnt++;
                }
                feat_index++;
            }
        }

        ///根据场景流判断是否是运动物体
        if(cnt>3){
            scene_vec = point_v / cnt;
            Velocity v;
            v.v = scene_vec;
            inst.history_vel.push_back(v);

            if(scene_vec.norm() > 1. || (std::abs(scene_vec.x())>0.8 ||
            std::abs(scene_vec.y())>0.8 || std::abs(scene_vec.z())>0.8) ){
                inst.is_static=false;
                inst.static_frame=0;
            }
            else{
                inst.static_frame++;
            }
        }

        if(!inst.history_vel.empty()){
            inst.point_vel.SetZero();
            for(auto &v:inst.history_vel){
                inst.point_vel.v += v.v;
            }
            inst.point_vel.v /= (double)inst.history_vel.size();
        }
        inst.point_vel.v.y()=0;

        if(inst.history_vel.size()>5){
            inst.history_vel.pop_front();
        }


        if(inst.static_frame>=3){
            inst.is_static=true;
        }

        log_text +=fmt::format("inst_id:{} is_static:{} vec_size:{} scene_vec:{} static_frame:{} point_vel.v:{} \n",
               inst.id,inst.is_static,cnt, VecToStr(scene_vec),inst.static_frame, VecToStr(inst.point_vel.v));

        },true);

    Debugv(log_text);
}


string InstanceManager::PrintInstanceInfo(bool output_lm,bool output_stereo){
    if(tracking_number_<1){
        return {};
    }

    string s =fmt::format("--------------InstanceInfo : {} --------------\n",e->headers[frame]);
    InstExec([&s,&output_lm,&output_stereo](int key,Instance& inst){
        if(inst.is_tracking){
            s+= fmt::format("inst_id:{} landmarks:{} is_init:{} is_tracking:{} is_static:{} "
                            "is_init_v:{} triangle_num:{} curr_avg_depth:{}\n",
                            inst.id,inst.landmarks.size(),inst.is_initial,inst.is_tracking,inst.is_static,inst.is_init_velocity,
                            inst.triangle_num,inst.AverageDepth());
            if(!output_lm)
                return;

            for(int i=0;i<=kWinSize;++i){
                if(inst.boxes3d[i]){
                    s+=fmt::format("{} box:{} dims:{}\n",i, VecToStr(inst.boxes3d[i]->center_pt),
                                   VecToStr(inst.boxes3d[i]->dims));
                }
            }

            for(auto &landmark : inst.landmarks){
                if(landmark.depth>0){
                    int triangle_num=0;
                    for(auto &feat:landmark.feats){
                        if(feat.is_triangulated)
                            triangle_num++;
                    }

                    s+=fmt::format("lid:{} feat_size:{} depth:{} start:{} triangle_num:{}\n",landmark.id,landmark.feats.size(),
                                   landmark.depth,landmark.feats.front().frame,triangle_num);
                    for(auto &feat:landmark.feats){
                        if(feat.is_stereo)
                            s+=fmt::format("S:{}|{} ", VecToStr(feat.point),VecToStr(feat.point_right));
                        else
                            s+=fmt::format("M:{} ", VecToStr(feat.point));
                    }
                    s+= "\n";
                    if(output_stereo){
                        int cnt=0;
                        for(auto &feat:landmark.feats){
                            if(feat.is_triangulated){
                                s+=fmt::format("{}-{} ",feat.frame, VecToStr(feat.p_w));
                                cnt++;
                            }
                        }
                        if(cnt>0) s+= "\n";
                    }
                }
            }
        }

        s+="\n";
        },true);

    s+="\n \n";

    ///将这些信息保存到文件中

    static bool first_run=true;
    if(first_run){
        std::ofstream fout( MyLogger::kLogOutputDir + "features_info.txt", std::ios::out);
        fout.close();
        first_run=false;
    }

    std::ofstream fout( MyLogger::kLogOutputDir + "features_info.txt", std::ios::app);
    fout<<s<<endl;
    fout.close();

    return {};
}


string InstanceManager::PrintInstancePoseInfo(bool output_lm){
    string log_text =fmt::format("--------------PrintInstancePoseInfo : {} --------------\n",e->headers[frame]);

    InstExec([&log_text,&output_lm](int key,Instance& inst){
        log_text += fmt::format("inst_id:{} info:\n box:{} v:{} a:{}\n",inst.id,
                                VecToStr(inst.box3d->dims),VecToStr(inst.vel.v),VecToStr(inst.vel.a));
        for(int i=0; i <= kWinSize; ++i){
            log_text+=fmt::format("{},P:({}),R:({})\n", i, VecToStr(inst.state[i].P),
                                  VecToStr(inst.state[i].R.eulerAngles(2,1,0)));
        }
        if(!output_lm)
            return;
        int lm_cnt=0;
        for(auto &lm: inst.landmarks){
            if(lm.depth<=0)continue;
            lm_cnt++;
            if(lm_cnt%5==0) log_text += "\n";
            log_text += fmt::format("<lid:{},n:{},d:{:.2f}> ",lm.id,lm.feats.size(),lm.depth);
        }
        log_text+="\n";

    });

    ///将这些信息保存到文件中

    static bool first_run=true;
    if(first_run){
        std::ofstream fout( MyLogger::kLogOutputDir + "pose_info.txt", std::ios::out);
        fout.close();
        first_run=false;
    }

    std::ofstream fout( MyLogger::kLogOutputDir + "pose_info.txt", std::ios::app);
    fout<<log_text<<endl;
    fout.close();

    return {};
}


/**
 * 设置所有动态物体的最新的位姿,dims信息到输出变量
 */
void InstanceManager::SetOutputInstInfo(){
    std::unique_lock<std::mutex> lk(vel_mutex_);
    insts_output.clear();
    if(tracking_number_<1){
        return;
    }
    //string log_text = "SetOutputInstInfo 物体的速度信息:";
    InstExec([this](int key,Instance& inst){
        //log_text += fmt::format("inst:{} v:{} a:{}", inst.id, VecToStr(inst.vel.v), VecToStr(inst.vel.a));
        InstEstimatedInfo estimated_info;
        estimated_info.is_init = inst.is_initial;
        estimated_info.is_init_velocity = inst.is_init_velocity;
        estimated_info.time = inst.state[frame].time;
        estimated_info.P = inst.state[frame].P;
        estimated_info.R = inst.state[frame].R;
        estimated_info.a = inst.vel.a;
        estimated_info.v = inst.vel.v;
        estimated_info.dims = inst.box3d->dims;

        insts_output.insert({inst.id, estimated_info});;
    });
    //Debugv(log_text);
}

/**
 * 保存所有物体在当前帧的位姿
 */
void InstanceManager::SaveTrajectory(){
    if(cfg::slam != SlamType::kDynamic){
        return;
    }

    Mat3d R_iw = e->Rs[frame].transpose();
    Vec3d P_iw = - R_iw * e->Ps[frame];
    Mat3d R_ci = e->ric[0].transpose();
    Vec3d P_ci = - R_ci * e->tic[0];

    Mat3d R_cw = R_ci * R_iw;
    Vec3d P_cw = R_ci * P_iw + P_ci;

    string object_tracking_path= io_para::kOutputFolder + cfg::kDatasetSequence+".txt";
    string object_object_dir= io_para::kOutputFolder + cfg::kDatasetSequence+"/";
    string object_tum_dir=io_para::kOutputFolder + cfg::kDatasetSequence+"_tum/";

    static bool first_run=true;
    if(first_run){
        std::ofstream fout(object_tracking_path, std::ios::out);
        fout.close();

        ClearDirectory(object_object_dir);//获取目录中所有的路径,并将这些文件全部删除
        ClearDirectory(object_tum_dir);

        first_run=false;
    }

    string object_object_path;
    if(cfg::dataset == DatasetType::kViode){
        object_object_path = object_object_dir+DoubleToStr(e->feature_frame.time,6)+".txt";
    }
    else{
        object_object_path = object_object_dir+ PadNumber(e->feature_frame.seq_id,6)+".txt";
    }
    std::ofstream fout_object(object_object_path, std::ios::out);

    std::ofstream fout_mot(object_tracking_path, std::ios::out | std::ios::app);//追加写入

    string log_text="InstanceManager::SaveTrajectory()\n";

    for(auto &[inst_id,inst] : instances){
        log_text+=fmt::format("inst_id:{},is_tracking:{},is_initial:{},is_curr_visible:{},"
                              "is_init_velocity:{},is_static:{},landmarks:{},triangle_num:{}\n",
               inst_id,inst.is_tracking,inst.is_initial,inst.is_curr_visible,inst.is_init_velocity,
               inst.is_static,inst.landmarks.size(),inst.triangle_num);

        if(!inst.is_tracking || !inst.is_initial)
            continue;
        if(!inst.is_curr_visible)
            continue;

        ///将最老帧的轨迹保存到历史轨迹中
        inst.history_pose.push_back(inst.state[kWinSize]);
        if(inst.history_pose.size()>100){
            inst.history_pose.erase(inst.history_pose.begin());
        }

        string object_tum_path=object_tum_dir+std::to_string(inst_id)+".txt";
        Eigen::Quaterniond q_obj(inst.state[kWinSize].R);
        std::ofstream foutC(object_tum_path,std::ios::out | std::ios::app);
        foutC.setf(std::ios::fixed, std::ios::floatfield);
        foutC << e->headers[kWinSize] << " "
        << inst.state[kWinSize].P.x() << " "
        << inst.state[kWinSize].P.y() << " "
        << inst.state[kWinSize].P.z() << " "
        <<q_obj.x()<<" "
        <<q_obj.y()<<" "
        <<q_obj.z()<<" "
        <<q_obj.w()<<endl;
        foutC.close();

        if(cfg::dataset==DatasetType::kKitti){
            ///变换到相机坐标系下
            Mat3d R_coi = R_cw * inst.state[frame].R;
            Vec3d P_coi = R_cw * inst.state[frame].P + P_cw;
            ///将物体位姿变换为kitti的位姿，即物体中心位于底部中心
            Mat3d R_offset;
            R_offset<<1,0,0,  0,cos(-M_PI/2),-sin(-M_PI/2),  0,sin(-M_PI/2),cos(-M_PI/2);
            double H = inst.box3d->dims.y();
            Vec3d P_offset(0,0,-H/2);

            Mat3d R_coi_kitti = R_offset * R_coi;
            Vec3d P_coi_kitti = R_offset*P_coi + P_offset;

            ///计算新的dims

            ///计算rotation_y
            //Vec3d eulerAngle=R_coi_kitti.eulerAngles(2,1,0);
            //double rotation_y = eulerAngle.y();
            Vector3d obj_z = R_coi_kitti * Vec3d(0,0,1);//车辆Z轴在相机坐标系下的方向
            Debugv("inst:{} obj_z:{}",inst.id, VecToStr(obj_z));
            Vector3d cam_x(1,0,0);//相机坐标系Z轴的向量
            double rotation_y = atan2(obj_z.z(),obj_z.x()) - atan2(cam_x.z(),cam_x.x());


            /// 计算观测角度 alpha
            double alpha = rotation_y;
            alpha += atan2(P_coi_kitti.z(),P_coi_kitti.x()) + 1.5 * M_PI;
            while(alpha > M_PI){
                alpha-=M_PI;
            }
            while(alpha < -M_PI){
                alpha+=M_PI;
            }

            ///计算3D包围框投影得到的2D包围框
            /*Vec3d minPt = - inst.box3d->dims/2;
            Vec3d maxPt = inst.box3d->dims/2;
            EigenContainer<Vec3d> vertex(8);
            vertex[0]=minPt;
            vertex[1].x()=maxPt.x();vertex[1].y()=minPt.y();vertex[1].z()=minPt.z();
            vertex[2].x()=maxPt.x();vertex[2].y()=minPt.y();vertex[2].z()=maxPt.z();
            vertex[3].x()=minPt.x();vertex[3].y()=minPt.y();vertex[3].z()=maxPt.z();
            vertex[4].x()=minPt.x();vertex[4].y()=maxPt.y();vertex[4].z()=maxPt.z();
            vertex[5] = maxPt;
            vertex[6].x()=maxPt.x();vertex[6].y()=maxPt.y();vertex[6].z()=minPt.z();
            vertex[7].x()=minPt.x();vertex[7].y()=maxPt.y();vertex[7].z()=minPt.z();
            for(int i=0;i<8;++i){
                vertex[i] = R_coi * vertex[i] + P_coi;
            }
            //投影点
            Mat28d corners_2d;
            for(int i=0;i<8;++i){
                Vec2d p;
                cam0->ProjectPoint(vertex[i],p);
                corners_2d.col(i) = p;
            }
            Vec2d corner2d_min_pt = corners_2d.rowwise().minCoeff();//包围框左上角的坐标
            Vec2d corner2d_max_pt = corners_2d.rowwise().maxCoeff();//包围框右下角的坐标*/


            ///保存为KITTI Tracking模式
            //帧号
            fout_mot << e->feature_frame.seq_id << " " <<
            //物体的id
            inst.id << " "<<
            //类别名称
            kitti::GetKittiName(inst.box3d->class_id) <<" "<<
            // truncated    Integer (0,1,2)
            -1<<" "<<
            //occluded     Integer (0,1,2,3)
            -1<<" "<<
            // Observation angle of object, ranging [-pi..pi]
            alpha<<" "<<
            // 2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates
            inst.box2d->min_pt.x<<" "<<inst.box2d->min_pt.y<<" "<<inst.box2d->max_pt.x<<" "<<inst.box2d->max_pt.y<<" "<<
            // 3D object dimensions: height, width, length (in meters)
            VecToStr(inst.box3d->dims)<< //VecToStr()函数会输出一个空格
            // 3D object location x,y,z in camera coordinates (in meters)
            VecToStr(P_coi)<<
            // Rotation ry around Y-axis in camera coordinates [-pi..pi]
            rotation_y<<" "<<
            //  Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better.
            inst.box3d->score<<
            endl;


            ///保存为KITTI Object模式
            //类别名称
            fout_object <<kitti::GetKittiName(inst.box3d->class_id) <<" "<<
            //Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries
            -1<<" "<<
            //occluded     Integer (0,1,2,3)
            -1<<" "<<
            // Observation angle of object, ranging [-pi..pi]
            alpha<<" "<<
            // 2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates
            inst.box2d->min_pt.x<<" "<<inst.box2d->min_pt.y<<" "<<inst.box2d->max_pt.x<<" "<<inst.box2d->max_pt.y<<" "<<
            // 3D object dimensions: height, width, length (in meters)
            VecToStr(inst.box3d->dims)<< //VecToStr()函数会输出一个空格
            // 3D object location x,y,z in camera coordinates (in meters)
            VecToStr(P_coi)<<
            // Rotation ry around Y-axis in camera coordinates [-pi..pi]
            rotation_y<<" "<<
            //  Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better.
            inst.box3d->score<<
            endl;
        }

    }


    fout_mot.close();
    fout_object.close();

    Debugv(log_text);
}


void InstanceManager::GetOptimizationParameters()
{
    InstExec([](int key,Instance& inst){
        inst.GetOptimizationParameters();
    });
}


/**
 * 设置需要特别参数化的优化变量
 * @param problem
 */
void InstanceManager::AddInstanceParameterBlock(ceres::Problem &problem)
{
    InstExec([&problem,this](int key,Instance& inst){
        inst.SetOptimizeParameters();

        if(cfg::use_plane_constraint){
            for(int i=0;i<=frame; i++){
                problem.AddParameterBlock(inst.para_state[i], kSizePose,
                                          new PoseConstraintLocalParameterization());

                /*problem.AddParameterBlock(inst.para_speed[0],6,
                                          new SpeedConstraintLocalParameterization());*/
            }
        }
        else{
            for(int i=0;i<=frame; i++){
                problem.AddParameterBlock(inst.para_state[i], kSizePose, new PoseLocalParameterization());
            }
        }


    });
}


void InstanceManager::AddResidualBlockForInstOpt(ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    if(tracking_number_ < 1)
        return;

    for(auto &[key,inst] : instances){
        if(!inst.is_initial || !inst.is_tracking)
            continue;
        if(inst.landmarks.size()<1)
            continue;

        std::unordered_map<string,int> statistics;//用于统计每个误差项的数量

        ///添加包围框预测误差
        for(int i=0;i<=kWinSize;++i){
            if(inst.boxes3d[i]){
                ///包围框大小误差
                problem.AddResidualBlock(new BoxDimsFactor(inst.boxes3d[i]->dims),loss_function,inst.para_box[0]);
                statistics["BoxDimsFactor"]++;
                ///物体的方向误差
                //Mat3d R_cioi = Box3D::GetCoordinateRotationFromCorners(inst.boxes3d[i]->corners);
                Mat3d R_cioi = inst.boxes3d[i]->R_cioi();
                problem.AddResidualBlock(
                        new BoxOrientationFactor(R_cioi,e->ric[0]),
                        nullptr,e->para_Pose[i],inst.para_state[i]);
                statistics["BoxOrientationFactor"]++;

                /*problem.AddResidualBlock(new BoxPoseFactor(R_cioi,inst.boxes3d[i]->center,e->ric[0],e->tic[0]),
                                         loss_function,e->para_Pose[i],inst.para_state[i]);
                statistics["BoxPoseFactor"]++;*/

                ///包围框中心误差
                if(inst.boxes3d[i]->center_pt.norm() < 50){
                    problem.AddResidualBlock(new BoxPositionFactor(inst.boxes3d[i]->center_pt,
                                                                   e->ric[0],e->tic[0],
                                                                   e->Rs[i],e->Ps[i]),
                                             loss_function,inst.para_state[i]);
                    statistics["BoxPositionFactor"]++;
                }


                /*///添加顶点误差
                //效果不好
                problem.AddResidualBlock(new BoxVertexFactor(inst.boxes3d[i]->corners,inst.boxes3d[i]->dims,Vec3d(-1,-1,-1),
                                                             e->ric[0],e->Rs[i]),
                                         loss_function,inst.para_state[i],inst.para_box[0]);
               problem.AddResidualBlock(new BoxVertexFactor(inst.boxes3d[i]->corners,inst.boxes3d[i]->dims,Vec3d(-1,-1,1),
                                                            e->ric[0],e->Rs[i]),
                                        loss_function,inst.para_state[i],inst.para_box[0]);
               problem.AddResidualBlock(new BoxVertexFactor(inst.boxes3d[i]->corners,inst.boxes3d[i]->dims,Vec3d(-1,1,1),
                                                            e->ric[0],e->Rs[i]),
                                        loss_function,inst.para_state[i],inst.para_box[0]);
               problem.AddResidualBlock(new BoxVertexFactor(inst.boxes3d[i]->corners,inst.boxes3d[i]->dims,Vec3d(-1,-1,1),
                                                            e->ric[0],e->Rs[i]),
                                        loss_function,inst.para_state[i],inst.para_box[0]);
               problem.AddResidualBlock(new BoxVertexFactor(inst.boxes3d[i]->corners,inst.boxes3d[i]->dims,Vec3d(1,-1,-1),
                                                            e->ric[0],e->Rs[i]),
                                        loss_function,inst.para_state[i],inst.para_box[0]);
               problem.AddResidualBlock(new BoxVertexFactor(inst.boxes3d[i]->corners,inst.boxes3d[i]->dims,Vec3d(1,-1,1),
                                                            e->ric[0],e->Rs[i]),
                                        loss_function,inst.para_state[i],inst.para_box[0]);
               problem.AddResidualBlock(new BoxVertexFactor(inst.boxes3d[i]->corners,inst.boxes3d[i]->dims,Vec3d(1,1,1),
                                                            e->ric[0],e->Rs[i]),
                                        loss_function,inst.para_state[i],inst.para_box[0]);
               problem.AddResidualBlock(new BoxVertexFactor(inst.boxes3d[i]->corners,inst.boxes3d[i]->dims,Vec3d(1,-1,1),
                                                            e->ric[0],e->Rs[i]),
                                        loss_function,inst.para_state[i],inst.para_box[0]);
               statistics["BoxVertexFactor"]++;*/
            }
        }
        if(inst.boxes3d[kWinSize]){
            problem.AddResidualBlock(new BoxPositionFactor(inst.boxes3d[kWinSize]->center_pt,
                                                               e->ric[0], e->tic[0],
                                                               e->Rs[kWinSize], e->Ps[kWinSize]),
                                     loss_function,inst.para_state[kWinSize]);
            statistics["BoxPositionFactor"]++;
        }


        int depth_index=-1;//注意,这里的depth_index的赋值过程要与 Instance::SetOptimizeParameters()中depth_index的赋值过程一致
        for(auto &lm : inst.landmarks){
            if(lm.depth < 0.2)
                continue;
            depth_index++;

            auto &feat_j=lm.feats.front();

            ///根据3D应该要落在包围框内产生的误差
            for(auto &feat : lm.feats){
                if(feat.is_triangulated){
                    problem.AddResidualBlock(
                            new BoxEncloseStereoPointFactor(feat.p_w,inst.box3d->dims,inst.id),
                                             loss_function,
                                             inst.para_state[feat.frame]);
                    statistics["BoxEncloseStereoPointFactor"]++;
                }
            }
            problem.AddResidualBlock(new BoxEncloseTrianglePointFactor(
                    feat_j.point,feat_j.vel,e->Rs[feat_j.frame],e->Ps[feat_j.frame],
                    e->ric[0],e->tic[0],feat_j.td,e->td),
                                     loss_function,
                                     inst.para_state[feat_j.frame],inst.para_box[0],inst.para_inv_depth[depth_index]);
            statistics["BoxEncloseTrianglePointFactor"]++;


            if(inst.is_static){
                continue;
            }

            ///位姿、点云、速度的约束
            /*problem.AddResidualBlock(new InstanceInitPowFactor(
                            feat_j.point,feat_j.vel,e->Rs[fj],e->Ps[fj],
                            e->ric[0],e->tic[0],feat_j.td, e->td),
                            loss_function,
                            inst.para_state[fj],
                            inst.para_inv_depth[depth_index]);*/
            /*problem.AddResidualBlock(new InstanceInitPowFactorSpeed(
                    feat_j.point, feat_j.vel, e->Rs[fj], e->Ps[fj],
                    e->ric[0], e->tic[0], feat_j.td, e->td,
                    e->headers[fj], e->headers[0], 1.0),
                                     loss_function,
                                     inst.para_state[0],
                                     inst.para_speed[0],
                                     inst.para_inv_depth[depth_index]);*/

            if(lm.feats.size() < 2)
                continue;

            ///优化物体速度
            if(!inst.is_static && inst.is_init_velocity){
                /*std::list<FeaturePoint>::iterator last_point,end_point;
                bool found_first=false;
                for(auto it=lm.feats.begin();it!=lm.feats.end();++it){
                    if(it->is_triangulated){
                        if(!found_first){
                            last_point=it;
                            found_first=true;
                        }
                        else{
                            double time_ij = e->headers[it->frame] - e->headers[last_point->frame];
                            problem.AddResidualBlock(
                                    new SpeedStereoPointFactor(last_point->p_w,it->p_w,time_ij),
                                    loss_function,inst.para_speed[0]);
                            statistics["SpeedStereoPointFactor"]++;

                            problem.AddResidualBlock(
                                    new ConstSpeedStereoPointFactor(first_point->p_w,end_point->p_w,time_ij,
                                                                    inst.last_vel.v,inst.last_vel.a),
                                                                    loss_function,inst.para_speed[0]);
                            statistics["ConstSpeedStereoPointFactor"]++;
                            last_point=it;
                        }
                    }
                }*/

                std::list<FeaturePoint>::iterator first_point,end_point;
                bool found_first=false,found_end=false;
                for(auto it=lm.feats.begin();it!=lm.feats.end();++it){
                    if(it->is_triangulated){
                        if(!found_first){
                            first_point=it;
                            found_first=true;
                        }
                        else{
                            end_point=it;
                            found_end=true;
                        }
                    }
                }
                /*if(found_first && found_end){
                    double time_ij = e->headers[end_point->frame] - e->headers[first_point->frame];
                    problem.AddResidualBlock(
                            new SpeedStereoPointFactor(end_point->p_w,first_point->p_w,time_ij),
                            loss_function,inst.para_speed[0]);
                    statistics["SpeedStereoPointFactor"]++;

                    problem.AddResidualBlock(
                            new ConstSpeedStereoPointFactor(first_point->p_w,end_point->p_w,time_ij,
                                                            inst.last_vel.v,inst.last_vel.a),
                                                            loss_function,inst.para_speed[0]);
                    statistics["ConstSpeedStereoPointFactor"]++;

                }*/

            }

        } //inst.landmarks

        ///速度-位姿误差
        if(!inst.is_static && inst.is_init_velocity){
            for(int i=0;i <= kWinSize-1;++i){
                /*problem.AddResidualBlock(new SpeedPoseFactor(inst.state[i].time,inst.state[kWinSize].time),
                                         loss_function,inst.para_state[i],inst.para_state[kWinSize],inst.para_speed[0]);
                statistics["SpeedPoseFactor"]++;*/

                /*problem.AddResidualBlock(new SpeedPoseSimpleFactor(inst.state[i-1].time,
                 inst.state[i].time,inst.state[i-1].R,inst.state[i-1].P,
                 inst.state[i].R,inst.state[i].P),
                                         loss_function,inst.para_speed[0]);
                statistics["SpeedPoseSimpleFactor"]++;*/

                /*problem.AddResidualBlock(new SpeedPoseFactor(inst.state[i].time,inst.state[i+1].time),
                                         loss_function,inst.para_state[i],inst.para_state[i+1],inst.para_speed[0]);
                statistics["SpeedPoseFactor"]++;*/
            }
        }


        //log
        string log_text;
        for(auto &pair : statistics)  if(pair.second>0) log_text += fmt::format("{} : {}\n",pair.first,pair.second);
        Debugv("inst:{} 各个残差项的数量: \n{}",inst.id,log_text);

    } //inst


}



void InstanceManager::Optimization(){
    TicToc tt,t_all;

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

    if(para::is_print_detail){
        PrintInstancePoseInfo(true);
    }
    ///添加残差块
    AddInstanceParameterBlock(problem);

    AddResidualBlockForInstOpt(problem,loss_function);

    Debugv("InstanceManager::Optimization | prepare:{} ms",tt.TocThenTic());

    ///设置ceres选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = para::KNumIter;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    options.max_solver_time_in_seconds = para::kMaxSolverTime;

    ///求解
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Debugv("InstanceManager::Optimization 优化完成 Iterations: {}", summary.iterations.size());
    Debugv("InstanceManager::Optimization | Solve:{} ms",tt.TocThenTic());

    GetOptimizationParameters();

    if(para::is_print_detail){
        PrintInstancePoseInfo(true);
    }

    Debugv("InstanceManager::Optimization all:{} ms",t_all.Toc());
}



/**
 * 添加残差块
 * @param problem
 * @param loss_function
 */
void InstanceManager::AddResidualBlockForJointOpt(ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    if(tracking_number_ < 1)
        return;
    std::unordered_map<string,int> statistics;//用于统计每个误差项的数量

    for(auto &[key,inst] : instances){
        if(!inst.is_initial || !inst.is_tracking)
            continue;
        if(inst.landmarks.size()<5)
            continue;

        int depth_index=-1;//注意,这里的depth_index的赋值过程要与 Instance::SetOptimizeParameters()中depth_index的赋值过程一致
        for(auto &lm : inst.landmarks){
            if(lm.depth < 0.2)
                continue;
            depth_index++;
            auto &feat_j=lm.feats.front();
            int fj=feat_j.frame;

            ///第一个特征点只用来优化深度
            if(cfg::is_stereo && feat_j.is_stereo){
                /*problem.AddResidualBlock(
                        new ProjInst12Factor(feat_j.point,feat_j.point_right),
                        loss_function,
                        e->para_ex_pose[0],
                        e->para_ex_pose[1],
                        inst.para_inv_depth[depth_index]);*/
                problem.AddResidualBlock(
                        new ProjInst12FactorSimple(feat_j.point,feat_j.point_right,
                                                   e->ric[0],e->tic[0],e->ric[1],e->tic[1]),
                        loss_function,
                        inst.para_inv_depth[depth_index]);
                statistics["ProjInst12FactorSimple"]++;
            }

            if(! inst.is_static){
                continue;
            }

            ///位姿、点云、速度的约束
            /*problem.AddResidualBlock(new InstanceInitPowFactor(
                            feat_j.point,feat_j.vel,e->Rs[fj],e->Ps[fj],
                            e->ric[0],e->tic[0],feat_j.td, e->td),
                            loss_function,
                            inst.para_state[fj],
                            inst.para_inv_depth[depth_index]);*/
            /*problem.AddResidualBlock(new InstanceInitPowFactorSpeed(
                    feat_j.point, feat_j.vel, e->Rs[fj], e->Ps[fj],
                    e->ric[0], e->tic[0], feat_j.td, e->td,
                    e->headers[fj], e->headers[0], 1.0),
                                     loss_function,
                                     inst.para_state[0],
                                     inst.para_speed[0],
                                     inst.para_inv_depth[depth_index]);*/

            if(lm.feats.size() < 2)
                continue;

            for(auto feat_it = (++lm.feats.begin()); feat_it !=lm.feats.end();++feat_it ){
                int fi = feat_it->frame;

                /*problem.AddResidualBlock(
                        new ProjectionInstanceFactor(feat_j.point,feat_it->point,feat_j.vel,feat_it->vel,
                                                     feat_j.td,feat_it->td,e->td),
                        loss_function,
                        e->para_Pose[fj],
                        e->para_Pose[fi],
                        e->para_ex_pose[0],
                        inst.para_state[fj],
                        inst.para_state[fi],
                        inst.para_inv_depth[depth_index]
                        );
                statistics["ProjectionInstanceFactor"]++;*/

                //double factor= 1.;//track_3>=5 ? 5. : 1.;
                ///优化相机位姿、物体的速度和位姿
                /*problem.AddResidualBlock(
                        new SpeedPoseFactor(feat_j.point,e->Headers[fj],e->Headers[fi]),
                        loss_function,
                        e->para_Pose[fj],
                        e->para_ex_pose[0],
                        inst.para_state[fj],
                        inst.para_state[fi],
                        inst.para_speed[0],
                        inst.para_inv_depth[depth_index]);*/

                ///优化相机位姿和物体的速度
                /*problem.AddResidualBlock(new ProjectionSpeedFactor(
                        feat_j.point, feat_it->point,feat_j.vel, feat_it->vel,
                        e->ric[0], e->tic[0],e->ric[0],e->tic[0],
                        feat_j.td, feat_it->td, e->td,e->headers[fj],e->headers[fi]),
                                         loss_function,
                                         e->para_Pose[fj],
                                         e->para_Pose[fi],
                                         inst.para_speed[0],
                                         inst.para_inv_depth[depth_index]);*/
                /*problem.AddResidualBlock(new ProjectionSpeedSimpleFactor(
                        feat_j.point, feat_i.point,feat_j.vel, feat_i.vel,
                        feat_j.td, feat_i.td, e->td, e->Headers[fj],e->Headers[fi],
                        e->Rs[fj], e->Ps[fj],e->Rs[fi], e->Ps[fi],
                        e->ric[0], e->tic[0],e->ric[0],e->tic[0],1.),
                                         loss_function,
                                         inst.para_speed[0],
                                         inst.para_inv_depth[depth_index]
                        );*/
                ///优化物体的速度和位姿
                /*problem.AddResidualBlock(new SpeedPoseSimpleFactor(
                        feat_j.point, e->headers[feat_j.frame], e->headers[feat_i.frame],
                        e->Rs[feat_j.frame], e->Ps[feat_j.frame], e->ric[0],
                        e->tic[0],feat_j.vel, feat_j.td, e->td),
                                         loss_function,
                                         inst.para_state[feat_j.frame],
                                         inst.para_state[feat_i.frame],
                                         inst.para_speed[0],
                                         inst.para_inv_depth[depth_index]);*/


                if(cfg::is_stereo && feat_it->is_stereo){
                    ///优化物体的位姿
                     //这一项效果不好,TODO
                     /*problem.AddResidualBlock(
                            new ProjInst22SimpleFactor(
                                    feat_j.point,feat_it->point,feat_j.vel,feat_it->vel,
                                    e->Rs[feat_j.frame],e->Ps[feat_j.frame],
                                    e->Rs[feat_it->frame],e->Ps[feat_it->frame],
                                    e->ric[0],e->tic[0],
                                    e->ric[1],e->tic[1],
                                    feat_j.td,feat_it->td,e->td,lm.id),
                                    loss_function,
                                    inst.para_state[feat_j.frame],
                                    inst.para_state[feat_it->frame],
                                    inst.para_inv_depth[depth_index]);
                     statistics["ProjInst22SimpleFactor"]++;*/

                    if(lm.feats.size() >= 2){

                        ///优化相机位姿和物体速度和特征点深度
                        /*problem.AddResidualBlock(new ProjectionSpeedFactor(
                                feat_j.point, feat_i.point_right,feat_j.vel, feat_i.vel_right,
                                e->ric[0], e->tic[0],e->ric[1],e->tic[1],
                                feat_j.td, feat_i.td, e->td,
                                e->Headers[fj],e->Headers[fi],factor),
                                                 loss_function,
                                                 e->para_Pose[fj],
                                                 e->para_Pose[fi],
                                                 inst.para_speed[0],
                                                 inst.para_inv_depth[depth_index]);*/
                    }
                }
            } // feat

        } //inst.landmarks

    } //inst

    //log
    string log_text;
    for(auto &pair : statistics)  if(pair.second>0) log_text += fmt::format("{} : {}\n",pair.first,pair.second);
    Debugv("各个残差项的数量: \n{}",log_text);
}


}