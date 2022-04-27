/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/
#include "instance_manager.h"

#include <algorithm>

#include "estimator.h"
#include "utils/visualization.h"
#include "utils/def.h"
#include "vio_parameters.h"

namespace dynamic_vins{\


void InstanceManager::set_estimator(Estimator* estimator){
    e_=estimator;
}

void InstanceManager::Triangulate(int frame_cnt)
{
    if(tracking_number_ < 1)
        return;
    Infov("triangulate 三角化:");

    auto getCamPose=[this](int index,int cam_id){
        assert(cam_id == 0 || cam_id == 1);
        Eigen::Matrix<double, 3, 4> pose;
        Vec3d t0 = e_->Ps[index] + e_->Rs[index] * e_->tic[cam_id];
        Mat3d R0 = e_->Rs[index] * e_->ric[cam_id];
        pose.leftCols<3>() = R0.transpose();
        pose.rightCols<1>() = -R0.transpose() * t0;
        return pose;
    };

    int num_triangle=0,num_failed=0,num_delete_landmark=0,num_mono=0;
    for(auto &[key,inst] : instances)
    {
        if(! inst.is_tracking)
            continue;

        string lm_msg;
        int inst_add_num=0;

        //计算平均深度
        double avg_depth=0.;
        int depth_cnt=0;
        for(auto &ld : inst.landmarks){
            if(ld.depth > 0){
                depth_cnt++;
                avg_depth+=ld.depth;
            }
        }
        avg_depth /= depth_cnt;

        for(auto it=inst.landmarks.begin(),it_next=it;it!=inst.landmarks.end();it=it_next)
        {
            it_next++;
            auto &lm=*it;
            if(lm.depth > 0 || lm.feats.empty())
                continue;
            ///双目三角化
            if(cfg::is_stereo && lm.feats[0].is_stereo)
            {
                int imu_i = lm.feats[0].frame;
                auto leftPose = getCamPose(imu_i,0);
                auto rightPose= getCamPose(imu_i,1);

                Vec2d point0 = lm.feats[0].point.head(2);
                Vec2d point1 = lm.feats[0].point_right.head(2);
                Eigen::Vector3d point3d_w;
                TriangulatePoint(leftPose, rightPose, point0, point1, point3d_w);
                Eigen::Vector3d localPoint = leftPose.leftCols<3>() * point3d_w + leftPose.rightCols<1>();
                double depth = localPoint.z();
                if (depth > 0.1){
                    //若未初始化或路标点太少，则加入
                    if(!inst.is_initial || depth_cnt < 5){
                        lm.depth = depth;
                        inst_add_num++;
                        lm_msg+=fmt::format("lid:{} NotInit S d:{:.2f} p:{}\n", lm.id, depth, VecToStr(point3d_w));
                    }
                    //判断深度值是否符合
                    else{
                        Eigen::Vector3d pts_obj_j=inst.state[lm.feats[0].frame].R.transpose() *
                                (point3d_w - inst.state[lm.feats[0].frame].P);
                        if(pts_obj_j.norm() < inst.box.norm()*4){
                            lm.depth = depth;
                            inst_add_num++;
                            lm_msg+=fmt::format("lid:{} S d:{:.2f} p:{}\n", lm.id, depth, VecToStr(point3d_w));
                        }
                        else{
                            lm_msg+=fmt::format("lid:{} outbox d:{:.2f} p:{}\n", lm.id, depth, VecToStr(point3d_w));
                        }
                    }
                }
                else{
                    num_failed++;
                    if(lm.feats.size() == 1){//删除该路标
                        inst.landmarks.erase(it);
                        num_delete_landmark++;
                    }
                    else{//删除该观测
                        lm.feats.erase(lm.feats.begin());
                    }
                }

                /*
                Vector3d ptsGt = pts_gt[it_per_id.feature_id];
                printf("stereo %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                                ptsGt.x(), ptsGt.y(), ptsGt.z());
                */
            }
            ///单目三角化，以开始帧和开始帧的后一帧的观测作为三角化的点
            else if(lm.feats.size() >= 2)
            {
                int imu_i = lm.feats[0].frame;
                Eigen::Matrix<double, 3, 4> leftPose = getCamPose(imu_i,0);
                int imu_j=lm.feats[1].frame;
                Eigen::Matrix<double, 3, 4> rightPose = getCamPose(imu_j,0);
                Vec2d point0 = lm.feats[0].point.head(2);
                Vec2d point1 = lm.feats[1].point.head(2);
                Vec3d point3d;

                //TriangulatePoint(leftPose, rightPose, point0, point1, point3d);
                //double delta_time=e->Headers[imu_j] - e->Headers[imu_i];
                //TriangulateDynamicPoint(leftPose,rightPose,point0,point1,inst.speed_v,inst.speed_a,delta_time,point3d);

                int index_j=lm.feats[0].frame;
                int index_i=lm.feats[1].frame;
                TriangulateDynamicPoint(leftPose, rightPose, point0, point1,
                                        inst.state[index_j].R, inst.state[index_j].P,
                                        inst.state[index_i].R, inst.state[index_i].P, point3d);

                Vec3d localPoint;
                localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
                double depth = localPoint.z();
                if (depth > 0.1){
                    if(!inst.is_initial || depth_cnt < 5 || depth <= 4 * avg_depth){
                        lm.depth = depth;
                        num_mono++;
                        inst_add_num++;
                        lm_msg+=fmt::format("lid:{} M d:{:.2f} p:{}\n", lm.id, depth, VecToStr(point3d));
                    }
                }
                else{
                    num_failed++;
                    lm.feats.erase(lm.feats.begin());//删除该观测
                }

                /*
                Vector3d ptsGt = pts_gt[it_per_id.feature_id];
                printf("motion  %d pts: %f %f %f gt: %f %f %f \n",it_per_id.feature_id, point3d.x(), point3d.y(), point3d.z(),
                                                                ptsGt.x(), ptsGt.y(), ptsGt.z());
                */
            }
        }

        int num_depth=0;
        for(auto &ld : inst.landmarks){
            if(ld.depth > 0)num_depth++;
        }
        Debugv("inst:{} landmarks.size{} depth.size:{} avg_depth:{} new_add:{} \n{}",
               inst.id, inst.landmarks.size(), num_depth, avg_depth, inst_add_num, lm_msg);
        num_triangle += inst_add_num;
    }

    if(num_triangle>0 || num_delete_landmark>0)
        Debugv("增加:{}=(M:{},S:{}) 失败:{} 删除:{}", num_triangle, num_mono, num_triangle - num_mono,
               num_failed, num_delete_landmark);
}


/**
 * 根据速度和上一帧的位姿预测动态物体在当前帧的位姿
 */
void InstanceManager::PredictCurrentPose()
{
    if(tracking_number_ < 1)
        return;

    SetVelMap();

    Infov("predictCurrentPose 位姿递推");

    int i=e_->frame;
    int j= e_->frame - 1;

    double time_ij= e_->headers[i] - e_->headers[j];


    for(auto &[key,inst] : instances){
        if(!inst.is_tracking || !inst.is_initial){
            continue;
        }
        inst.state[i].time = e_->headers[i];
        Mat3d Roioj=Sophus::SO3d::exp(inst.vel.a*time_ij).matrix();
        Vec3d Poioj=inst.vel.v*time_ij;
        inst.state[i].R=Roioj * inst.state[j].R;
        inst.state[i].P=Roioj* inst.state[j].P + Poioj;

        Debugv("inst:{} v:{} a:{}", inst.id, VecToStr(inst.vel.v), VecToStr(inst.vel.a));
        /*      State new_pose;
                new_pose.time=e->Headers[1];
                double time_ij=new_pose.time - inst.state[0].time;

                printf("Time: new_pose_time:%.2lf\n",new_pose.time);
                for(int i=0;i<=kWindowSize;i++){
                    printf("%d:%.2lf ",i,inst.state[i].time);
                }
                cout<<endl;


                Mat3d Roioj=Sophus::SO3d::exp(inst.speed_a*time_ij).matrix();
                Vec3d Poioj=inst.speed_v*time_ij;
                new_pose.R=Roioj * inst.state[0].R;
                new_pose.P=Roioj* inst.state[0].P + Poioj;

                printf("Inst:%d old:(%.2lf,%.2lf,%.2lf) new:(%.2lf,%.2lf,%.2lf) \n",inst.id,inst.state[0].P.x(),
                inst.state[0].P.y(),inst.state[0].P.z(),new_pose.P.x(),new_pose.P.y(),new_pose.P.z());
                cout<<"time:"<<time_ij<<endl;
                cout<<"Roioj:\n"<<Roioj.matrix()<<endl;
                cout<<"Poioj:\n"<<Poioj.matrix()<<endl;

                inst.state[0] = new_pose;
                inst.SetWindowPose(e);*/
    }
}



void InstanceManager::SlideWindow()
{
    if(tracking_number_ < 1)
        return;

    if(e_->frame != kWinSize)
        return;
    Infov("动态特征边缘化:");
    //printf("动态特征边缘化:");
    if(e_->margin_flag == MarginFlag::kMarginOld)
        Infov("最老帧 | ");
    else
        Infov("次新帧 | ");


    for(auto &[key,inst] : instances){
        if(!inst.is_tracking)
            continue;
        ///测试
        /*
        if(inst.id==108091207){
            printf("Instance%d Landmarks Info:\n",inst.id);
            for(auto &landmark : inst.landmarks){
                printf("Landmark%d start_frame:%d feature_num:%d \n",landmark.id,landmark.start_frame,landmark.feats.size());
            }
        }*/
        /// 边缘化最老的帧
        int debug_num=0;
        if (e_->margin_flag == MarginFlag::kMarginOld)
            debug_num= inst.SlideWindowOld();
        /// 去掉次新帧
        else
            debug_num= inst.SlideWindowNew();

        ///当物体没有正在跟踪的特征点时，将其设置为不在跟踪状态
        if(inst.landmarks.empty()){
            inst.is_tracking=false;
            tracking_number_--;
        }
        ///
        if(inst.is_tracking){
            int tri_num=0;
            for(auto& lm: inst.landmarks){
                if(lm.depth>0){
                    tri_num++;
                }
            }
            if(tri_num==0){
                inst.is_initial=false;
            }

        }

        if(debug_num>0){
            Debugv("<Inst:{},del:{}> ", inst.id, debug_num);
        }
    }
}




/**
 * 添加动态物体的特征点
 * @param frame_id
 * @param instance_id
 * @param input_insts
 */
void InstanceManager::PushBack(unsigned int frame_id, InstancesFeatureMap &input_insts)
{
    if(e_->solver_flag == SolverFlag::kInitial)
        return;
    Debugv("PushBack | push_back current insts size:{}", instances.size());

    for(auto &[instance_id , inst_feat] : input_insts){
        ///创建物体
        auto inst_iter = instances.find(instance_id);
        if(inst_iter == instances.end()){
            Instance new_inst(frame_id, instance_id, e_);
            auto [it,is_insert] = instances.insert({instance_id, new_inst});
            it->second.is_initial=false;
            it->second.color = inst_feat.color;
            tracking_number_++;

            for(auto &[feat_id,feat_vector] : inst_feat){
                FeaturePoint featPoint(feat_vector, e_->frame, e_->td);
                LandmarkPoint landmarkPoint(feat_id);//创建Landmark
                it->second.landmarks.push_back(landmarkPoint);
                it->second.landmarks.back().feats.emplace_back(featPoint);//添加第一个观测
            }
            Debugv("PushBack | create new inst:{}", instance_id);
        }
        ///将特征添加到物体中
        else
        {
            auto &landmarks = inst_iter->second.landmarks;
            for(auto &[feat_id,feat_vector] : inst_feat)
            {
                FeaturePoint feat_point(feat_vector, e_->frame, e_->td);
                //若不存在，则创建路标
                if (auto it = std::find_if(landmarks.begin(),landmarks.end(),
                                           [id=feat_id](const LandmarkPoint &it){ return it.id == id;});
                    it ==landmarks.end())
                {
                    landmarks.emplace_back(feat_id);//创建Landmarks
                    landmarks.back().feats.emplace_back(feat_point);//添加第一个观测
                    if(!inst_iter->second.is_tracking){
                        inst_iter->second.is_tracking=true;
                        tracking_number_++;
                    }
                }
                //如果路标存在
                else{
                    it->feats.emplace_back(feat_point);//添加一个观测
                }
            }

        }
    }

    ///输出的信息
    Debugv("push_back all insts:");
    for(const auto& [key,inst] : instances){
        if(inst.is_tracking)
            Debugv("inst:{} landmarks.size:{} ", key, inst.landmarks.size());
    }
    if(!input_insts.empty()){
        Infov("push_back 观测增加:{},{}",
              input_insts.size(),
              std::accumulate(input_insts.begin(), input_insts.end(), 0,
                              [](int sum, auto &p) { return sum + p.second.size(); }));
    }

}


void InstanceManager::GetOptimizationParameters()
{
    if(tracking_number_ < 1)
        return;
    Debugv("InstanceManager 优化前后的位姿对比:");
    for(auto &[key,inst] : instances){
        if(!inst.is_initial || !inst.is_tracking)
            continue;

        Debugv("Inst {}", inst.id);

        string lm_msg="优化前 Depth: ";
        int lm_cnt=0;
        for(auto &lm: inst.landmarks){
            if(lm.depth<=0)continue;
            lm_cnt++;
            if(lm_cnt%5==0) lm_msg += "\n";
            lm_msg += fmt::format("<lid:{},n:{},d:{:.2f}> ",lm.id,lm.feats.size(),lm.depth);
        }
        Debugv(lm_msg);

        string pose_msg;
        pose_msg += fmt::format("优化前 Speed: v:({}) a:({})\n", VecToStr(inst.vel.v), VecToStr(inst.vel.a));
        pose_msg +="优化前 Pose:";
        for(int i=0; i <= kWinSize; ++i){
            pose_msg+=fmt::format("{}:({}) ", i, VecToStr(inst.state[i].P));
            if(i==4)pose_msg+="\n";
        }
        pose_msg+="\n";

        inst.GetOptimizationParameters();

        pose_msg +="优化后 Pose:";
        for(int i=0; i <= kWinSize; ++i){
            pose_msg+=fmt::format("{}:({}) ", i, VecToStr(inst.state[i].P));
            if(i==4)pose_msg+="\n";
        }
        pose_msg +=fmt::format("\n优化后 Speed v:({}) a:({})", VecToStr(inst.vel.v), VecToStr(inst.vel.a));
        Debugv(pose_msg);

        lm_msg="优化后 Depth: ";
        lm_cnt=0;
        for(auto &lm: inst.landmarks){
            if(lm.depth<=0)continue;
            lm_cnt++;
            if(lm_cnt%5==0) lm_msg += "\n";
            lm_msg += fmt::format("<lid:{},n:{},d:{:.2f}>",lm.id,lm.feats.size(),lm.depth);
        }
        Debugv(lm_msg);
    }
}




/**
 * 设置需要特别参数化的优化变量
 * @param problem
 */
void InstanceManager::AddInstanceParameterBlock(ceres::Problem &problem)
{
    if(tracking_number_ < 1) return;
    for(auto &[key,inst] : instances){
        if(!inst.is_initial || !inst.is_tracking)continue;
        for(int i=0;i<=(int)e_->frame; i++){
            problem.AddParameterBlock(inst.para_state[i], kSizePose, new PoseLocalParameterization());
        }
    }
}


/**
 * 添加残差块
 * @param problem
 * @param loss_function
 */
void InstanceManager::AddResidualBlock(ceres::Problem &problem, ceres::LossFunction *loss_function)
{
    if(tracking_number_ < 1)
        return;
    Infov("添加Instance残差:");
    int res21=0,res22=0,res12=0;

    for(auto &[key,inst] : instances){
        if(!inst.is_initial || !inst.is_tracking)
            continue;
        if(inst.landmarks.size()<5)
            continue;

        string debug_msg;

        ///特征点太多，则优先使用观测次数多的100个特征点
        const int kMaxLandmarkNum=100;
        if(inst.landmarks.size() > kMaxLandmarkNum){
            inst.landmarks.sort([](const LandmarkPoint &lp1,const LandmarkPoint &lp2){
                return lp1.feats.size() > lp2.feats.size();});
        }

        int depth_index=-1;
        int number_speed=0;//速度约束的数量
        int track_3=0,track_2;

        ///根据当前约束的情况判断是否优化速度
        for(auto& ldm : inst.landmarks){
            if(ldm.depth > 0){
                if(ldm.feats.size() >= 3) track_3++;
            }
        }
        /*if(track_3>=2)
            inst.opt_vel=true;
        else
            inst.opt_vel=false;*/
        inst.opt_vel=true;


        for(auto &lm : inst.landmarks)
        {
            if(depth_index >= kMaxLandmarkNum)
                break;
            if(lm.depth < 0.2)
                continue;
            depth_index++;

            auto feat_j=lm.feats[0];
            int fj=feat_j.frame;
            debug_msg += fmt::format("lid:{} depth:{} feats.size:{}\n", lm.id, lm.depth, lm.feats.size());

            ///第一个特征点只用来优化深度
            if(cfg::is_stereo && feat_j.is_stereo){
                problem.AddResidualBlock(
                        new ProjInst12Factor(feat_j.point,feat_j.point_right),
                        loss_function,
                        e_->para_ex_pose[0],
                        e_->para_ex_pose[1],
                        inst.para_inv_depth[depth_index]);
                /*problem.AddResidualBlock(
                        new ProjInst12FactorSimple(feat_j.point,feat_j.point_right,e->ric[0],e->tic[0],e->ric[1],e->tic[1]),
                        loss_function,
                        inst.para_inv_depth[depth_index]);*/
            }



            ///添加包围框残差
            /*problem.AddResidualBlock(
                    new ProjBoxSimpleFactor(
                            feat_j.point,feat_j.vel,feat_j.td,e->td,
                            inst.state[feat_j.frame].R,inst.state[feat_j.frame].P),
                            nullptr,
                            e->para_Pose[feat_j.frame],
                            e->para_ex_pose[0],
                            inst.para_box[0],
                            inst.para_inv_depth[depth_index]);*/
            /*auto* box_simple_cost=new BoxAbsFactor(feat_j.point,feat_j.vel,e->Rs[feat_j.frame],e->Ps[feat_j.frame],
                                                   e->ric[0],e->tic[0],feat_j.td,e->td);*/
            /*auto* box_simple_cost=new BoxSqrtFactor(feat_j.point,feat_j.vel,e->Rs[feat_j.frame],e->Ps[feat_j.frame],
                                                    e->ric[0],e->tic[0],feat_j.td,e->td);*/
            /*problem.AddResidualBlock(
                    new BoxPowFactor(
                            feat_j.point,feat_j.vel,e->Rs[feat_j.frame],e->Ps[feat_j.frame],
                            e->ric[0],e->tic[0],inst.state[feat_j.frame].R,
                            inst.state[feat_j.frame].P,feat_j.td,e->td),
                            nullptr,
                            inst.para_box[0],
                            inst.para_inv_depth[depth_index]);*/


            ///位姿、点云、速度的约束
            /*problem.AddResidualBlock(new InstanceInitPowFactor(
                            feat_j.point,feat_j.vel,e->Rs[fj],e->Ps[fj],
                            e->ric[0],e->tic[0],feat_j.td, e->td),
                            loss_function,
                            inst.para_state[fj],
                            inst.para_inv_depth[depth_index]);*/
            /*problem.AddResidualBlock(new InstanceInitPowFactorSpeed(
                    feat_j.point, feat_j.vel, e_->Rs[fj], e_->Ps[fj],
                    e_->ric[0], e_->tic[0], feat_j.td, e_->td,
                    e_->headers[fj], e_->headers[0], 1.0),
                                     loss_function,
                                     inst.para_state[0],
                                     inst.para_speed[0],
                                     inst.para_inv_depth[depth_index]);*/
            number_speed++;

            if(lm.feats.size() < 2)
                continue;
            string msg;

            ///有多个特征点,添加重投影残差
            for(int i=1;i<(int)lm.feats.size(); ++i){
                auto &feat_i=lm.feats[i];
                int fi = feat_i.frame;

                ///优化物体位姿和特征点深度
                /*problem.AddResidualBlock(
                        new ProjInst21SimpleFactor(
                                feat_j.point,feat_i.point,feat_j.vel,feat_i.vel,
                                e->Rs[feat_j.frame],e->Ps[feat_j.frame],
                                e->Rs[feat_i.frame],e->Ps[feat_i.frame],
                                e->ric[0],e->tic[0],
                                feat_j.td,feat_i.td,e->td,(int)lm.id),
                                loss_function,
                                inst.para_state[feat_j.frame],
                                inst.para_state[feat_i.frame],
                                inst.para_inv_depth[depth_index]);*/
                res21++;

                double factor= 1.;//track_3>=5 ? 5. : 1.;
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
                        feat_j.point, feat_i.point,feat_j.vel, feat_i.vel,
                        e->ric[0], e->tic[0],e->ric[0],e->tic[0],
                        feat_j.td, feat_i.td, e->td,e->Headers[fj],e->Headers[fi],factor),
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
                        feat_j.point, e_->headers[feat_j.frame], e_->headers[feat_i.frame],
                        e_->Rs[feat_j.frame], e_->Ps[feat_j.frame], e_->ric[0],
                        e_->tic[0],feat_j.vel, feat_j.td, e_->td),
                                         loss_function,
                                         inst.para_state[feat_j.frame],
                                         inst.para_state[feat_i.frame],
                                         inst.para_speed[0],
                                         inst.para_inv_depth[depth_index]);*/


                if(cfg::is_stereo && feat_i.is_stereo){
                    ///优化物体的位姿
                    /*problem.AddResidualBlock(
                            new ProjInst22SimpleFactor(
                                    feat_j.point,feat_i.point,feat_j.vel,feat_i.vel,
                                    e->Rs[feat_j.frame],e->Ps[feat_j.frame],
                                    e->Rs[feat_i.frame],e->Ps[feat_i.frame],
                                    e->ric[0],e->tic[0],
                                    e->ric[1],e->tic[1],
                                    feat_j.td,feat_i.td,e->td,lm.id),
                                    loss_function,
                                    inst.para_state[feat_j.frame],
                                    inst.para_state[feat_i.frame],
                                    inst.para_inv_depth[depth_index]);*/
                    res22++;
                    if(lm.feats.size() >= 3){
                        ///优化速度和特征点深度
                        /*problem.AddResidualBlock(new ProjectionSpeedFactor(
                                feat_j.point, feat_i.point_right,feat_j.vel, feat_i.vel_right,
                                e->ric[0], e->tic[0],e->ric[1],e->tic[1],
                                feat_j.td, feat_i.td, e->td,e->Headers[fj],e->Headers[fi],factor),
                                                 loss_function,
                                                 e->para_Pose[fj],
                                                 e->para_Pose[fi],
                                                 inst.para_speed[0],
                                                 inst.para_inv_depth[depth_index]);*/
                        ///优化相机位姿和物体速度
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
            }
        }

        ///添加恒定速度误差
        /*problem.AddResidualBlock(new ConstSpeedSimpleFactor(
                inst.last_vel.v,inst.last_vel.a,number_speed*10),
                nullptr,
                inst.para_speed[0]);*/


        Debugv("inst:{} landmarks.size:{} isOptimizeVel:{} track_3:{} \n {}", inst.id, inst.landmarks.size(),
               inst.opt_vel, track_3, debug_msg);

    }
    Debugv("残差项数量: res21:{},res22:{},res12:{}", res21, res22, res12);
}


}