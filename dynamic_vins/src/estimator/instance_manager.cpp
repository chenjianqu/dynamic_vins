/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "instance_manager.h"

#include <algorithm>

#include "estimator.h"
 #include "utils/def.h"
#include "vio_parameters.h"

#include "factor/pose_local_parameterization.h"
#include "factor/instance_factor.h"
#include "factor/speed_factor.h"
#include "factor/box_factor.h"

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
    Debugv("PushBack 输入的跟踪实例的数量和特征数量:{},{}",
           input_insts.size(),std::accumulate(input_insts.begin(), input_insts.end(), 0,
                                              [](int sum, auto &p) { return sum + p.second.size(); }));
    string log_text;


    auto getCamPose=[this](int index,int cam_id){
        assert(cam_id == 0 || cam_id == 1);
        Mat34d pose;
        Vec3d t0 = e->Ps[index] + e->Rs[index] * e->tic[cam_id];
        Mat3d R0 = e->Rs[index] * e->ric[cam_id];
        pose.leftCols<3>() = R0.transpose();
        pose.rightCols<1>() = -R0.transpose() * t0;
        return pose;
    };

    for(auto &[instance_id , inst_feat] : input_insts){
        log_text += fmt::format("PushBack 跟踪实例id:{} 特征点数:{}\n",instance_id,inst_feat.size());
        ///创建物体
        auto inst_iter = instances.find(instance_id);
        if(inst_iter == instances.end()){
            Instance new_inst(frame, instance_id, e);
            auto [it,is_insert] = instances.insert({instance_id, new_inst});
            it->second.is_initial=false;
            it->second.color = inst_feat.color;
            it->second.boxes3d[frame] = inst_feat.box3d;
            tracking_number_++;

            for(auto &[feat_id,feat_vector] : inst_feat){
                LandmarkPoint lm(feat_id);//创建Landmark

                FeaturePoint feat(feat_vector, e->frame, e->td);
                ///对双目的观测进行三角化,用于后面的场景流计算
                if(feat.is_stereo){
                    auto leftPose = getCamPose(e->frame,0);
                    auto rightPose= getCamPose(e->frame,1);
                    Vec2d point0 = feat.point.head(2);
                    Vec2d point1 = feat.point_right.head(2);
                    Eigen::Vector3d point3d_w;
                    TriangulatePoint(leftPose, rightPose, point0, point1, point3d_w);
                    Eigen::Vector3d localPoint = leftPose.leftCols<3>() * point3d_w + leftPose.rightCols<1>();//变换到相机坐标系下
                    double depth = localPoint.z();
                    if (depth > kDynamicDepthMin && depth<kDynamicDepthMax){//如果深度有效
                        feat.is_triangulated = true;
                        feat.p_w = point3d_w;
                        //顺便把landmark也初始化了
                        lm.depth = depth;
                        //log_text += fmt::format("stereo feat_id:{} p_w:{}\n", feat_id, VecToStr(point3d_w));
                    }
                }
                lm.feats.push_back(feat);//添加第一个观测
                it->second.landmarks.push_back(lm);
            }
            log_text += fmt::format("PushBack | 创建实例:{}\n", instance_id);
        }
        ///将特征添加到物体中
        else
        {
            auto &landmarks = inst_iter->second.landmarks;
            inst_iter->second.boxes3d[frame] = inst_feat.box3d;

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

                FeaturePoint feat(feat_vector, e->frame, e->td);
                //双目三角化
                if(feat.is_stereo){
                    auto leftPose = getCamPose(e->frame,0);
                    auto rightPose= getCamPose(e->frame,1);
                    Vec2d point0 = feat.point.head(2);
                    Vec2d point1 = feat.point_right.head(2);
                    Eigen::Vector3d point3d_w;
                    TriangulatePoint(leftPose, rightPose, point0, point1, point3d_w);
                    Eigen::Vector3d localPoint = leftPose.leftCols<3>() * point3d_w + leftPose.rightCols<1>();//变换到相机坐标系下
                    double depth = localPoint.z();
                    if (depth > kDynamicDepthMin && depth<kDynamicDepthMax){//如果深度有效
                        feat.is_triangulated = true;
                        feat.p_w = point3d_w;
                        //这里的情况更加复杂一些,只有当该路标点未初始化时,才会进行初始化
                        if(it->depth <=0 ){
                            //将该路标点的观测清空
                            it->feats.clear();
                            it->depth = depth;
                        }
                        //log_text += fmt::format("stereo feat_id:{} p_w:{}\n", feat_id, VecToStr(point3d_w));
                    }
                }
                //添加观测
                it->feats.push_back(feat);
            }

        }
    }

    Debugv(log_text);

}



/**
 * 三角化动态特征点,并限制三角化的点的数量在50以内
 * @param frame_cnt
 */
void InstanceManager::Triangulate(int frame_cnt)
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

    string log_text = "InstanceManager::Triangulate:";

    int num_triangle=0,num_failed=0,num_delete_landmark=0,num_mono=0;
    for(auto &[key,inst] : instances){
        if(!inst.is_tracking )
            continue;

        string log_inst_text;
        int inst_add_num=0;
        //double avg_depth= inst.AverageDepth();//在当前帧平均深度

        for(auto it=inst.landmarks.begin(),it_next=it;it!=inst.landmarks.end();it=it_next){
            it_next++;
            auto &lm=*it;
            if(lm.depth > 0 || lm.feats.empty())
                continue;
            int imu_i = lm.feats.front().frame;
            Eigen::Vector3d point3d_w;
            ///双目三角化
            if(cfg::is_stereo && lm.feats.front().is_stereo){
                //由于在 PushBack() 函数中已经三角化过了,因此无需重复三角化
                auto leftPose = getCamPose(imu_i,0);
                auto rightPose= getCamPose(imu_i,1);
                Vec2d point0 = lm.feats.front().point.head(2);
                Vec2d point1 = lm.feats.front().point_right.head(2);
                TriangulatePoint(leftPose, rightPose, point0, point1, point3d_w);
            }
            ///单目三角化，以开始帧和开始帧的后一帧的观测作为三角化的点
            else if(lm.feats.size() >= 2){
                Mat34d leftPose = getCamPose(imu_i,0);
                auto feat_j = (++lm.feats.begin());
                int imu_j=feat_j->frame;
                Mat34d rightPose = getCamPose(imu_j,0);
                Vec2d point0 = lm.feats.front().point.head(2);
                Vec2d point1 = feat_j->point.head(2);
                TriangulateDynamicPoint(leftPose, rightPose, point0, point1,
                                        inst.state[imu_i].R, inst.state[imu_i].P,
                                        inst.state[imu_j].R, inst.state[imu_j].P, point3d_w);
            }

            //将点投影当前帧
            Vec3d pts_cj = e->ric[0].transpose() * (e->Rs[e->frame].transpose() * (point3d_w - e->Ps[e->frame]) - e->tic[0]);
            double depth = pts_cj.z();

            if (depth > kDynamicDepthMin && depth<kDynamicDepthMax){ //判断深度值是否符合
                if(!inst.is_initial || inst.triangle_num < 5){ //若未初始化或路标点太少，则加入
                    lm.depth = depth;
                    inst_add_num++;
                    log_inst_text+=fmt::format("lid:{} NotInit d:{:.2f} p:{}\n", lm.id, depth, VecToStr(point3d_w));
                }
                else{ //判断是否在包围框附近
                    Eigen::Vector3d pts_obj_j=inst.state[imu_i].R.transpose() * (point3d_w - inst.state[imu_i].P);
                    if(pts_obj_j.norm() < inst.box.norm()*4){
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

    ///计算三角化点的数量和总的深度
    for(auto &[key,inst] : instances){
        if(!inst.is_tracking)
            continue;
        inst.triangle_num = 0;
        for(auto &lm:inst.landmarks){
            if(lm.depth>0){
                inst.triangle_num++;
            }
        }
    }

    ///管理路标点
    for(auto &[key,inst] : instances){
        if(!inst.is_tracking)
            continue;
        if(inst.triangle_num < 50)
            continue;

        for(auto it=inst.landmarks.begin(),it_next=it;it!=inst.landmarks.end();it=it_next){
            it_next++;
            if(it->feats.size()<=1 && it->feats.front().frame < e->frame){ //只有一个观测,且该观测不在当前帧
                if(it->depth>0){
                    inst.triangle_num--;
                }
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

    int i=e->frame;
    int j= e->frame - 1;
    double time_ij= e->headers[i] - e->headers[j];

    InstExec([&](int key,Instance& inst){
        inst.state[i].time = e->headers[i];

        inst.state[i].R = inst.state[j].R;
        inst.state[i].P = inst.state[j].P;

        /*if(inst.is_static){
            inst.state[i].R = inst.state[j].R;
            inst.state[i].P = inst.state[j].P;
        }
        else{
            Mat3d Roioj=Sophus::SO3d::exp(inst.vel.a*time_ij).matrix();
            Vec3d Poioj=inst.vel.v*time_ij;
            inst.state[i].R=Roioj * inst.state[j].R;
            inst.state[i].P=Roioj* inst.state[j].P + Poioj;
        }*/
    },true);

}



void InstanceManager::SlideWindow()
{
    if(e->frame != kWinSize)
        return;

    if(e->margin_flag == MarginFlag::kMarginOld)
        Debugv("InstanceManager::SlideWindow margin_flag = kMarginOld");
    else
        Debugv("InstanceManager::SlideWindow margin_flag = kMarginSecondNew | ");

    for(auto &[key,inst] : instances){
        if(!inst.is_tracking)
            continue;

        int debug_num=0;
        if (e->margin_flag == MarginFlag::kMarginOld)/// 边缘化最老的帧
            debug_num= inst.SlideWindowOld();
        else/// 去掉次新帧
            debug_num= inst.SlideWindowNew();

        if(debug_num>0){
            Debugv("InstanceManager::SlideWindow Inst:{},del:{} ", inst.id, debug_num);
        }

        ///当物体没有正在跟踪的特征点时，将其设置为不在跟踪状态
        if(inst.landmarks.empty()){
            inst.ClearState();
            tracking_number_--;
        }
        else if(inst.is_tracking && inst.triangle_num==0){
            inst.is_initial=false;
        }

    }
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
            inst.vel.v = avg_t/cnt_t;
            inst.is_init_velocity = false;
        }

    }
}


/**
* 进行物体的位姿初始化
*/
void InstanceManager::InitialInstance(std::map<unsigned int,FeatureInstance> &input_insts){

    for(auto &[inst_id,inst] : instances){
        if(inst.is_initial || !inst.is_tracking)
            continue;
        if(!inst.boxes3d[frame]){ //未检测到box3d
            continue;
        }

        ///寻找当前帧三角化的路标点
        Vec3d center_point=Vec3d::Zero();
        int cnt=0;
        for(auto &lm : inst.landmarks){
            if(lm.feats.empty() || lm.depth<=0)
                continue;
            if(lm.feats.front().frame == e->frame){
                center_point += (lm.feats.front().point * lm.depth);
                cnt++;
            }
        }

        if(cnt <= para::kInstanceInitMinNum){ //路标数量太少了
            return;
        }

        ///根据box初始化物体的位姿和包围框

        auto cam_to_world = [this](const Vec3d &p){
            return e->Rs[e->frame] * (e->ric[0] * p + e->tic[0]) + e->Ps[e->frame];
        };

        State init_state;

        //首先将box中心点从相机坐标系转换到世界坐标系
        Vec3d avg_center_point = center_point / double(cnt);
        //Debugv("center_point:{} cnt:{}", VecToStr(avg_center_point),cnt);
        init_state.P = cam_to_world(avg_center_point);

        //将包围框的8个顶点转换到世界坐标系下
        Mat38d corners;
        for(int i=0;i<8;++i){
            corners.col(i) = cam_to_world( inst.boxes3d[frame]->corners.col(i));
        }
        //在box中构建坐标系
        init_state.R = Box3D::GetCoordinateRotationFromCorners(corners);

        inst.state[e->frame].time=e->headers[0];
        inst.vel.SetZero();

        //设置滑动窗口内所有时刻的位姿
        for(int i=0; i <= kWinSize; i++){
            inst.state[i] = init_state;
            inst.state[i].time = e->headers[i];
        }

        inst.box = inst.boxes3d[frame]->dims;
        inst.is_initial=true;

        Debugv("Initialized id:{},cnt_max:{},初始位姿:P:{},R:{} 初始box:{}",
               inst.id, cnt, VecToStr(init_state.P), VecToStr(init_state.R.eulerAngles(2,1,0)),VecToStr(inst.box));

        ///删去初始化之前的观测
        for(auto it=inst.landmarks.begin(),it_next=it;it!=inst.landmarks.end();it=it_next){
            it_next++;
            if(it->feats.size() == 1 && it->feats.front().frame < e->frame){
                inst.landmarks.erase(it);
            }
        }
        for(auto &lm:inst.landmarks){
            if(lm.feats.front().frame < e->frame){
                for(auto it=lm.feats.begin(),it_next=it; it != lm.feats.end(); it=it_next){
                    it_next++;
                    if(it->frame < e->frame) {
                        lm.feats.erase(it); ///删掉掉前面的观测
                    }
                }
                if(lm.depth > 0){
                    lm.depth=-1.0;///需要重新进行三角化
                }
            }
        }


    }


}

string InstanceManager::PrintInstanceInfo(bool output_lm,bool output_stereo){
    if(tracking_number_<1){
        return {};
    }
    string s="InstanceInfo :\n";
    InstExec([&s,&output_lm,&output_stereo](int key,Instance& inst){
        if(inst.is_tracking){
            s+= fmt::format("id:{} landmarks:{} is_init:{} is_tracking:{} is_static:{} triangle_num:{} avg_depth:{}\n",
                            inst.id,inst.landmarks.size(),inst.is_initial,inst.is_tracking,inst.is_static,inst.triangle_num,inst.AverageDepth());
            if(!output_lm)
                return;
            for(int i=0;i<=kWinSize;++i){
                if(inst.boxes3d[i]){
                    s+=fmt::format("{} box:{} dims:{}\n",i, VecToStr(inst.boxes3d[i]->center), VecToStr(inst.boxes3d[i]->dims));
                }
            }
            for(auto &landmark : inst.landmarks){
                if(landmark.depth>0){
                    s+=fmt::format("lid:{} feat_size:{} depth:{} start:{} \n",landmark.id,landmark.feats.size(),
                                   landmark.depth,landmark.feats.front().frame);
                    if(output_stereo){
                        int cnt=0;
                        for(auto &feat:landmark.feats){
                            s+=fmt::format("{} ", VecToStr(feat.p_w));
                            cnt++;
                        }
                        if(cnt>0){
                            s+= "\n";
                        }
                    }
                }
            }
        }

        },true);
    return s;
}

string InstanceManager::PrintInstancePoseInfo(bool output_lm){
    string log_text ;
    InstExec([&log_text,&output_lm](int key,Instance& inst){
        log_text += fmt::format("id:{} info:\n box:{} v:{} a:{}\n",inst.id,VecToStr(inst.box),VecToStr(inst.vel.v),VecToStr(inst.vel.a));
        for(int i=0; i <= kWinSize; ++i){
            log_text+=fmt::format("{},P:({}),R:({})\n", i, VecToStr(inst.state[i].P),VecToStr(inst.state[i].R.eulerAngles(2,1,0)));
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
    });
    return log_text;
}


void InstanceManager::SetVelMap(){
    std::unique_lock<std::mutex> lk(vel_mutex_);
    vel_map_.clear();
    if(tracking_number_<1){
        return;
    }
    string log_text = "SetVelMap 物体的速度信息:";
    InstExec([&log_text,this](int key,Instance& inst){
        log_text += fmt::format("inst:{} v:{} a:{}", inst.id, VecToStr(inst.vel.v), VecToStr(inst.vel.a));
        if(inst.vel.v.norm() > 0)
            vel_map_.insert({inst.id, inst.vel});;
    });
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

        for(int i=0;i<=(int)e->frame; i++){
            problem.AddParameterBlock(inst.para_state[i], kSizePose, new PoseLocalParameterization());
        }
    });
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

    std::unordered_map<string,int> statistics=
            {{"BoxEncloseStereoPointFactor",0},{"BoxEncloseTrianglePointFactor",0},
             {"ProjInst12Factor",0},{"ConstSpeedSimpleFactor",0},{"BoxVertexFactor",0},
             {"BoxDimsFactor",0},{"BoxOrientationFactor",0} ,{"SpeedPoseFactor",0},
             {"SpeedStereoPointFactor",0},{"ConstSpeedStereoPointFactor",0}};



    Debugv("添加Instance残差:");

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

        inst.opt_vel=true;

        int depth_index=-1;//注意,这里的depth_index的赋值过程要与 Instance::SetOptimizeParameters()中depth_index的赋值过程一致
        for(auto &lm : inst.landmarks){
            if(depth_index >= kMaxLandmarkNum)
                break;
            if(lm.depth < 0.2)
                continue;
            depth_index++;

            auto &feat_j=lm.feats.front();
            int fj=feat_j.frame;
            //debug_msg += fmt::format("lid:{} depth:{} feats.size:{}\n", lm.id, lm.depth, lm.feats.size());

            ///第一个特征点只用来优化深度
            /*if(cfg::is_stereo && feat_j.is_stereo){
                problem.AddResidualBlock(
                        new ProjInst12Factor(feat_j.point,feat_j.point_right),
                        loss_function,
                        e->para_ex_pose[0],
                        e->para_ex_pose[1],
                        inst.para_inv_depth[depth_index]);
                *//*problem.AddResidualBlock(
                        new ProjInst12FactorSimple(feat_j.point,feat_j.point_right,e->ric[0],e->tic[0],e->ric[1],e->tic[1]),
                        loss_function,
                        inst.para_inv_depth[depth_index]);*//*
            }*/

                        
           ///根据3D应该要落在包围框内产生的误差
            /*for(auto &feat : lm.feats){
                if(feat.is_triangulated){
                    //Debugv("lm:{} frame:{} p_w:{}",lm.id,feat.frame, VecToStr(feat.p_w));
                    problem.AddResidualBlock(new BoxEncloseStereoPointFactor(feat.p_w),
                                             loss_function,
                                             inst.para_state[feat.frame],inst.para_box[0]);
                    statistics["BoxEncloseStereoPointFactor"]++;
                }
            }
            problem.AddResidualBlock(new BoxEncloseTrianglePointFactor(
                    feat_j.point,feat_j.vel,e->Rs[feat_j.frame],e->Ps[feat_j.frame], e->ric[0],e->tic[0],feat_j.td,e->td),
                                     loss_function,
                                     inst.para_state[feat_j.frame],inst.para_box[0],inst.para_inv_depth[depth_index]);
            statistics["BoxEncloseTrianglePointFactor"]++;*/

            if(inst.is_static){ //对于静态物体,仅仅refine包围框
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

                ///优化物体位姿和特征点深度
                /* //这一项效果不好, TODO
                 problem.AddResidualBlock(
                        new ProjInst21SimpleFactor(
                                feat_j.point,feat_i.point,feat_j.vel,feat_i.vel,
                                e->Rs[feat_j.frame],e->Ps[feat_j.frame],
                                e->Rs[feat_i.frame],e->Ps[feat_i.frame],
                                e->ric[0],e->tic[0],
                                feat_j.td,feat_i.td,e->td,(int)lm.id),
                                loss_function,
                                inst.para_state[feat_j.frame],
                                inst.para_state[feat_i.frame],
                                inst.para_inv_depth[depth_index]);
                res21++;*/

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
                    /* //这一项效果不好,TODO
                     problem.AddResidualBlock(
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

            if(inst.is_init_velocity){
                ///优化物体速度
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
                if(found_first && found_end){
                    double time_ij = e->headers[end_point->frame] - e->headers[first_point->frame];
                    problem.AddResidualBlock(
                            new SpeedStereoPointFactor(first_point->p_w,end_point->p_w,time_ij),
                            loss_function,inst.para_speed[0]);
                    statistics["SpeedStereoPointFactor"]++;

                    problem.AddResidualBlock(
                            new ConstSpeedStereoPointFactor(first_point->p_w,end_point->p_w,time_ij,
                                                            inst.last_vel.v,inst.last_vel.a),
                            loss_function,inst.para_speed[0]);
                    statistics["ConstSpeedStereoPointFactor"]++;

                }

            }


        }

        if(inst.is_init_velocity){
        //if(!inst.is_static){
            ///添加恒定速度误差
            /*problem.AddResidualBlock(new ConstSpeedSimpleFactor(
                    inst.last_vel.v,inst.last_vel.a,number_speed*10),
                                     nullptr,
                                     inst.para_speed[0]);
            statistics["ConstSpeedSimpleFactor"]++;*/

            ///速度-位姿误差
            for(int i=1;i<=kWinSize;++i){
                //problem.AddResidualBlock(new SpeedPoseFactor(inst.state[i-1].time,inst.state[i].time),
                //                         loss_function,inst.para_state[i-1],inst.para_state[i],inst.para_speed[0]);
                //statistics["SpeedPoseFactor"]++;
            }

        //}
        }

        ///添加包围框预测误差
        for(int i=0;i<=kWinSize;++i){
            if(inst.boxes3d[i]){
                ///包围框大小误差
/*                problem.AddResidualBlock(new BoxDimsFactor(inst.boxes3d[i]->dims),loss_function,inst.para_box[0]);
                statistics["BoxDimsFactor"]++;
                ///物体的方向误差
                Mat3d R_cioi = Box3D::GetCoordinateRotationFromCorners(inst.boxes3d[i]->corners);
                problem.AddResidualBlock(new BoxOrientationFactor(R_cioi,e->ric[0]), nullptr,e->para_Pose[i],inst.para_state[i]);
                statistics["BoxOrientationFactor"]++;*/

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


    }

    //log
    string log_text;
    for(auto &pair : statistics)  if(pair.second>0) log_text += fmt::format("{} : {}\n",pair.first,pair.second);
    Debugv("各个残差项的数量: \n{}",log_text);
}


}