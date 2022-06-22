/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "output.h"

#include "utils/io/io_parameters.h"
#include "utils/io/io_utils.h"
#include "utils/dataset/kitti_utils.h"


namespace dynamic_vins{ \


string PrintInstanceInfo(InstanceManager& im,bool output_lm,bool output_stereo){
    if(im.tracking_number()<1){
        return {};
    }

    string s =fmt::format("--------------InstanceInfo : {} --------------\n",body.headers[body.frame]);
    im.InstExec([&s,&output_lm,&output_stereo](int key,Instance& inst){
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

            for(auto &lm : inst.landmarks){
                if(lm.bad)
                    continue;
                if(lm.depth > 0){
                    int triangle_num=0;
                    for(auto &feat:lm.feats){
                        if(feat->is_triangulated)
                            triangle_num++;
                    }

                    s+=fmt::format("lid:{} feat_size:{} depth:{} start:{} triangle_num:{}\n", lm.id, lm.feats.size(),
                                   lm.depth, lm.frame(), triangle_num);
                    for(auto &feat:lm.feats){
                        if(feat->is_stereo)
                            s+=fmt::format("S:{}|{} ", VecToStr(feat->point),VecToStr(feat->point_right));
                        else
                            s+=fmt::format("M:{} ", VecToStr(feat->point));
                    }
                    s+= "\n";
                    if(output_stereo){
                        int cnt=0;
                        for(auto &feat:lm.feats){
                            if(feat->is_triangulated){
                                s+=fmt::format("{}-{} ",feat->frame, VecToStr(feat->p_w));
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





string PrintInstancePoseInfo(InstanceManager& im,bool output_lm){
    string log_text =fmt::format("--------------PrintInstancePoseInfo : {} --------------\n",body.headers[body.frame]);

    im.InstExec([&log_text,&output_lm](int key,Instance& inst){
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
            if(lm.bad)
                continue;
            if(lm.depth<=0)continue;
            lm_cnt++;
            if(lm_cnt%5==0) log_text += "\n";
            log_text += fmt::format("<lid:{},n:{},d:{:.2f}> ",lm.id,lm.size(),lm.depth);
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
 * 保存所有物体在当前帧的位姿
 */
void SaveTrajectory(InstanceManager& im){
    if(cfg::slam != SlamType::kDynamic){
        return;
    }

    Mat3d R_iw = body.Rs[body.frame].transpose();
    Vec3d P_iw = - R_iw * body.Ps[body.frame];
    Mat3d R_ci = body.ric[0].transpose();
    Vec3d P_ci = - R_ci * body.tic[0];

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
        object_object_path = object_object_dir+DoubleToStr(body.frame_time,6)+".txt";
    }
    else{
        object_object_path = object_object_dir+ PadNumber(body.seq_id,6)+".txt";
    }
    std::ofstream fout_object(object_object_path, std::ios::out);

    std::ofstream fout_mot(object_tracking_path, std::ios::out | std::ios::app);//追加写入

    string log_text="InstanceManager::SaveTrajectory()\n";

    for(auto &[inst_id,inst] : im.instances){
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
        foutC << body.headers[kWinSize] << " "
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
            Mat3d R_coi = R_cw * inst.state[body.frame].R;
            Vec3d P_coi = R_cw * inst.state[body.frame].P + P_cw;
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
            Vec3d obj_z = R_coi_kitti * Vec3d(0,0,1);//车辆Z轴在相机坐标系下的方向
            Debugv("inst:{} obj_z:{}",inst.id, VecToStr(obj_z));
            Vec3d cam_x(1,0,0);//相机坐标系Z轴的向量
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
            fout_mot << body.seq_id << " " <<
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





}
