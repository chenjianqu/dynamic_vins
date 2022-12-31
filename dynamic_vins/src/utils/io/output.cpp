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

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

#include "utils/io/io_parameters.h"
#include "utils/file_utils.h"
#include "utils/convert_utils.h"
#include "utils/dataset/kitti_utils.h"
#include "estimator/feature_manager.h"

namespace dynamic_vins{ \


/**
 * 输出Debug信息
 * @param im
 * @return
 */
string PrintFactorDebugMsg(InstanceManager& im){
    string log_text=fmt::format("***********PrintFactorDebugMsg:{}***************\n",body.frame_time);

    for(auto &[key,inst]:im.instances){
        if(key==1){
            log_text += fmt::format("Time:{} inst:{} P_woj:{} \n",
                                    body.frame_time,key,
                                    VecToStr(inst.state[body.frame].P));
            log_text += fmt::format("dims:{} \n", VecToStr(inst.box3d->dims));

            for(auto &lm:inst.landmarks){
                if(lm.bad || lm.depth<=0)
                    continue;
                ///误差计算
                Vec3d pts_obj_j=inst.state[body.frame].R.transpose() * (lm.front()->p_w - inst.state[body.frame].P);
                Vec3d abs_v=pts_obj_j.cwiseAbs();
                Vec3d vec_err = abs_v - inst.box3d->dims/2;
                vec_err *=10;
                log_text += fmt::format("p_w:{} abs_v:{} vec_err:{}\n",
                                        VecToStr(lm.front()->p_w), VecToStr(abs_v),VecToStr(vec_err));
            }
        }

    }

    WriteTextFile(MyLogger::kLogOutputDir + "factor_debug.txt",log_text);

    return {};
}


string PrintFeaturesInfo(InstanceManager& im, bool output_lm, bool output_stereo){
    if(im.tracking_number()<1){
        return {};
    }

    string s =fmt::format("--------------InstanceInfo : {} --------------\n",body.headers[body.frame]);
    im.InstExec([&s,&output_lm,&output_stereo](int key,Instance& inst){
        if(io_para::inst_ids_print.count(key)==0)
            return;

        if(inst.is_tracking){
            s+= fmt::format("Time:{} inst:{} landmarks:{} is_init:{} is_tracking:{} is_static:{} "
                            "is_init_v:{} triangle_num:{} curr_avg_depth:{}\n",
                            body.frame_time,inst.id,inst.landmarks.size(),inst.is_initial,inst.is_tracking,
                            inst.is_static,inst.is_init_velocity,inst.triangle_num,inst.AverageDepth());
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
    WriteTextFile(MyLogger::kLogOutputDir + "features_info.txt",s);
    return {};
}





string PrintInstancePoseInfo(InstanceManager& im,bool output_lm){
    string log_text =fmt::format("--------------PrintInstancePoseInfo : {} --------------\n",body.headers[body.frame]);

    im.InstExec([&log_text,&output_lm](int key,Instance& inst){
        if(!inst.is_tracking || !inst.is_curr_visible){
            return;
        }
        if(io_para::inst_ids_print.count(key)==0)
            return;

        log_text += fmt::format("Time:{} inst_id:{} info:\n box:{} v:{} a:{}\n",body.frame_time,inst.id,
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

    },true);

    ///将这些信息保存到文件中
    WriteTextFile(MyLogger::kLogOutputDir + "pose_info.txt",log_text);

    return {};
}



string PrintLineInfo(FeatureManager &fm){

    string log_text =fmt::format("--------------PrintLineInfo : {} --------------\n",body.headers[body.frame]);

    for(auto &line:fm.line_landmarks){
        if(!line.is_triangulation){
            continue;
        }
        log_text += fmt::format("lid:{},pw1:{},pw2:{}\n",line.feature_id, VecToStr(line.ptw1),VecToStr(line.ptw2));
    }
    log_text+="\n \n";

    WriteTextFile(MyLogger::kLogOutputDir + "line_info.txt",log_text);
    return {};
}


void SaveBodyTrajectory(const std_msgs::Header &header){

    static bool is_first_run=true;
    if(is_first_run){
        is_first_run=false;
        //清空
        std::ofstream fout(io_para::kVinsResultPath, std::ios::out);
        fout.close();
    }

    std::ofstream fout(io_para::kVinsResultPath, std::ios::app);
    fout.setf(std::ios::fixed, std::ios::floatfield);

    Quaterniond tmp_Q (body.Rs[kWinSize]);

    /*
    fout.precision(0);
    fout << header.stamp.toSec() * 1e9 << ",";
    fout.precision(5);
    fout << e.Ps[kWindowSize].x() << ","
          << e.Ps[kWindowSize].y() << ","
          << e.Ps[kWindowSize].z() << ","
          << tmp_Q.w() << ","
          << tmp_Q.x() << ","
          << tmp_Q.y() << ","
          << tmp_Q.z() << ","
          << e.Vs[kWindowSize].x() << ","
          << e.Vs[kWindowSize].y() << ","
          << e.Vs[kWindowSize].z() << endl;
        */
    fout << header.stamp << " "
    << body.Ps[kWinSize].x() << " "
    << body.Ps[kWinSize].y() << " "
    << body.Ps[kWinSize].z() << " "
    <<tmp_Q.x()<<" "
    <<tmp_Q.y()<<" "
    <<tmp_Q.z()<<" "
    <<tmp_Q.w()<<endl;
    fout.close();

    Eigen::Vector3d tmp_T = body.Ps[kWinSize];
    printf("time: %f, t: %f %f %f q: %f %f %f %f \n", header.stamp.toSec(),
           tmp_T.x(), tmp_T.y(), tmp_T.z(),
           tmp_Q.w(), tmp_Q.x(), tmp_Q.y(), tmp_Q.z());
}


/**
 * 保存物体的点云到磁盘
 * @param im
 */
void SaveInstancesPointCloud(InstanceManager& im){
    if(cfg::slam != SLAM::kDynamic){
        return;
    }
    const string object_base_path = io_para::kOutputFolder + cfg::kDatasetSequence + "/point_cloud/";
    static bool first_run=true;
    if(first_run){
        ClearDirectory(object_base_path);//获取目录中所有的路径,并将这些文件全部删除
        first_run=false;
    }

    for(auto &[inst_id,inst] : im.instances){
        if(!inst.is_curr_visible)
            continue;
        //if(!inst.is_tracking || !inst.is_initial)
        //    continue;

        if(io_para::inst_ids_print.count(inst_id)==0)
            continue;

        if(!inst.points_extra[body.frame].empty()){
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr stereo_pc =
                    EigenToPclXYZRGB(inst.points_extra[body.frame],inst.color);

            const string save_path = object_base_path+
                    fmt::format("{}_{}.pcd",PadNumber(body.seq_id,6),inst_id);
            pcl::io::savePCDFile(save_path,*stereo_pc);
        }

    }


}



/**
 * 保存所有物体在当前帧的位姿
 */
void SaveInstancesTrajectory(InstanceManager& im){
    if(cfg::slam != SLAM::kDynamic){
        return;
    }
    ///文件夹设置
    const string object_base_path = io_para::kOutputFolder + cfg::kDatasetSequence + "/";
    const string object_tracking_path = object_base_path+cfg::kDatasetSequence+".txt";
    const string object_object_dir= object_base_path + cfg::kDatasetSequence+"/";
    const string object_tum_dir=object_base_path + cfg::kDatasetSequence+"_tum/";

    static bool first_run=true;
    if(first_run){
        ClearDirectory(object_object_dir);//获取目录中所有的路径,并将这些文件全部删除
        ClearDirectory(object_tum_dir);

        std::ofstream fout(object_tracking_path, std::ios::out);
        fout.close();

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

    string log_text="InstanceManager::SaveInstancesTrajectory()\n";

    Mat3d R_iw = body.Rs[body.frame].transpose();
    Vec3d P_iw = - R_iw * body.Ps[body.frame];
    Mat3d R_ci = body.ric[0].transpose();
    Vec3d P_ci = - R_ci * body.tic[0];

    Mat3d R_cw = R_ci * R_iw;
    Vec3d P_cw = R_ci * P_iw + P_ci;

    for(auto &[inst_id,inst] : im.instances){
        if(!inst.is_curr_visible)
            continue;
        log_text+=fmt::format("inst_id:{},is_tracking:{},is_initial:{},is_curr_visible:{},"
                              "is_init_velocity:{},is_static:{},landmarks:{},triangle_num:{}\n",
                              inst_id,inst.is_tracking,inst.is_initial,inst.is_curr_visible,inst.is_init_velocity,
                              inst.is_static,inst.landmarks.size(),inst.triangle_num);

        if(!inst.is_tracking || !inst.is_initial)
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
        else{
            std::cerr<<"SaveInstancesTrajectory() not is implemented, as dataset is "<<cfg::dataset_name<<endl;

        }

    }


    fout_mot.close();
    fout_object.close();

    Debugv(log_text);
}


/**
 * 将物体坐标系下的点投影到12mx12m的区域中,并绘制到图像中
 * @param im
 * @param size
 * @return
 */
cv::Mat DrawTopView(InstanceManager& im,cv::Size size){
    cv::Mat img(size,CV_8UC3,cv::Scalar(255,255,255));

    double half_size=size.width / 2.;
    double half_metric = 6.;

    for(auto &[key,inst] : im.instances){
        //绘制车辆包围框的矩形
        double x_max = inst.box3d->dims.x()/2.;
        x_max = x_max / half_metric * half_size + half_size;
        double x_min = -inst.box3d->dims.x()/2.;
        x_min = x_min / half_metric * half_size + half_size;
        double z_max = inst.box3d->dims.z()/2.;
        z_max = z_max / half_metric * half_size + half_size;
        double z_min = -inst.box3d->dims.z()/2.;
        z_min = z_min / half_metric * half_size + half_size;
        cv::Rect2i rect(cv::Point2i(x_max,z_max),cv::Point2i(x_min,z_min));
        cv::rectangle(img,rect, BgrColor("blue",false),2);

        //将点变换到物体坐标系
        for(auto &lm:inst.landmarks){
            if(!lm.bad && lm.depth>0){
                Vec3d pts_obj = inst.WorldToObject(lm.front()->p_w,lm.frame());

                double x = pts_obj.x() / half_metric * half_size + half_size;
                double z = pts_obj.z() / half_metric * half_size + half_size;

                cv::circle(img,cv::Point2d(x,z),2, BgrColor("red",false),-1);

            }

        }

    }


    return img;

}















}
