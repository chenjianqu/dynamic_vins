/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/



#include "visualization.h"

#include <string>

#include "utils/file_utils.h"
#include "utils/convert_utils.h"
#include "utils/io/markers_utils.h"
#include "estimator/estimator_insts.h"
#include "det3d/detector3d.h"
#include "utils/io/publisher_map.h"


namespace dynamic_vins{\

using namespace std;
using namespace ros;
using namespace Eigen;


std::set<int> last_marker_ids;//保存上一时刻发布的marker的id,用于将上一时刻的id清除


static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

CameraPoseVisualization::Ptr camera_pose_visual;

std::shared_ptr<tf::TransformBroadcaster> transform_broadcaster;

nav_msgs::Path path;


inline PointT PointPCL(const Vec3d &v,double r,double g,double b){
    PointT ps(r,g,b);
    ps.x =(float) v.x();
    ps.y =(float) v.y();
    ps.z =(float) v.z();
    return ps;
}


/**
 * 构造ROS发布器
 * @param n
 */
void Publisher::RegisterPub(ros::NodeHandle &n)
{
    transform_broadcaster = std::make_shared<tf::TransformBroadcaster>();

    PublisherMap pub_map(n);

    camera_pose_visual=std::make_shared<CameraPoseVisualization>(1, 0, 0, 1);
    if(cfg::dataset==DatasetType::kCustom){
        camera_pose_visual->setScale(0.05);
        camera_pose_visual->setLineWidth(0.01);
    }
    else{
        camera_pose_visual->setScale(1.);
        camera_pose_visual->setLineWidth(0.1);
    }

}

void Publisher::PubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t)
{
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(t);
    odometry.header.frame_id = "world";
    odometry.pose.pose.position = EigenToGeometryPoint(P);
    odometry.pose.pose.orientation = EigenToGeometryQuaternion(Q);
    odometry.twist.twist.linear = EigenToGeometryVector3(V);

    PublisherMap::Pub<nav_msgs::Odometry>(odometry,"imu_propagate");
    //Debugv("Publisher::pubLatestOdometry:{}", VecToStr(P));
}


void Publisher::PrintStatistics(double t)
{
    if (e->solver_flag != SolverFlag::kNonLinear)
        return;
    if (cfg::is_estimate_ex){
        cv::FileStorage fs(cfg::kExCalibResultPath, cv::FileStorage::WRITE);
        for (int i = 0; i < cfg::kCamNum; i++){
            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = body.ric[i];
            eigen_T.block<3, 1>(0, 3) = body.tic[i];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            if(i == 0)
                fs << "body_T_cam0" << cv_T ;
            else
                fs << "body_T_cam1" << cv_T ;
        }
        fs.release();
    }

    sum_of_path += (body.Ps[kWinSize] - last_path).norm();
    last_path = body.Ps[kWinSize];
}


/**
 * pub位姿,并将位姿输出到文件中
 * @param estimator
 * @param header
 */
void Publisher::PubOdometry(const std_msgs::Header &header)
{
    if(e->solver_flag != SolverFlag::kNonLinear)
        return;

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.child_frame_id = "world";
    Quaterniond tmp_Q;
    tmp_Q = Quaterniond(body.Rs[kWinSize]);
    odometry.pose.pose.position = EigenToGeometryPoint(body.Ps[kWinSize]);
    odometry.pose.pose.orientation = EigenToGeometryQuaternion(tmp_Q);
    odometry.twist.twist.linear = EigenToGeometryVector3(body.Vs[kWinSize]);
    PublisherMap::Pub<nav_msgs::Odometry>(odometry,"odometry");

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometry.pose.pose;
    path.header = header;
    path.header.frame_id = "world";
    path.poses.push_back(pose_stamped);

    PublisherMap::Pub<nav_msgs::Path>(path,"path");
}


void Publisher::PubKeyPoses(const std_msgs::Header &header)
{
    if (e->key_poses.empty())
        return;
    Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = Marker::SPHERE_LIST;
    key_poses.action = Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.1;
    key_poses.scale.y = 0.1;
    key_poses.scale.z = 0.1;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= kWinSize; i++){
        key_poses.points.push_back(EigenToGeometryPoint(e->key_poses[i]));
    }
    PublisherMap::Pub<Marker>(key_poses,"key_poses");
}


/**
 * 可视化相机的marker
 * @param header
 */
void Publisher::PubCameraPose(const std_msgs::Header &header)
{
    if(e->solver_flag != SolverFlag::kNonLinear)
        return;

    int idx2 = kWinSize - 1;
    int i = idx2;
    Vector3d P_eigen = body.Ps[i] + body.Rs[i] * body.tic[0];
    Quaterniond Q_eigen = Quaterniond(body.Rs[i] * body.ric[0]);

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position = EigenToGeometryPoint(P_eigen);
    odometry.pose.pose.orientation = EigenToGeometryQuaternion(Q_eigen);

    PublisherMap::Pub<nav_msgs::Odometry>(odometry,"camera_pose");

    camera_pose_visual->reset();
    camera_pose_visual->add_pose(P_eigen, Q_eigen);
    if(cfg::is_stereo){
        Vector3d P = body.Ps[i] + body.Rs[i] * body.tic[1];
        Quaterniond R = Quaterniond(body.Rs[i] * body.ric[1]);
        camera_pose_visual->add_pose(P, R);
    }

    ros::Publisher cam_pub = PublisherMap::GetPublisher<MarkerArray>("camera_pose_visual");
    camera_pose_visual->publish_by(cam_pub ,odometry.header);
}

/**
 * 输出点云和边缘化点
 * @param header
 */
void Publisher::PubPointCloud(const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;

    for (auto &lm : e->feat_manager.point_landmarks){
        int used_num = (int)lm.feats.size();
        if (!(used_num >= 2 && lm.start_frame < kWinSize - 2))
            continue;
        if (lm.start_frame > kWinSize * 3.0 / 4.0 || lm.solve_flag != 1)
            continue;
        int imu_i = lm.start_frame;
        point_cloud.points.push_back(EigenToGeometryPoint32(
                body.CamToWorld(lm.feats[0].point * lm.depth, imu_i)));
    }

    PublisherMap::PubPointCloud(point_cloud,"point_cloud");

    // pub margined potin
    sensor_msgs::PointCloud margin_cloud;

    for (auto &lm : e->feat_manager.point_landmarks){
        int used_num = (int)lm.feats.size();
        if (!(used_num >= 2 && lm.start_frame < kWinSize - 2))
            continue;
        //if (lm->start_frame > kWindowSize * 3.0 / 4.0 || lm->solve_flag != 1)
        //        continue;

        if (lm.start_frame == 0 && lm.feats.size() <= 2 && lm.solve_flag == 1 ){
            int imu_i = lm.start_frame;
            margin_cloud.points.push_back(EigenToGeometryPoint32(
                    body.CamToWorld(lm.feats[0].point * lm.depth, imu_i)) );
        }
    }

    PublisherMap::PubPointCloud(margin_cloud,"margin_cloud");
}



void Publisher::PubTransform(const Mat3d &R,const Vec3d &P,tf::TransformBroadcaster &br,ros::Time time,
                  const string &frame_id,const string &child_frame_id){
    Quaterniond q_eigen(R);
    tf::Transform transform;
    transform.setOrigin(EigenToTfVector(P));
    transform.setRotation(EigenToTfQuaternion(q_eigen));
    br.sendTransform(tf::StampedTransform(transform, time, frame_id, child_frame_id));
}



void Publisher::PubTF(const std_msgs::Header &header)
{
    if( e->solver_flag != SolverFlag::kNonLinear)
        return;

    auto [R,P,R_bc,P_bc] = e->GetOutputEgoInfo();

    PubTransform(R,P,*transform_broadcaster,ros::Time::now(),"world","body");
    PubTransform(R_bc,P_bc,*transform_broadcaster,ros::Time::now(),"body","camera");

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position = EigenToGeometryPoint(P_bc);
    Quaterniond q_eigen{body.ric[0]};
    odometry.pose.pose.orientation = EigenToGeometryQuaternion(q_eigen);
    PublisherMap::Pub<nav_msgs::Odometry>(odometry,"extrinsic");
}



void Publisher::PubKeyframe()
{
    // pub camera pose, 2D-3D points of keyframe
    if (e->solver_flag == SolverFlag::kNonLinear && e->margin_flag == MarginFlag::kMarginOld)
    {
        int i = kWinSize - 2;
        //Vector3d P = e.Ps[i] + e.Rs[i] * e.tic[0];
        Vector3d P = body.Ps[i];
        auto q_eigen = Quaterniond(body.Rs[i]);

        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time(body.headers[kWinSize - 2]);
        odometry.header.frame_id = "world";
        odometry.pose.pose.position = EigenToGeometryPoint(P);
        odometry.pose.pose.orientation = EigenToGeometryQuaternion(q_eigen);
        PublisherMap::Pub<nav_msgs::Odometry>(odometry,"keyframe_pose");

        sensor_msgs::PointCloud point_cloud;
        point_cloud.header.stamp = ros::Time(body.headers[kWinSize - 2]);
        point_cloud.header.frame_id = "world";
        for (auto &lm : e->feat_manager.point_landmarks){
            int frame_size = (int)lm.feats.size();
            if(lm.start_frame < kWinSize - 2 && lm.start_frame + frame_size - 1 >= kWinSize - 2
            && lm.solve_flag == 1){
                int imu_i = lm.start_frame;
                point_cloud.points.push_back(EigenToGeometryPoint32(
                        body.CamToWorld(lm.feats[0].point * lm.depth, imu_i)));

                int imu_j = kWinSize - 2 - lm.start_frame;
                sensor_msgs::ChannelFloat32 p_2d;
                p_2d.values.push_back(float(lm.feats[imu_j].point.x()));
                p_2d.values.push_back(float(lm.feats[imu_j].point.y()));
                p_2d.values.push_back(float(lm.feats[imu_j].uv.x()));
                p_2d.values.push_back(float(lm.feats[imu_j].uv.y()));
                p_2d.values.push_back(float(lm.feature_id));
                point_cloud.channels.push_back(p_2d);
            }
        }
        PublisherMap::PubPointCloud(point_cloud,"keyframe_point");
    }
}



/**
 * 可视化3D检测框
 * @param boxes
 */
void Publisher::PubPredictBox3D(const std_msgs::Header &header)
{
    MarkerArray markers;
    cv::Scalar color_norm(0.5,0.5,0.5);

    int frame = body.frame-1;//因为已经经过了边缘化，所以要-1

    for(auto &[key,inst] : e->im.instances){
        if(!inst.is_curr_visible || !inst.boxes3d[frame]){
            continue;
        }
        Mat38d corners = inst.boxes3d[frame]->corners;
        for(int i=0;i<8;++i)
            corners.col(i) = body.CamToWorld(corners.col(i),frame);

        auto cube_marker = CubeMarker(corners, inst.id + 4000, color_norm,
                                      0.05,Marker::ADD,"predict_boxes",3,20);
        markers.markers.push_back(cube_marker);
    }

    PublisherMap::PubMarkers(markers,"instance_marker");
}


void Publisher::PubGroundTruthBox3D(const std_msgs::Header &header)
{
    MarkerArray markers;
    cv::Scalar color_norm(0.5,0.5,0.5);

    int frame = body.frame-1;//因为已经经过了边缘化，所以要-1

    ///可视化gt框
    if(cfg::dataset==DatasetType::kKitti){
        int index=10000;
        auto boxes_gt = Detector3D::ReadGroundtruthFromKittiTracking(body.seq_id);
        for(auto &box : boxes_gt){
            Mat38d corners_w = box->corners;
            for(int i=0;i<8;++i){
                corners_w.col(i) = body.CamToWorld(corners_w.col(i),frame);
            }
            auto gt_cube_marker = CubeMarker(corners_w, index, BgrColor("magenta"),
                                             0.06, Marker::ADD,"gt_boxes",3,20);
            markers.markers.push_back(gt_cube_marker);
            index++;
        }

        PublisherMap::PubMarkers(markers,"instance_marker");
    }
    else{
        throw std::runtime_error(fmt::format("Publisher::PubGroundTruthBox3D(): the visualization "
                                             "of gt_boxes of {} is not implemented",cfg::dataset_name));
    }

}


void Publisher::PubLines(const std_msgs::Header &header)
{
    MarkerArray markers;
    ///根据box初始化物体的位姿和包围框
    cv::Scalar color_norm(0.5,0.5,0.5);

    for(auto &line:e->feat_manager.line_landmarks){
        if(!line.is_triangulation){
            continue;
        }
        if(cfg::dataset == DatasetType::kCustom){
            Marker m = LineMarker(line.ptw1,line.ptw2,line.feature_id,color_norm,0.02);
            markers.markers.push_back(m);
        }
        else{
            Marker m = LineMarker(line.ptw1,line.ptw2,line.feature_id,color_norm);
            markers.markers.push_back(m);
        }

    }

    PublisherMap::PubMarkers(markers,"lines");
}




/**
 * 输出物体的点云
 * @param header
 */
void Publisher::PubInstancePointCloud(const std_msgs::Header &header){
    if(e->im.tracking_number() < 1)
        return;

    PointCloud instance_point_cloud;
    PointCloud stereo_point_cloud;

    bool is_visual_all_point= false;

    ///输出角点
    for(auto &[key,inst] : e->im.instances){
        if(!inst.is_tracking ){
            continue;
        }
        PointCloud cloud;
        for(auto &lm : inst.landmarks){
            if(lm.depth <= 0)continue;
            int frame_j=lm.frame();
            int frame_i=e->frame;
            Vec3d pt = inst.ObjectToWorld(inst.CamToObject(lm.feats.front()->point * lm.depth,frame_j),frame_i);
            cloud.push_back(PointPCL(pt,inst.color[2],inst.color[1],inst.color[0]));

            auto &back_p = lm.feats.back();
            if(!is_visual_all_point){
                if(back_p->frame >= e->frame-1 && back_p->is_triangulated){
                    cloud.push_back(PointPCL(back_p->p_w,128,128,128));
                }
            }
            else{
                for(auto &feat : lm.feats){
                    if(feat->is_triangulated){
                        cloud.push_back(PointPCL(feat->p_w,128,128,128));
                    }
                }
            }
        }
        instance_point_cloud+=cloud;
    }

    printf("所有实例点云的数量: %ld\n",instance_point_cloud.size());
    PublisherMap::PubPointCloud(instance_point_cloud,"instance_point_cloud");

    ///输出额外点
    PointCloud::Ptr pc(new PointCloud);
    for(auto &[key,inst] : e->im.instances){
        if(!inst.is_tracking ){
            continue;
        }
        if(!(inst.points_extra_pcl[body.frame-1]) || inst.points_extra_pcl[body.frame-1]->empty()){
            continue;
        }
        *pc += *(inst.points_extra_pcl[body.frame-1]);
    }

    PublisherMap::PubPointCloud(*pc,"stereo_point_cloud");
    //PointCloudPublisher::Pub(stereo_point_cloud,"instance_stereo_point_cloud");
}


/**
 * 可视化场景流
 * @param header
 */
void Publisher::PubSceneVec(const std_msgs::Header &header)
{
    if(e->im.tracking_number() < 1)
        return;
    MarkerArray markers;
    for(auto &[key,inst] : e->im.instances){
        if(!inst.is_tracking ){
            continue;
        }
        if(!inst.is_initial || !inst.is_curr_visible){
            //if(!inst.is_curr_visible){
            continue;
        }
        cv::Scalar color_norm = inst.color / 255.f;
        color_norm[3]=1.0;

        for(auto &lm:inst.landmarks){
            //需要多个观测，且在当前帧有观测
            if(!lm.bad && lm.feats.size()>1 && lm.feats.back()->frame==body.frame-1){
                if(lm.feats.back()->is_triangulated && lm.penultimate()->is_triangulated){
                    auto line = LineMarker(lm.feats.back()->p_w,lm.penultimate()->p_w,lm.id+10000,color_norm,
                               0.02,Marker::ADD,"scene_vec");
                    markers.markers.push_back(line);
                }
            }
        }
    }

    PublisherMap::PubMarkers(markers,"scene_vec");

}



void Publisher::PubInstances(const std_msgs::Header &header)
{
    if(e->im.tracking_number() < 1)
        return;

    MarkerArray markers;

    for(auto &[key,inst] : e->im.instances){
        if(!inst.is_tracking ){
            continue;
        }
        if(!inst.is_initial || !inst.is_curr_visible){
        //if(!inst.is_curr_visible){
            continue;
        }

        Debugv("PubInstances() inst:{}",key);

        Marker::_action_type action=Marker::ADD;
        if(!inst.is_curr_visible){
            action=Marker::DELETE;
        }

        cv::Scalar color_norm = inst.color / 255.f;
        color_norm[3]=1.0;
        cv::Scalar color_transparent = color_norm;
        color_transparent[3] = 0.3;//设置透明度为0.5
        cv::Scalar color_inv;
        color_inv[0] = 1. - color_norm[0];
        color_inv[1] = 1. - color_norm[1];
        color_inv[2] = 1. - color_norm[2];
        color_inv[3] = 1.;

        ///可视化滑动窗口内物体的位姿
        /*if(!inst.is_static){
            for(int i=0; i <= kWinSize; i++){
                auto text_marker = BuildTextMarker(inst.state[i].P,i,to_string(i),color_transparent,action);
                markers.markers.push_back(text_marker);
            }
        }*/

        ///可视化估计的包围框
        /*EigenContainer<Eigen::Vector3d> vertex;
        inst.GetBoxVertex(vertex);
        auto lineStripMarker = LineStripMarker(vertex, key, color_norm, 0.1, action);
        markers.markers.push_back(lineStripMarker);*/

        Mat38d corners_w = Box3D::GetCorners(inst.box3d->dims,inst.state[e->frame].R,inst.state[e->frame].P);
        auto estimate_cube_marker = CubeMarker(corners_w, key, color_norm,0.1, action,"cube_estimation",7);
        markers.markers.push_back(estimate_cube_marker);

        ///构建坐标轴
        if(io_para::is_pub_object_axis){
            double axis_len=4.;
            Mat34d axis_matrix;
            axis_matrix.col(0) = Vec3d::Zero();
            axis_matrix.col(1) = Vec3d(axis_len,0,0);
            axis_matrix.col(2) = Vec3d(0,axis_len,0);
            axis_matrix.col(3) = Vec3d(0,0,axis_len);
            for(int i=0;i<4;++i){
                axis_matrix.col(i) = inst.state[e->frame].R * axis_matrix.col(i) + inst.state[e->frame].P;
            }
            auto axis_markers = AxisMarker(axis_matrix, key);
            markers.markers.push_back(std::get<0>(axis_markers));
            markers.markers.push_back(std::get<1>(axis_markers));
            markers.markers.push_back(std::get<2>(axis_markers));
        }

        ///可视化历史轨迹
        /*if(io_para::is_pub_object_trajectory){
            if(inst.is_initial ){
                auto history_marker = BuildTrajectoryMarker(key,inst.history_pose,inst.state,color_norm,action);
                markers.markers.push_back(history_marker);
            }
        }*/

        string text=fmt::format("{}\n p:{}", inst.id,VecToStr(inst.state[kWinSize].P));
        ///计算可视化文字信息
        if(!inst.is_static && inst.is_init_velocity && inst.vel.v.norm() > 1.){
            Eigen::Vector3d vel = hat(inst.vel.a) * inst.state[0].P + inst.vel.v;
            text += fmt::format("\n v:{}",VecToStr(vel));

            //可视化速度
            //Eigen::Vector3d end= inst.state[kWinSize].P + vel.normalized() * 2;
            //auto arrowMarker = ArrowMarker(inst.state[kWinSize].P, end, key, color_norm, 0.1, action);
            //markers.markers.push_back(arrowMarker);
        }
        else if(inst.is_static){
            text += "\nstatic";
        }
        auto textMarker = TextMarker(inst.state[kWinSize].P, key, text, BgrColor("blue"), 1.2, action);
        markers.markers.push_back(textMarker);
    }

    ///设置删除当前帧不显示的marker
    /*std::set<int> curr_marker_ids;
    for(auto &m:markers.markers){
        curr_marker_ids.insert(m.id);
    }
    //求差集
    vector<int> ids_not_curr;
    std::set_difference(last_marker_ids.begin(),last_marker_ids.end(),curr_marker_ids.begin(),curr_marker_ids.end(),
                        std::back_inserter(ids_not_curr));
    //设置要删除的marker
    for(auto m_id:ids_not_curr){
        Marker msg;
        msg.header.frame_id="world";
        msg.header.stamp=ros::Time::now();
        msg.ns="box_strip";
        msg.action=Marker::DELETE;
        msg.id=m_id;
        msg.type=Marker::LINE_STRIP;
        markers.markers.push_back(msg);
    }
    last_marker_ids=curr_marker_ids;*/

    PublisherMap::PubMarkers(markers,"instance_marker");


}




}


