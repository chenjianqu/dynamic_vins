/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/
/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/



#include "visualization.h"

#include <fstream>
#include <string>

#include "estimator/instance_manager.h"


namespace dynamic_vins{\

using namespace std;
using namespace ros;
using namespace Eigen;

constexpr int32_t kMarkerTypeNumber=10;

std::set<int> last_marker_ids;//保存上一时刻发布的marker的id,用于将上一时刻的id清除


static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);

CameraPoseVisualization::Ptr camera_pose_visual;

std::shared_ptr<ros::Publisher> pub_odometry;
std::shared_ptr<ros::Publisher> pub_latest_odometry;
std::shared_ptr<ros::Publisher> pub_path;
std::shared_ptr<ros::Publisher> pub_point_cloud;
std::shared_ptr<ros::Publisher> pub_margin_cloud;
std::shared_ptr<ros::Publisher> pub_key_poses;
std::shared_ptr<ros::Publisher> pub_camera_pose;
std::shared_ptr<ros::Publisher> pub_camera_pose_visual;
std::shared_ptr<ros::Publisher> pub_keyframe_pose;
std::shared_ptr<ros::Publisher> pub_keyframe_point;
std::shared_ptr<ros::Publisher> pub_extrinsic;
std::shared_ptr<ros::Publisher> pub_image_track;
std::shared_ptr<ros::Publisher> pub_instance_pointcloud;
std::shared_ptr<ros::Publisher> pub_instance_marker;
std::shared_ptr<ros::Publisher> pub_stereo_pointcloud;

std::shared_ptr<tf::TransformBroadcaster> transform_broadcaster;

nav_msgs::Path path;

/**
 * 构造ROS发布器
 * @param n
 */
void Publisher::RegisterPub(ros::NodeHandle &n)
{
    transform_broadcaster = std::make_shared<tf::TransformBroadcaster>();

    pub_latest_odometry=std::make_shared<ros::Publisher>();
    *pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);

    pub_path=std::make_shared<ros::Publisher>();
    *pub_path = n.advertise<nav_msgs::Path>("path", 1000);

    pub_odometry=std::make_shared<ros::Publisher>();
    *pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);

    pub_point_cloud=std::make_shared<ros::Publisher>();
    *pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);

    pub_margin_cloud=std::make_shared<ros::Publisher>();
    *pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud", 1000);

    pub_key_poses=std::make_shared<ros::Publisher>();
    *pub_key_poses = n.advertise<Marker>("key_poses", 1000);

    pub_camera_pose=std::make_shared<ros::Publisher>();
    *pub_camera_pose = n.advertise<nav_msgs::Odometry>("camera_pose", 1000);

    pub_camera_pose_visual=std::make_shared<ros::Publisher>();
    *pub_camera_pose_visual = n.advertise<MarkerArray>("camera_pose_visual", 1000);

    pub_keyframe_pose=std::make_shared<ros::Publisher>();
    *pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);

    pub_keyframe_point=std::make_shared<ros::Publisher>();
    *pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);

    pub_extrinsic=std::make_shared<ros::Publisher>();
    *pub_extrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 1000);

    pub_image_track=std::make_shared<ros::Publisher>();
    *pub_image_track = n.advertise<sensor_msgs::Image>("image_track", 1000);

    pub_instance_pointcloud=std::make_shared<ros::Publisher>();
    *pub_instance_pointcloud=n.advertise<sensor_msgs::PointCloud2>("instance_point_cloud", 1000);

    pub_instance_marker=std::make_shared<ros::Publisher>();
    *pub_instance_marker=n.advertise<MarkerArray>("instance_marker", 1000);

    pub_stereo_pointcloud=std::make_shared<ros::Publisher>();
    *pub_stereo_pointcloud=n.advertise<sensor_msgs::PointCloud2>("instance_stereo_point_cloud", 1000);

    camera_pose_visual=std::make_shared<CameraPoseVisualization>(1, 0, 0, 1);
    camera_pose_visual->setScale(0.5);
    camera_pose_visual->setLineWidth(0.05);
}

void Publisher::PubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t)
{
    nav_msgs::Odometry odometry;
    odometry.header.stamp = ros::Time(t);
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = Q.x();
    odometry.pose.pose.orientation.y = Q.y();
    odometry.pose.pose.orientation.z = Q.z();
    odometry.pose.pose.orientation.w = Q.w();
    odometry.twist.twist.linear.x = V.x();
    odometry.twist.twist.linear.y = V.y();
    odometry.twist.twist.linear.z = V.z();
    pub_latest_odometry->publish(odometry);

    //Debugv("Publisher::pubLatestOdometry:{}", VecToStr(P));
}

void Publisher::PubTrackImage(const cv::Mat &imgTrack, const double t)
{
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(t);
    sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
    pub_image_track->publish(imgTrackMsg);
}


void Publisher::PrintStatistics(double t)
{
    if (e->solver_flag != SolverFlag::kNonLinear)
        return;
    if (cfg::is_estimate_ex){
        cv::FileStorage fs(cfg::kExCalibResultPath, cv::FileStorage::WRITE);
        for (int i = 0; i < cfg::kCamNum; i++){
            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = e->ric[i];
            eigen_T.block<3, 1>(0, 3) = e->tic[i];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            if(i == 0)
                fs << "body_T_cam0" << cv_T ;
            else
                fs << "body_T_cam1" << cv_T ;
        }
        fs.release();
    }

    sum_of_path += (e->Ps[kWinSize] - last_path).norm();
    last_path = e->Ps[kWinSize];
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
    tmp_Q = Quaterniond(e->Rs[kWinSize]);
    odometry.pose.pose.position.x = e->Ps[kWinSize].x();
    odometry.pose.pose.position.y = e->Ps[kWinSize].y();
    odometry.pose.pose.position.z = e->Ps[kWinSize].z();
    odometry.pose.pose.orientation.x = tmp_Q.x();
    odometry.pose.pose.orientation.y = tmp_Q.y();
    odometry.pose.pose.orientation.z = tmp_Q.z();
    odometry.pose.pose.orientation.w = tmp_Q.w();
    odometry.twist.twist.linear.x = e->Vs[kWinSize].x();
    odometry.twist.twist.linear.y = e->Vs[kWinSize].y();
    odometry.twist.twist.linear.z = e->Vs[kWinSize].z();
    pub_odometry->publish(odometry);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = header;
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose = odometry.pose.pose;
    path.header = header;
    path.header.frame_id = "world";
    path.poses.push_back(pose_stamped);
    pub_path->publish(path);

    // write result to file
    ofstream foutC(io_para::kVinsResultPath, ios::app);
    foutC.setf(ios::fixed, ios::floatfield);
    /*
    foutC.precision(0);
    foutC << header.stamp.toSec() * 1e9 << ",";
    foutC.precision(5);
    foutC << e.Ps[kWindowSize].x() << ","
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
    foutC << header.stamp << " "
    << e->Ps[kWinSize].x() << " "
    << e->Ps[kWinSize].y() << " "
    << e->Ps[kWinSize].z() << " "
    <<tmp_Q.x()<<" "
    <<tmp_Q.y()<<" "
    <<tmp_Q.z()<<" "
    <<tmp_Q.w()<<endl;
    foutC.close();

    Eigen::Vector3d tmp_T = e->Ps[kWinSize];
    printf("time: %f, t: %f %f %f q: %f %f %f %f \n", header.stamp.toSec(),
           tmp_T.x(), tmp_T.y(), tmp_T.z(),
           tmp_Q.w(), tmp_Q.x(), tmp_Q.y(), tmp_Q.z());
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
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= kWinSize; i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = e->key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses->publish(key_poses);
}


void Publisher::PubCameraPose(const std_msgs::Header &header)
{
    if(e->solver_flag != SolverFlag::kNonLinear)
        return;

    int idx2 = kWinSize - 1;
    int i = idx2;
    Vector3d P = e->Ps[i] + e->Rs[i] * e->tic[0];
    Quaterniond R = Quaterniond(e->Rs[i] * e->ric[0]);

    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = P.x();
    odometry.pose.pose.position.y = P.y();
    odometry.pose.pose.position.z = P.z();
    odometry.pose.pose.orientation.x = R.x();
    odometry.pose.pose.orientation.y = R.y();
    odometry.pose.pose.orientation.z = R.z();
    odometry.pose.pose.orientation.w = R.w();

    pub_camera_pose->publish(odometry);

    camera_pose_visual->reset();
    camera_pose_visual->add_pose(P, R);
    if(cfg::is_stereo){
        Vector3d P = e->Ps[i] + e->Rs[i] * e->tic[1];
        Quaterniond R = Quaterniond(e->Rs[i] * e->ric[1]);
        camera_pose_visual->add_pose(P, R);
    }
    camera_pose_visual->publish_by(*pub_camera_pose_visual, odometry.header);

}


void Publisher::PubPointCloud(const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;

    for (auto &it_per_id : e->f_manager.feature)
    {
        int used_num = (int)it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < kWinSize - 2))
            continue;
        if (it_per_id.start_frame > kWinSize * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = e->Rs[imu_i] * (e->ric[0] * pts_i + e->tic[0]) + e->Ps[imu_i];

        geometry_msgs::Point32 p;
        p.x = (float)w_pts_i(0);
        p.y = (float)w_pts_i(1);
        p.z = (float)w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud->publish(point_cloud);

    // pub margined potin
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &it_per_id : e->f_manager.feature){
        int used_num = (int)it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < kWinSize - 2))
            continue;
        //if (it_per_id->start_frame > kWindowSize * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
        && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = e->Rs[imu_i] * (e->ric[0] * pts_i + e->tic[0]) + e->Ps[imu_i];

            geometry_msgs::Point32 p;
            p.x = (float)w_pts_i(0);
            p.y = (float)w_pts_i(1);
            p.z = (float)w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud->publish(margin_cloud);
}



void Publisher::PubTransform(const Mat3d &R,const Vec3d &P,tf::TransformBroadcaster &br,ros::Time time,
                  const string &frame_id,const string &child_frame_id){
    Quaterniond q_eigen(R);

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(P.x(),P.y(),P.z()));

    tf::Quaternion q;
    q.setW(q_eigen.w());
    q.setX(q_eigen.x());
    q.setY(q_eigen.y());
    q.setZ(q_eigen.z());

    transform.setRotation(q);

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
    odometry.pose.pose.position.x = P_bc.x();
    odometry.pose.pose.position.y = P_bc.y();
    odometry.pose.pose.position.z = P_bc.z();
    Quaterniond tmp_q{e->ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    pub_extrinsic->publish(odometry);
}

void Publisher::PubKeyframe()
{
    // pub camera pose, 2D-3D points of keyframe
    if (e->solver_flag == SolverFlag::kNonLinear && e->margin_flag == MarginFlag::kMarginOld)
    {
        int i = kWinSize - 2;
        //Vector3d P = e.Ps[i] + e.Rs[i] * e.tic[0];
        Vector3d P = e->Ps[i];
        auto R = Quaterniond(e->Rs[i]);

        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time(e->headers[kWinSize - 2]);
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();

        pub_keyframe_pose->publish(odometry);

        sensor_msgs::PointCloud point_cloud;
        point_cloud.header.stamp = ros::Time(e->headers[kWinSize - 2]);
        point_cloud.header.frame_id = "world";
        for (auto &it_per_id : e->f_manager.feature){
            int frame_size = (int)it_per_id.feature_per_frame.size();
            if(it_per_id.start_frame < kWinSize - 2 && it_per_id.start_frame + frame_size - 1 >= kWinSize - 2
            && it_per_id.solve_flag == 1){
                int imu_i = it_per_id.start_frame;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = e->Rs[imu_i] * (e->ric[0] * pts_i + e->tic[0])
                        + e->Ps[imu_i];
                geometry_msgs::Point32 p;
                p.x = (float)w_pts_i(0);
                p.y = (float)w_pts_i(1);
                p.z = (float)w_pts_i(2);
                point_cloud.points.push_back(p);

                int imu_j = kWinSize - 2 - it_per_id.start_frame;
                sensor_msgs::ChannelFloat32 p_2d;
                p_2d.values.push_back(float(it_per_id.feature_per_frame[imu_j].point.x()));
                p_2d.values.push_back(float(it_per_id.feature_per_frame[imu_j].point.y()));
                p_2d.values.push_back(float(it_per_id.feature_per_frame[imu_j].uv.x()));
                p_2d.values.push_back(float(it_per_id.feature_per_frame[imu_j].uv.y()));
                p_2d.values.push_back(float(it_per_id.feature_id));
                point_cloud.channels.push_back(p_2d);
            }

        }
        pub_keyframe_point->publish(point_cloud);
    }
}



Marker Publisher::BuildLineStripMarker(PointT &maxPt,PointT &minPt,unsigned int id,const cv::Scalar &color,
                                       Marker::_action_type action,int offset)
{
    //设置立方体的八个顶点
    geometry_msgs::Point p[8];
    p[0].x=minPt.x;p[0].y=minPt.y;p[0].z=minPt.z;
    p[1].x=maxPt.x;p[1].y=minPt.y;p[1].z=minPt.z;
    p[2].x=maxPt.x;p[2].y=minPt.y;p[2].z=maxPt.z;
    p[3].x=minPt.x;p[3].y=minPt.y;p[3].z=maxPt.z;
    p[4].x=minPt.x;p[4].y=maxPt.y;p[4].z=maxPt.z;
    p[5].x=maxPt.x;p[5].y=maxPt.y;p[5].z=maxPt.z;
    p[6].x=maxPt.x;p[6].y=maxPt.y;p[6].z=minPt.z;
    p[7].x=minPt.x;p[7].y=maxPt.y;p[7].z=minPt.z;

    return BuildLineStripMarker(p,id,color,action,offset);
}


Marker Publisher::BuildLineStripMarker(EigenContainer<Eigen::Vector3d> &p,unsigned int id,const cv::Scalar &color,
                                       Marker::_action_type action,int offset){
    geometry_msgs::Point points[8];
    for(int i=0;i<8;++i){
        points[i].x = p[i].x();
        points[i].y = p[i].y();
        points[i].z = p[i].z();
    }
    return BuildLineStripMarker(points,id,color,action,offset);
}


Marker Publisher::BuildLineStripMarker(geometry_msgs::Point p[8],unsigned int id,const cv::Scalar &color,
                                       Marker::_action_type action,int offset)
{
    Marker msg;
    msg.header.stamp=ros::Time::now();
    msg.header.frame_id="world";
    msg.ns="box_strip";
    msg.action=action;
    msg.pose.orientation.w=1.0;

    msg.id=id * kMarkerTypeNumber + offset;//当存在多个marker时用于标志出来
    msg.type=Marker::LINE_STRIP;//marker的类型

    if(action==Marker::DELETE){
        return msg;
    }

    msg.lifetime=ros::Duration(io_para::kVisualInstDuration);//持续时间，若为ros::Duration()表示一直持续
    msg.scale.x=0.08;//线宽
    msg.color.r=(float)color[2];msg.color.g=(float)color[1];msg.color.b=(float)color[0];//颜色:0-1
    msg.color.a=1.0;//不透明度

    //这个类型仅将相邻点进行连线
    for(int i=0;i<8;++i){
        msg.points.push_back(p[i]);
    }
    //为了保证矩形框的其它边存在：
    msg.points.push_back(p[0]);
    msg.points.push_back(p[3]);
    msg.points.push_back(p[2]);
    msg.points.push_back(p[5]);
    msg.points.push_back(p[6]);
    msg.points.push_back(p[1]);
    msg.points.push_back(p[0]);
    msg.points.push_back(p[7]);
    msg.points.push_back(p[4]);

    return msg;
}


Marker Publisher::BuildTextMarker(const Eigen::Vector3d &point,unsigned int id,const std::string &text,const cv::Scalar &color,
                                  const double scale,Marker::_action_type action,int offset){
    Marker msg;
    msg.header.stamp=ros::Time::now();
    msg.header.frame_id="world";
    msg.ns="box_text";
    msg.action=action;

    msg.id=id * kMarkerTypeNumber + offset;//当存在多个marker时用于标志出来
    msg.type=Marker::TEXT_VIEW_FACING;//marker的类型
    if(action==Marker::DELETE){
        return msg;
    }

    msg.lifetime=ros::Duration(io_para::kVisualInstDuration);//持续时间4s，若为ros::Duration()表示一直持续

    msg.scale.z=scale;//字体大小
    msg.color.r=(float)color[2];msg.color.g=(float)color[1];msg.color.b=(float)color[0];
    msg.color.a=(float)color[3];//不透明度

    geometry_msgs::Pose pose;
    pose.position.x=point.x();pose.position.y=point.y();pose.position.z=point.z();
    pose.orientation.w=1.0;
    msg.pose=pose;

    msg.text=text;

    return msg;
}


Marker  Publisher::BuildTextMarker(const PointT &point,unsigned int id,const std::string &text,
                                   const cv::Scalar &color,const double scale,Marker::_action_type action,int offset)
{
    Eigen::Vector3d eigen_pt;
    eigen_pt<<point.x,point.y,point.z;
    return BuildTextMarker(eigen_pt,id,text,color,scale,action,offset);
}

Marker Publisher::BuildArrowMarker(const Eigen::Vector3d &start_pt,const Eigen::Vector3d &end_pt,unsigned int id,
                                   const cv::Scalar &color,Marker::_action_type action,int offset)
{
    Marker msg;
    msg.header.frame_id="world";
    msg.header.stamp=ros::Time::now();
    msg.ns="arrow_strip";
    msg.action=action;

    msg.id=id * kMarkerTypeNumber + offset;//当存在多个marker时用于标志出来
    msg.type=Marker::LINE_STRIP;//marker的类型

    if(action==Marker::DELETE){
        return msg;
    }

    msg.pose.orientation.w=1.0;

    msg.lifetime=ros::Duration(io_para::kVisualInstDuration);

    msg.scale.x=0.01;//线宽
    msg.color.r=(float)color[2];msg.color.g=(float)color[1];msg.color.b=(float)color[0];
    msg.color.a=1.0;//不透明度

    geometry_msgs::Point start,end;
    start.x=start_pt.x();start.y=start_pt.y();start.z=start_pt.z();
    end.x=end_pt.x();end.y=end_pt.y();end.z=end_pt.z();

    const double arrow_len=std::sqrt((start.x-end.x) * (start.x-end.x) +(start.y-end.y) * (start.y-end.y) +
            (start.z-end.z) * (start.z-end.z)) /8.;

    geometry_msgs::Point p[7];
    p[0]=start;
    p[1]=end;
    p[2]=end;
    if(start.x < end.x){
        p[2].x -= arrow_len;
    }else{
        p[2].x += arrow_len;
    }
    p[3]=end;
    p[4]=end;
    if(start.y < end.y){
        p[4].y -= arrow_len;
    }else{
        p[4].y += arrow_len;
    }
    p[5]=end;
    p[6]=end;
    if(start.z < end.z){
        p[6].z -= arrow_len;
    }else{
        p[6].z += arrow_len;
    }

    //这个类型仅将相邻点进行连线
    for(auto &pt : p)
        msg.points.push_back(pt);

    return msg;
}


Marker Publisher::BuildCubeMarker(Eigen::Matrix<double,8,3> &corners,unsigned int id,const cv::Scalar &color,
                                  Marker::_action_type action,int offset){
    Marker msg;

    msg.header.frame_id="world";
    msg.header.stamp=ros::Time::now();
    msg.ns="box_strip";
    msg.action=action;

    msg.id=id * kMarkerTypeNumber + offset;//当存在多个marker时用于标志出来
    msg.type=Marker::LINE_STRIP;//marker的类型
    if(action==Marker::DELETE){
        return msg;
    }

    msg.lifetime=ros::Duration(io_para::kVisualInstDuration);//持续时间3s，若为ros::Duration()表示一直持续

    msg.pose.orientation.w=1.0;

    msg.scale.x=0.01;//线宽
    msg.color.r=(float)color[2];msg.color.g=(float)color[1];msg.color.b=(float)color[0];
    msg.color.a=1.0;

    //设置立方体的八个顶点
    geometry_msgs::Point p[8];
    p[0].x= corners(0,0);p[0].y=corners(0,1);p[0].z=corners(0,2);
    p[1].x= corners(1,0);p[1].y=corners(1,1);p[1].z=corners(1,2);
    p[2].x= corners(2,0);p[2].y=corners(2,1);p[2].z=corners(2,2);
    p[3].x= corners(3,0);p[3].y=corners(3,1);p[3].z=corners(3,2);
    p[4].x= corners(4,0);p[4].y=corners(4,1);p[4].z=corners(4,2);
    p[5].x= corners(5,0);p[5].y=corners(5,1);p[5].z=corners(5,2);
    p[6].x= corners(6,0);p[6].y=corners(6,1);p[6].z=corners(6,2);
    p[7].x= corners(7,0);p[7].y=corners(7,1);p[7].z=corners(7,2);
    /**
             .. code-block:: none

                             front z
                                    /
                                   /
                   p1(x0, y0, z1) + -----------  + p5(x1, y0, z1)
                                 /|            / |
                                / |           /  |
                p0(x0, y0, z0) + ----------- +   + p6(x1, y1, z1)
                               |  /      .   |  /
                               | / origin    | /
                p3(x0, y1, z0) + ----------- + -------> x right
                               |             p7(x1, y1, z0)
                               |
                               v
                        down y
     输入的点序列:p0:0,0,0, p1: 0,0,1,  p2: 0,1,1,  p3: 0,1,0,  p4: 1,0,0,  p5: 1,0,1,  p6: 1,1,1,  p7: 1,1,0;

     */
    msg.points.push_back(p[0]);
    msg.points.push_back(p[1]);
    msg.points.push_back(p[5]);
    msg.points.push_back(p[4]);
    msg.points.push_back(p[0]);
    msg.points.push_back(p[3]);
    msg.points.push_back(p[7]);
    msg.points.push_back(p[4]);
    msg.points.push_back(p[7]);
    msg.points.push_back(p[6]);
    msg.points.push_back(p[5]);
    msg.points.push_back(p[6]);
    msg.points.push_back(p[2]);
    msg.points.push_back(p[1]);
    msg.points.push_back(p[2]);
    msg.points.push_back(p[3]);

    return msg;
}


Marker Publisher::BuildTrajectoryMarker(unsigned int id,std::list<State> &history,State* sliding_window,
                                        const cv::Scalar &color,Marker::_action_type action,int offset){
    Marker msg;

    msg.header.frame_id="world";
    msg.header.stamp=ros::Time::now();
    msg.ns="box_strip";
    msg.action=action;
    msg.id=id * kMarkerTypeNumber + offset;//当存在多个marker时用于标志出来
    msg.type=Marker::LINE_STRIP;//marker的类型
    if(action==Marker::DELETE){
        return msg;
    }

    msg.pose.orientation.w=1.0;

    msg.lifetime=ros::Duration(io_para::kVisualInstDuration);//持续时间，若为ros::Duration()

    msg.scale.x=0.1;//线宽
    msg.color.r=(float)color[2];msg.color.g=(float)color[1];msg.color.b=(float)color[0];
    msg.color.a=1.0;//不透明度

    for(auto &pose : history){
        geometry_msgs::Point point;
        point.x = pose.P.x();
        point.y = pose.P.y();
        point.z = pose.P.z();
        msg.points.push_back(point);
    }

    for(int i=0;i<=kWinSize;++i){
        geometry_msgs::Point point;
        point.x = sliding_window[i].P.x();
        point.y = sliding_window[i].P.y();
        point.z = sliding_window[i].P.z();
        msg.points.push_back(point);
    }

    return msg;
}








void Publisher::PubPredictBox3D(std::vector<Box3D> &boxes)
{
    MarkerArray markers;

    ///根据box初始化物体的位姿和包围框
    auto cam_to_world = [](const Vec3d &p){
        Vec3d p_imu = e->ric[0] * p + e->tic[0];
        Vec3d p_world = e->Rs[e->frame] * p_imu + e->Ps[e->frame];
        return p_world;
    };

    cv::Scalar color_norm(0.5,0.5,0.5);

    int index=0;
    for(auto &box : boxes){
        //将包围框的8个顶点转换到世界坐标系下
        Mat38d corners = box.corners;
        for(int i=0;i<8;++i){
            corners.col(i) = cam_to_world(corners.col(i));
        }
        Eigen::Matrix<double,8,3> corners8x3=corners.transpose();
        string log_text = fmt::format("id:{} class:{} score:{}\n",index,box.class_id,box.score);
        log_text += EigenToStr(box.corners);
        Debugv(log_text);
        auto cube_marker = BuildCubeMarker(corners8x3,index+4000,color_norm);

        markers.markers.push_back(cube_marker);

        index++;
    }

    pub_instance_marker->publish(markers);
}





void Publisher::PubInstancePointCloud(const std_msgs::Header &header)
{
    if(e->insts_manager.tracking_number() < 1)
        return;

    MarkerArray markers;
    PointCloud instance_point_cloud;
    PointCloud stereo_point_cloud;

    for(auto &[key,inst] : e->insts_manager.instances){
        if(!inst.is_tracking ){
            continue;
        }

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
        EigenContainer<Eigen::Vector3d> vertex;
        inst.GetBoxVertex(vertex);
        auto lineStripMarker = BuildLineStripMarker(vertex,key,color_norm,action,0);
        markers.markers.push_back(lineStripMarker);

        ///可视化检测得到的包围框
        /*for(int i=0;i<=kWinSize;++i){
            if(inst.boxes3d[i]){
                Mat38d corners_w = inst.boxes3d[i]->GetCornersInWorld(e->Rs[i],e->Ps[i],
                                                                      e->ric[0],e->tic[0]);
                Eigen::Matrix<double,8,3> corners_w_t = corners_w.transpose();
                auto detect_cube_marker = BuildCubeMarker(corners_w_t,key,action);
                markers.markers.push_back(detect_cube_marker);
            }
        }*/
        if(inst.boxes3d[e->frame-1]){
            Mat38d corners_w = inst.boxes3d[e->frame-1]->GetCornersInWorld(
                    e->Rs[e->frame-1], e->Ps[e->frame-1],
                    e->ric[0],e->tic[0]);
            Eigen::Matrix<double,8,3> corners_w_t = corners_w.transpose();
            auto detect_cube_marker = BuildCubeMarker(corners_w_t,key,cv::Scalar(0.5,0.5,0.5),action,3);
            markers.markers.push_back(detect_cube_marker);
        }

        ///可视化历史轨迹
        if(inst.is_initial ){
            auto history_marker = BuildTrajectoryMarker(key,inst.history_pose,inst.state,color_norm,action,4);
            markers.markers.push_back(history_marker);
        }


        ///计算可视化的速度
        if(!inst.is_static && inst.is_init_velocity && inst.vel.v.norm() > 1.){
            Eigen::Vector3d vel = hat(inst.vel.a) * inst.state[0].P + inst.vel.v;
            string text=fmt::format("{}\n({})", inst.id, VecToStr(vel));
            auto textMarker = BuildTextMarker(inst.state[kWinSize].P, key, text, color_inv, 1.2,action,1);
            markers.markers.push_back(textMarker);

            Eigen::Vector3d end= inst.state[kWinSize].P + vel.normalized() * 4;
            auto arrowMarker = BuildArrowMarker(inst.state[kWinSize].P, end, key, color_norm,action,2);
            markers.markers.push_back(arrowMarker);
        }
        else if(inst.is_static){
            string text=fmt::format("{} static", inst.id);
            auto textMarker = BuildTextMarker(inst.state[kWinSize].P, key, text, color_inv, 1.2,action,1);
            markers.markers.push_back(textMarker);
        }
        else{
            string text=fmt::format("{}", inst.id);
            auto textMarker = BuildTextMarker(inst.state[kWinSize].P, key, text, color_inv, 1.2,action,1);
            markers.markers.push_back(textMarker);
        }


        ///可视化点
        bool is_visual_all_point= true;
        string log_text= fmt::format("inst:{} 3D Points\n",inst.id);

        PointCloud cloud;
        /*for(auto &pt : inst.point3d_curr){
            PointT p;
            p.x = (float)pt(0);p.y = (float)pt(1);p.z = (float)pt(2);
            p.r=(uint8_t)inst.color[2];p.g=(uint8_t)inst.color[1];p.b=(uint8_t)inst.color[0];
            cloud.push_back(p);
        }*/

        for(auto &lm : inst.landmarks){
            if(lm.depth <= 0)
                continue;

            int frame_j=lm.feats.front().frame;
            int frame_i=e->frame;
            Vec3d pts_cam_j = lm.feats.front().point * lm.depth;//k点在j时刻的相机坐标
            Vec3d pts_imu_j = e->ric[0] * pts_cam_j + e->tic[0];//k点在j时刻的IMU坐标
            Vec3d pt_w_j=e->Rs[frame_j] * pts_imu_j + e->Ps[frame_j];//k点在j时刻的世界坐标
            Vec3d pt_obj = inst.state[frame_j].R.transpose() * (pt_w_j - inst.state[frame_j].P);
            Vec3d pt = inst.state[frame_i].R *pt_obj + inst.state[frame_i].P;

            PointT p;
            p.x = (float)pt(0);p.y = (float)pt(1);p.z = (float)pt(2);
            p.r=(uint8_t)inst.color[2];p.g=(uint8_t)inst.color[1];p.b=(uint8_t)inst.color[0];
            cloud.push_back(p);

            //log_text += VecToStr(pt)+" ";

            if(!is_visual_all_point){
                if(lm.feats.back().frame >= e->frame-1 && lm.feats.back().is_triangulated){
                    PointT ps(255,255,255);
                    ps.x =(float) lm.feats.back().p_w.x();
                    ps.y =(float) lm.feats.back().p_w.y();
                    ps.z =(float) lm.feats.back().p_w.z();
                    stereo_point_cloud.push_back(ps);
                }
            }
            else{
                for(auto &feat : lm.feats){
                    if(feat.is_triangulated){
                        PointT ps(255,255,255);
                        ps.x =(float) feat.p_w.x();
                        ps.y =(float) feat.p_w.y();
                        ps.z =(float) feat.p_w.z();
                        stereo_point_cloud.push_back(ps);
                    }
                }
            }

        }
        //Debugv(log_text);

        instance_point_cloud+=cloud;
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

    pub_instance_marker->publish(markers);


    printf("实例点云的数量: %ld\n",instance_point_cloud.size());

    sensor_msgs::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(instance_point_cloud,point_cloud_msg);
    point_cloud_msg.header = header;
    pub_instance_pointcloud->publish(point_cloud_msg);

    sensor_msgs::PointCloud2 point_stereo_cloud_msg;
    pcl::toROSMsg(stereo_point_cloud,point_stereo_cloud_msg);
    point_stereo_cloud_msg.header = header;
    pub_stereo_pointcloud->publish(point_stereo_cloud_msg);
}





/**
 * 保存到文件中每行的内容
 * 1    frame        Frame within the sequence where the object appearers
   1    track id     Unique tracking id of this object within this sequence
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Integer (0,1,2) indicating the level of truncation.
                     Note that this is in contrast to the object detection
                     benchmark where truncation is a float in [0,1].
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

alpha和rotation_y的区别：
 The coordinates in the camera coordinate system can be projected in the image
by using the 3x4 projection matrix in the calib folder, where for the left
color camera for which the images are provided, P2 must be used. The
difference between rotation_y and alpha is, that rotation_y is directly
given in camera coordinates, while alpha also considers the vector from the
camera center to the object center, to compute the relative orientation of
the object with respect to the camera. For example, a car which is facing
along the X-axis of the camera coordinate system corresponds to rotation_y=0,
no matter where it is located in the X/Z plane (bird's eye view), while
alpha is zero only, when this object is located along the Z-axis of the
camera. When moving the car away from the Z-axis, the observation angle
(\alpha) will change.
 */

void SaveInstanceTrajectory(unsigned int frame_id,unsigned int track_id,std::string &type,
                            int truncated,int occluded,double alpha,Vec4d &box,
                            Vec3d &dims,Vec3d &location,double rotation_y,double score){
    //追加写入
    ofstream fout(io_para::kObjectResultPath,std::ios::out | std::ios::app);

    fout<<frame_id<<" "<<track_id<<" "<<type<<" "<<truncated<<" "<<occluded<<" ";
    fout<<alpha<<" "<<fmt::format("{} {} {} {}",box.x(),box.y(),box.z(),box.w())<<" "
    <<VecToStr(dims)<<" "<<VecToStr(location)<<
    " "<<rotation_y<<" "<<score;
    fout<<endl;
    fout.close();
}





}
