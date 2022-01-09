/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "visualization.h"
#include "../estimator/instance_manager.h"

namespace dynamic_vins{\


using namespace ros;
using namespace Eigen;

ros::Publisher pub_odometry, pub_latest_odometry;
ros::Publisher pub_path;
ros::Publisher pub_point_cloud, pub_margin_cloud;
ros::Publisher pub_key_poses;
ros::Publisher pub_camera_pose;
ros::Publisher pub_camera_pose_visual;
nav_msgs::Path path;

ros::Publisher pub_keyframe_pose;
ros::Publisher pub_keyframe_point;
ros::Publisher pub_extrinsic;

ros::Publisher pub_image_track;


ros::Publisher pub_instance_pointcloud;
ros::Publisher pub_instance_marker;

CameraPoseVisualization cameraposevisual(1, 0, 0, 1);
static double sum_of_path = 0;
static Vector3d last_path(0.0, 0.0, 0.0);


unsigned int MarkerTypeNumber=3;

void registerPub(ros::NodeHandle &n)
{
    pub_latest_odometry = n.advertise<nav_msgs::Odometry>("imu_propagate", 1000);
    pub_path = n.advertise<nav_msgs::Path>("path", 1000);
    pub_odometry = n.advertise<nav_msgs::Odometry>("odometry", 1000);
    pub_point_cloud = n.advertise<sensor_msgs::PointCloud>("point_cloud", 1000);
    pub_margin_cloud = n.advertise<sensor_msgs::PointCloud>("margin_cloud", 1000);
    pub_key_poses = n.advertise<visualization_msgs::Marker>("key_poses", 1000);
    pub_camera_pose = n.advertise<nav_msgs::Odometry>("camera_pose", 1000);
    pub_camera_pose_visual = n.advertise<visualization_msgs::MarkerArray>("camera_pose_visual", 1000);
    pub_keyframe_pose = n.advertise<nav_msgs::Odometry>("keyframe_pose", 1000);
    pub_keyframe_point = n.advertise<sensor_msgs::PointCloud>("keyframe_point", 1000);
    pub_extrinsic = n.advertise<nav_msgs::Odometry>("extrinsic", 1000);
    pub_image_track = n.advertise<sensor_msgs::Image>("image_track", 1000);

    pub_instance_pointcloud=n.advertise<sensor_msgs::PointCloud2>("instance_point_cloud", 1000);
    pub_instance_marker=n.advertise<visualization_msgs::MarkerArray>("instance_marker", 1000);


    cameraposevisual.setScale(0.1);
    cameraposevisual.setLineWidth(0.01);
}

void pubLatestOdometry(const Eigen::Vector3d &P, const Eigen::Quaterniond &Q, const Eigen::Vector3d &V, double t)
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
    pub_latest_odometry.publish(odometry);
}

void PubTrackImage(const cv::Mat &imgTrack, const double t)
{
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(t);
    sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
    pub_image_track.publish(imgTrackMsg);
}


void printStatistics(const Estimator &estimator, double t)
{
    if (estimator.solver_flag != SolverFlag::kNonLinear)
        return;
    //printf("position: %f, %f, %f\r", e.Ps[kWindowSize].x(), e.Ps[kWindowSize].y(), e.Ps[kWindowSize].z());
    ROS_DEBUG_STREAM("position: " << estimator.Ps[kWindowSize].transpose());
    ROS_DEBUG_STREAM("orientation: " << estimator.Vs[kWindowSize].transpose());
    if (cfg::ESTIMATE_EXTRINSIC)
    {
        cv::FileStorage fs(cfg::kExCalibResultPath, cv::FileStorage::WRITE);
        for (int i = 0; i < cfg::kCamNum; i++)
        {
            //ROS_DEBUG("calibration result for camera %d", i);
            ROS_DEBUG_STREAM("extirnsic Tic: " << estimator.tic[i].transpose());
            ROS_DEBUG_STREAM("extrinsic ric: " << Utility::R2ypr(estimator.ric[i]).transpose());

            Eigen::Matrix4d eigen_T = Eigen::Matrix4d::Identity();
            eigen_T.block<3, 3>(0, 0) = estimator.ric[i];
            eigen_T.block<3, 1>(0, 3) = estimator.tic[i];
            cv::Mat cv_T;
            cv::eigen2cv(eigen_T, cv_T);
            if(i == 0)
                fs << "body_T_cam0" << cv_T ;
            else
                fs << "body_T_cam1" << cv_T ;
        }
        fs.release();
    }

    static double sum_of_time = 0;
    static int sum_of_calculation = 0;
    sum_of_time += t;
    sum_of_calculation++;
    ROS_DEBUG("vo solver costs: %f ms", t);
    ROS_DEBUG("average of time %f ms", sum_of_time / sum_of_calculation);

    sum_of_path += (estimator.Ps[kWindowSize] - last_path).norm();
    last_path = estimator.Ps[kWindowSize];
    ROS_DEBUG("sum of path %f", sum_of_path);
    if (cfg::is_estimate_td)
        ROS_INFO("td %f", estimator.td);
}

void pubOdometry(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.solver_flag == SolverFlag::kNonLinear)
    {
        nav_msgs::Odometry odometry;
        odometry.header = header;
        odometry.header.frame_id = "world";
        odometry.child_frame_id = "world";
        Quaterniond tmp_Q;
        tmp_Q = Quaterniond(estimator.Rs[kWindowSize]);
        odometry.pose.pose.position.x = estimator.Ps[kWindowSize].x();
        odometry.pose.pose.position.y = estimator.Ps[kWindowSize].y();
        odometry.pose.pose.position.z = estimator.Ps[kWindowSize].z();
        odometry.pose.pose.orientation.x = tmp_Q.x();
        odometry.pose.pose.orientation.y = tmp_Q.y();
        odometry.pose.pose.orientation.z = tmp_Q.z();
        odometry.pose.pose.orientation.w = tmp_Q.w();
        odometry.twist.twist.linear.x = estimator.Vs[kWindowSize].x();
        odometry.twist.twist.linear.y = estimator.Vs[kWindowSize].y();
        odometry.twist.twist.linear.z = estimator.Vs[kWindowSize].z();
        pub_odometry.publish(odometry);

        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header = header;
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose = odometry.pose.pose;
        path.header = header;
        path.header.frame_id = "world";
        path.poses.push_back(pose_stamped);
        pub_path.publish(path);

        // write result to file
        ofstream foutC(cfg::kVinsResultPath, ios::app);
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
        << estimator.Ps[kWindowSize].x() << " "
        << estimator.Ps[kWindowSize].y() << " "
        << estimator.Ps[kWindowSize].z() << " "
        <<tmp_Q.x()<<" "
        <<tmp_Q.y()<<" "
        <<tmp_Q.z()<<" "
        <<tmp_Q.w()<<endl;
        foutC.close();
        Eigen::Vector3d tmp_T = estimator.Ps[kWindowSize];
        printf("time: %f, t: %f %f %f q: %f %f %f %f \n", header.stamp.toSec(), tmp_T.x(), tmp_T.y(), tmp_T.z(),
               tmp_Q.w(), tmp_Q.x(), tmp_Q.y(), tmp_Q.z());
    }
}

void pubKeyPoses(const Estimator &estimator, const std_msgs::Header &header)
{
    if (estimator.key_poses.empty())
        return;
    visualization_msgs::Marker key_poses;
    key_poses.header = header;
    key_poses.header.frame_id = "world";
    key_poses.ns = "key_poses";
    key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
    key_poses.action = visualization_msgs::Marker::ADD;
    key_poses.pose.orientation.w = 1.0;
    key_poses.lifetime = ros::Duration();

    //static int key_poses_id = 0;
    key_poses.id = 0; //key_poses_id++;
    key_poses.scale.x = 0.05;
    key_poses.scale.y = 0.05;
    key_poses.scale.z = 0.05;
    key_poses.color.r = 1.0;
    key_poses.color.a = 1.0;

    for (int i = 0; i <= kWindowSize; i++)
    {
        geometry_msgs::Point pose_marker;
        Vector3d correct_pose;
        correct_pose = estimator.key_poses[i];
        pose_marker.x = correct_pose.x();
        pose_marker.y = correct_pose.y();
        pose_marker.z = correct_pose.z();
        key_poses.points.push_back(pose_marker);
    }
    pub_key_poses.publish(key_poses);
}

void pubCameraPose(const Estimator &estimator, const std_msgs::Header &header)
{
    int idx2 = kWindowSize - 1;

    if (estimator.solver_flag == SolverFlag::kNonLinear)
    {
        int i = idx2;
        Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[0];
        Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[0]);

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

        pub_camera_pose.publish(odometry);

        cameraposevisual.reset();
        cameraposevisual.add_pose(P, R);
        if(cfg::is_stereo)
        {
            Vector3d P = estimator.Ps[i] + estimator.Rs[i] * estimator.tic[1];
            Quaterniond R = Quaterniond(estimator.Rs[i] * estimator.ric[1]);
            cameraposevisual.add_pose(P, R);
        }
        cameraposevisual.publish_by(pub_camera_pose_visual, odometry.header);
    }
}


void pubPointCloud(const Estimator &estimator, const std_msgs::Header &header)
{
    sensor_msgs::PointCloud point_cloud, loop_point_cloud;
    point_cloud.header = header;
    loop_point_cloud.header = header;


    for (auto &it_per_id : estimator.f_manager.feature)
    {
        int used_num = (int)it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < kWindowSize - 2))
            continue;
        if (it_per_id.start_frame > kWindowSize * 3.0 / 4.0 || it_per_id.solve_flag != 1)
            continue;
        int imu_i = it_per_id.start_frame;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
        Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

        geometry_msgs::Point32 p;
        p.x = (float)w_pts_i(0);
        p.y = (float)w_pts_i(1);
        p.z = (float)w_pts_i(2);
        point_cloud.points.push_back(p);
    }
    pub_point_cloud.publish(point_cloud);


    // pub margined potin
    sensor_msgs::PointCloud margin_cloud;
    margin_cloud.header = header;

    for (auto &it_per_id : estimator.f_manager.feature)
    { 
        int used_num = (int)it_per_id.feature_per_frame.size();
        if (!(used_num >= 2 && it_per_id.start_frame < kWindowSize - 2))
            continue;
        //if (it_per_id->start_frame > kWindowSize * 3.0 / 4.0 || it_per_id->solve_flag != 1)
        //        continue;

        if (it_per_id.start_frame == 0 && it_per_id.feature_per_frame.size() <= 2 
        && it_per_id.solve_flag == 1 )
        {
            int imu_i = it_per_id.start_frame;
            Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
            Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0]) + estimator.Ps[imu_i];

            geometry_msgs::Point32 p;
            p.x = (float)w_pts_i(0);
            p.y = (float)w_pts_i(1);
            p.z = (float)w_pts_i(2);
            margin_cloud.points.push_back(p);
        }
    }
    pub_margin_cloud.publish(margin_cloud);
}


void pubTF(const Estimator &estimator, const std_msgs::Header &header)
{
    if( estimator.solver_flag != SolverFlag::kNonLinear)
        return;
    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame
    Vector3d correct_t;
    Quaterniond correct_q;
    correct_t = estimator.Ps[kWindowSize];
    correct_q = estimator.Rs[kWindowSize];

    transform.setOrigin(tf::Vector3(correct_t(0),
                                    correct_t(1),
                                    correct_t(2)));
    q.setW(correct_q.w());
    q.setX(correct_q.x());
    q.setY(correct_q.y());
    q.setZ(correct_q.z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "world", "body"));

    // camera frame
    transform.setOrigin(tf::Vector3(estimator.tic[0].x(),
                                    estimator.tic[0].y(),
                                    estimator.tic[0].z()));
    q.setW(Quaterniond(estimator.ric[0]).w());
    q.setX(Quaterniond(estimator.ric[0]).x());
    q.setY(Quaterniond(estimator.ric[0]).y());
    q.setZ(Quaterniond(estimator.ric[0]).z());
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, header.stamp, "body", "camera"));

    
    nav_msgs::Odometry odometry;
    odometry.header = header;
    odometry.header.frame_id = "world";
    odometry.pose.pose.position.x = estimator.tic[0].x();
    odometry.pose.pose.position.y = estimator.tic[0].y();
    odometry.pose.pose.position.z = estimator.tic[0].z();
    Quaterniond tmp_q{estimator.ric[0]};
    odometry.pose.pose.orientation.x = tmp_q.x();
    odometry.pose.pose.orientation.y = tmp_q.y();
    odometry.pose.pose.orientation.z = tmp_q.z();
    odometry.pose.pose.orientation.w = tmp_q.w();
    pub_extrinsic.publish(odometry);

}

void pubKeyframe(const Estimator &estimator)
{
    // pub camera pose, 2D-3D points of keyframe
    if (estimator.solver_flag == SolverFlag::kNonLinear && estimator.margin_flag == MarginFlag::kMarginOld)
    {
        int i = kWindowSize - 2;
        //Vector3d P = e.Ps[i] + e.Rs[i] * e.tic[0];
        Vector3d P = estimator.Ps[i];
        auto R = Quaterniond(estimator.Rs[i]);

        nav_msgs::Odometry odometry;
        odometry.header.stamp = ros::Time(estimator.headers[kWindowSize - 2]);
        odometry.header.frame_id = "world";
        odometry.pose.pose.position.x = P.x();
        odometry.pose.pose.position.y = P.y();
        odometry.pose.pose.position.z = P.z();
        odometry.pose.pose.orientation.x = R.x();
        odometry.pose.pose.orientation.y = R.y();
        odometry.pose.pose.orientation.z = R.z();
        odometry.pose.pose.orientation.w = R.w();
        //printf("time: %f t: %f %f %f r: %f %f %f %f\n", odometry.header.stamp.toSec(), P.x(), P.y(), P.z(), R.w(), R.x(), R.y(), R.z());

        pub_keyframe_pose.publish(odometry);


        sensor_msgs::PointCloud point_cloud;
        point_cloud.header.stamp = ros::Time(estimator.headers[kWindowSize - 2]);
        point_cloud.header.frame_id = "world";
        for (auto &it_per_id : estimator.f_manager.feature)
        {
            int frame_size = (int)it_per_id.feature_per_frame.size();
            if(it_per_id.start_frame < kWindowSize - 2 && it_per_id.start_frame + frame_size - 1 >= kWindowSize - 2 && it_per_id.solve_flag == 1)
            {

                int imu_i = it_per_id.start_frame;
                Vector3d pts_i = it_per_id.feature_per_frame[0].point * it_per_id.estimated_depth;
                Vector3d w_pts_i = estimator.Rs[imu_i] * (estimator.ric[0] * pts_i + estimator.tic[0])
                        + estimator.Ps[imu_i];
                geometry_msgs::Point32 p;
                p.x = (float)w_pts_i(0);
                p.y = (float)w_pts_i(1);
                p.z = (float)w_pts_i(2);
                point_cloud.points.push_back(p);

                int imu_j = kWindowSize - 2 - it_per_id.start_frame;
                sensor_msgs::ChannelFloat32 p_2d;
                p_2d.values.push_back(float(it_per_id.feature_per_frame[imu_j].point.x()));
                p_2d.values.push_back(float(it_per_id.feature_per_frame[imu_j].point.y()));
                p_2d.values.push_back(float(it_per_id.feature_per_frame[imu_j].uv.x()));
                p_2d.values.push_back(float(it_per_id.feature_per_frame[imu_j].uv.y()));
                p_2d.values.push_back(float(it_per_id.feature_id));
                point_cloud.channels.push_back(p_2d);
            }

        }
        pub_keyframe_point.publish(point_cloud);
    }
}



void pubInstancePointCloud( Estimator &estimator, const std_msgs::Header &header)
{
    if(estimator.insts_manager.tracking_number() < 1)
        return;

    visualization_msgs::MarkerArray markers;
    PointCloud pointCloud;

    for(auto &[key,inst] : estimator.insts_manager.instances){
        if(!inst.is_tracking || !inst.is_initial){
            continue;
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
        //vio_logger->debug("inst:{} point3d_curr num:{}",inst.id,inst.point3d_curr.size());

        PointCloud cloud;
        for(auto &pt : inst.point3d_curr){
            PointT p;
            p.x = (float)pt(0);p.y = (float)pt(1);p.z = (float)pt(2);
            p.r=(uint8_t)inst.color[2];p.g=(uint8_t)inst.color[1];p.b=(uint8_t)inst.color[0];
            cloud.push_back(p);
            //DebugV("({})", VecToStr(pt));
        }

        /*
        PointT minPt,maxPt;
        pcl::getMinMax3D(cloud, box_min_pt, box_max_pt);
        double x_center=(box_min_pt.x+box_max_pt.x)/2.0;
        double y_center=(box_min_pt.y+box_max_pt.y)/2.0;
        double z_center=(box_min_pt.z+box_max_pt.z)/2.0;
        double x_radius=abs(box_min_pt.x-box_max_pt.x)/2.0;
        double y_radius=abs(box_min_pt.y-box_max_pt.y)/2.0;
        double z_radius=abs(box_min_pt.z-box_max_pt.z)/2.0;

        double x_center=inst.state[kWindowSize].P.x();
        double y_center=inst.state[kWindowSize].P.y();
        double z_center=inst.state[kWindowSize].P.z();
        box_min_pt.x=x_center-0.5;box_min_pt.y=y_center-0.5;box_min_pt.z=z_center-0.5;
        box_max_pt.x=x_center+0.5;box_max_pt.y=y_center+0.5;box_max_pt.z=z_center+0.5;
         minPt.x=inst.state[kWindowSize].P.x() - inst.box.x();
        minPt.y=inst.state[kWindowSize].P.y() - inst.box.y();
        minPt.z=inst.state[kWindowSize].P.z() - inst.box.z();
        maxPt.x=inst.state[kWindowSize].P.x() + inst.box.x();
        maxPt.y=inst.state[kWindowSize].P.y() + inst.box.y();
        maxPt.z=inst.state[kWindowSize].P.z() + inst.box.z();
         */

        //各个时刻的位姿
        for(int i=0; i <= kWindowSize; i++){
            auto text_marker = BuildTextMarker(inst.state[i].P,i,to_string(i),color_transparent);
            markers.markers.push_back(text_marker);
        }

        //auto lineStripMarker = BuildLineStripMarker(maxPt,minPt,key,color_norm);
        EigenContainer<Eigen::Vector3d> vertex;
        inst.GetBoxVertex(vertex);
        auto lineStripMarker = BuildLineStripMarker(vertex,key,color_norm);

        //计算可视化的速度
        Eigen::Vector3d vel = Hat(inst.vel.a) * inst.state[0].P + inst.vel.v;
        auto text=fmt::format("{}\n({})", inst.id, VecToStr(vel));
        auto textMarker = BuildTextMarker(inst.state[kWindowSize].P, key, text, color_inv, 1.2);

        Eigen::Vector3d end= inst.state[kWindowSize].P + vel.normalized() * 4;
        auto arrowMarker = BuildArrowMarker(inst.state[kWindowSize].P, end, key, color_norm);

        markers.markers.push_back(lineStripMarker);
        markers.markers.push_back(textMarker);
        markers.markers.push_back(arrowMarker);

        pointCloud+=cloud;
    }

    pub_instance_marker.publish(markers);
    printf("实例点云的数量: %ld\n",pointCloud.size());

    sensor_msgs::PointCloud2 point_cloud_msg;
    pcl::toROSMsg(pointCloud,point_cloud_msg);
    point_cloud_msg.header = header;
    pub_instance_pointcloud.publish(point_cloud_msg);
}





void printInstanceData(const Estimator &estimator)
{
    if(estimator.insts_manager.tracking_number() < 1)
        return;

    printf("实例数量:%d\n",estimator.insts_manager.tracking_number());
    for(auto &pair : estimator.insts_manager.instances){
        auto &inst = pair.second;
        auto &key=pair.first;

        if(!inst.is_tracking)
            continue;

        int num_triangular=0,num_feat=0;
        for(auto &landmark : inst.landmarks){
            num_feat+=landmark.feats.size();
            if(landmark.depth > 0)
                num_triangular++;
        }

        printf("Instance:%d is_initial:%d is_tracking:%d feats:%d landmarks:%ld num_triangle:%d | ",
               key, inst.is_initial, inst.is_tracking, num_feat, inst.landmarks.size(), num_triangular);

        int cnt[kWindowSize + 1]={0};
        for(auto &landmark : inst.landmarks){
            cnt[landmark.feats[0].frame]++;
        }
        for(int i=0; i <= kWindowSize; ++i){
            printf("%d:%d  ",i,cnt[i]);
        }

        printf("\nInst:%d ",inst.id);
        for(int i=0; i <= kWindowSize; ++i){
            printf("%d:(%.2lf,%.2lf,%.2lf)  ",i,inst.state[i].P.x(),inst.state[i].P.y(),inst.state[i].P.z());
        }

        printf("\n");
    }

}


void printInstancePose(Instance &inst)
{
    if(!inst.is_tracking || !inst.is_initial){
        return;
    }
    //printf("Inst:%d | ",inst.id);
    for(int i=0; i <= kWindowSize; ++i){
        //Eigen::Quaterniond q(inst.state[i].R);
        //printf("%d:<%.2lf,%.2lf,%.2lf | %.2lf,%.2lf,%.2lf,%.2lf> ",i,inst.state[i].P.x(),inst.state[i].P.y(),inst.state[i].P.z(),q.x(),q.y(),q.z(),q.w());
        printf("%d:<%.2lf,%.2lf,%.2lf> ",i,inst.state[i].P.x(),inst.state[i].P.y(),inst.state[i].P.z());
        //if(i==5) printf("\n  ");
    }
    printf("\n");
}

void printInstanceDepth(Instance &inst)
{
    if(!inst.is_tracking || !inst.is_initial)
        return;
    //printf("Inst:%d | ",inst.id);

    for(auto& landmark : inst.landmarks)
        if(landmark.depth > 0.)
            printf("<lid:%d:d:%.2lf> ",landmark.id,landmark.depth);
        printf("\n");
}


visualization_msgs::Marker BuildLineStripMarker(PointT &maxPt,PointT &minPt,int id,const cv::Scalar &color)
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

    return BuildLineStripMarker(p,id,color);
}


visualization_msgs::Marker BuildLineStripMarker(EigenContainer<Eigen::Vector3d> &p,int id,const cv::Scalar &color)
{
    geometry_msgs::Point points[8];
    for(int i=0;i<8;++i){
        points[i].x = p[i].x();
        points[i].y = p[i].y();
        points[i].z = p[i].z();
    }
    return BuildLineStripMarker(points,id,color);
}



visualization_msgs::Marker BuildLineStripMarker(geometry_msgs::Point p[8],int id,const cv::Scalar &color)
{
    visualization_msgs::Marker msg;
    msg.header.stamp=ros::Time::now();
    msg.header.frame_id="world";
    msg.ns="box_strip";
    msg.action=visualization_msgs::Marker::ADD;
    msg.pose.orientation.w=1.0;

    //暂时使用类别代替这个ID
    msg.id=id * MarkerTypeNumber + 0;//当存在多个marker时用于标志出来
    msg.lifetime=ros::Duration(cfg::kVisualInstDuration);//持续时间，若为ros::Duration()表示一直持续

    msg.type=visualization_msgs::Marker::LINE_STRIP;//marker的类型
    msg.scale.x=0.01;//线宽
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


visualization_msgs::Marker BuildTextMarker(const Eigen::Vector3d &point,int id,const std::string &text,const cv::Scalar &color,const double scale){
    visualization_msgs::Marker msg;
    msg.header.stamp=ros::Time::now();
    msg.header.frame_id="world";
    msg.ns="box_text";
    msg.action=visualization_msgs::Marker::ADD;

    //暂时使用类别代替这个ID
    msg.id=id * MarkerTypeNumber + 1;//当存在多个marker时用于标志出来
    msg.lifetime=ros::Duration(cfg::kVisualInstDuration);//持续时间4s，若为ros::Duration()表示一直持续

    msg.type=visualization_msgs::Marker::TEXT_VIEW_FACING;//marker的类型
    msg.scale.z=scale;//字体大小
    msg.color.r=(float)color[2];msg.color.g=(float)color[1];msg.color.b=(float)color[0];
    msg.color.a=(float)color[3];//不透明度

    geometry_msgs::Pose pose;
    pose.position.x=point.x();pose.position.y=point.y();pose.position.z=point.z();
    pose.orientation.w=1.0;
    msg.pose=pose;

    msg.text=text.c_str();

    return msg;
}



visualization_msgs::Marker  BuildTextMarker(const PointT &point,int id,const std::string &text,const cv::Scalar &color,const double scale)
{
    Eigen::Vector3d eigen_pt;
    eigen_pt<<point.x,point.y,point.z;
    return BuildTextMarker(eigen_pt,id,text,color,scale);
}

visualization_msgs::Marker BuildArrowMarker(const Eigen::Vector3d &start_pt,const Eigen::Vector3d &end_pt,int id,const cv::Scalar &color)
{
    visualization_msgs::Marker msg;
    msg.header.frame_id="world";
    msg.header.stamp=ros::Time::now();
    msg.ns="arrow_strip";
    msg.action=visualization_msgs::Marker::ADD;
    msg.pose.orientation.w=1.0;

    //暂时使用类别代替这个ID
    msg.id=id * MarkerTypeNumber + 2;//当存在多个marker时用于标志出来
    msg.lifetime=ros::Duration(cfg::kVisualInstDuration);

    msg.type=visualization_msgs::Marker::LINE_STRIP;//marker的类型
    msg.scale.x=0.01;//线宽
    msg.color.r=(float)color[2];msg.color.g=(float)color[1];msg.color.b=(float)color[0];
    msg.color.a=1.0;//不透明度

    geometry_msgs::Point start,end;
    start.x=start_pt.x();start.y=start_pt.y();start.z=start_pt.z();
    end.x=end_pt.x();end.y=end_pt.y();end.z=end_pt.z();

    const double arrow_len=std::sqrt((start.x-end.x) * (start.x-end.x) +(start.y-end.y) * (start.y-end.y) + (start.z-end.z) * (start.z-end.z)) /8.;

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



}
