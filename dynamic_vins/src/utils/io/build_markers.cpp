/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "build_markers.h"

#include "utils/io_utils.h"
#include "io_parameters.h"

namespace dynamic_vins{\

int32_t kMarkerTypeNumber=10;


Marker LineStripMarker(EigenContainer<Eigen::Vector3d> &p, unsigned int id, const cv::Scalar &color,
                       double scale, Marker::_action_type action, const string &ns,int offset){
    geometry_msgs::Point points[8];
    for(int i=0;i<8;++i){
        points[i] = EigenToGeometryPoint(p[i]);
    }
    return LineStripMarker(points, id, color, scale, action, ns,offset);
}


Marker LineStripMarker(geometry_msgs::Point p[8], unsigned int id, const cv::Scalar &color,
                       double scale, Marker::_action_type action,const string &ns, int offset){
    Marker msg;
    msg.header.stamp=ros::Time::now();
    msg.header.frame_id="world";
    msg.ns=ns;
    msg.action=action;
    msg.pose.orientation.w=1.0;

    msg.id=id * kMarkerTypeNumber + offset;
    msg.type=Marker::LINE_STRIP;//marker的类型

    if(action==Marker::DELETE){
        return msg;
    }

    msg.lifetime=ros::Duration(io_para::kVisualInstDuration);//持续时间，若为ros::Duration()表示一直持续
    msg.scale.x=scale;//线宽
    msg.color = ScalarBgrToColorRGBA(color);

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


Marker LineMarker(const Eigen::Vector3d &p1,const Eigen::Vector3d &p2,unsigned int id, const cv::Scalar &color,
                  double scale, Marker::_action_type action, const string &ns,int offset){

    Marker msg;
    msg.header.stamp=ros::Time::now();
    msg.header.frame_id="world";
    msg.ns=ns;
    msg.action=action;
    msg.pose.orientation.w=1.0;

    msg.id=id * kMarkerTypeNumber + offset;
    msg.type=Marker::LINE_STRIP;//marker的类型

    if(action==Marker::DELETE){
        return msg;
    }

    msg.lifetime=ros::Duration(io_para::kVisualInstDuration);//持续时间，若为ros::Duration()表示一直持续
    msg.scale.x=scale;//线宽
    msg.scale.y=scale;//线宽
    msg.scale.z=scale;//线宽
    msg.color = ScalarBgrToColorRGBA(color);

    geometry_msgs::Point pw1 = EigenToGeometryPoint(p1);
    geometry_msgs::Point pw2 = EigenToGeometryPoint(p2);
    msg.points.push_back(pw1);
    msg.points.push_back(pw2);

    return msg;
}


Marker TextMarker(const Eigen::Vector3d &point, unsigned int id, const std::string &text, const cv::Scalar &color,
                  double scale, Marker::_action_type action, const string &ns,int offset){
    Marker msg;
    msg.header.stamp=ros::Time::now();
    msg.header.frame_id="world";
    msg.ns=ns;
    msg.action=action;

    msg.id=id * kMarkerTypeNumber + offset;
    msg.type=Marker::TEXT_VIEW_FACING;//marker的类型
    if(action==Marker::DELETE){
        return msg;
    }

    msg.lifetime=ros::Duration(io_para::kVisualInstDuration);//持续时间4s，若为ros::Duration()表示一直持续

    msg.scale.z=scale;//字体大小
    msg.color = ScalarBgrToColorRGBA(color);

    msg.pose.position = EigenToGeometryPoint(point);
    msg.pose.orientation.w=1.0;

    msg.text=text;

    return msg;
}


Marker TextMarker(const PointT &point, unsigned int id, const std::string &text,
                  const cv::Scalar &color, double scale, Marker::_action_type action,
                  const string &ns,int offset){
    Eigen::Vector3d eigen_pt;
    eigen_pt<<point.x,point.y,point.z;
    return TextMarker(eigen_pt, id, text, color, scale, action, ns,offset);
}

Marker ArrowMarker(const Eigen::Vector3d &start_pt, const Eigen::Vector3d &end_pt, unsigned int id,
                   const cv::Scalar &color, double scale, Marker::_action_type action,
                   const string &ns,int offset){

    Marker msg_x;
    msg_x.header.frame_id="world";
    msg_x.header.stamp=ros::Time::now();
    msg_x.ns=ns;
    msg_x.action=action;
    msg_x.type = Marker::ARROW;
    msg_x.lifetime=ros::Duration(io_para::kVisualInstDuration);//若为ros::Duration()表示一直持续
    msg_x.pose.orientation.w=1.0;
    msg_x.scale.x=scale;//线宽
    msg_x.scale.y=scale;
    msg_x.scale.z=scale;

    msg_x.id=id * kMarkerTypeNumber + offset;
    if(action==Marker::DELETE){
        return msg_x;
    }

    msg_x.points.push_back(EigenToGeometryPoint(start_pt));
    msg_x.points.push_back(EigenToGeometryPoint(end_pt));
    msg_x.color = ScalarBgrToColorRGBA(color);

    return msg_x;
}


Marker CubeMarker(Mat38d &corners, unsigned int id, const cv::Scalar &color, double scale,
                  Marker::_action_type action,const string &ns ,int offset){
    Marker msg;

    msg.header.frame_id="world";
    msg.header.stamp=ros::Time::now();
    msg.ns=ns;
    msg.action=action;

    msg.id=id * kMarkerTypeNumber + offset;
    msg.type=Marker::LINE_STRIP;//marker的类型
    if(action==Marker::DELETE){
        return msg;
    }

    msg.lifetime=ros::Duration(io_para::kVisualInstDuration);//持续时间3s，若为ros::Duration()表示一直持续

    msg.pose.orientation.w=1.0;

    msg.scale.x=scale;//线宽
    msg.color = ScalarBgrToColorRGBA(color);

    //设置立方体的八个顶点
    geometry_msgs::Point p[8];
    p[0] = EigenToGeometryPoint(corners.col(0));
    p[1] = EigenToGeometryPoint(corners.col(1));
    p[2] = EigenToGeometryPoint(corners.col(2));
    p[3] = EigenToGeometryPoint(corners.col(3));
    p[4] = EigenToGeometryPoint(corners.col(4));
    p[5] = EigenToGeometryPoint(corners.col(5));
    p[6] = EigenToGeometryPoint(corners.col(6));
    p[7] = EigenToGeometryPoint(corners.col(7));

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



std::tuple<Marker,Marker,Marker> AxisMarker(Mat34d &axis, unsigned int id, Marker::_action_type action,const string &ns, int offset)
{
    Marker msg_x;
    msg_x.header.frame_id="world";
    msg_x.header.stamp=ros::Time::now();
    msg_x.ns=ns;
    msg_x.action=action;
    msg_x.type = Marker::ARROW;
    msg_x.lifetime=ros::Duration(io_para::kVisualInstDuration);//若为ros::Duration()表示一直持续
    msg_x.pose.orientation.w=1.0;
    msg_x.scale.x=0.1;//线宽
    msg_x.scale.y=0.1;
    msg_x.scale.z=0.1;

    Marker msg_y= msg_x;
    Marker msg_z= msg_x;

    msg_x.id=id * kMarkerTypeNumber + offset;
    msg_y.id=id * kMarkerTypeNumber + offset + 1;
    msg_z.id=id * kMarkerTypeNumber + offset + 2;
    if(action==Marker::DELETE){
        return {msg_x,msg_y,msg_z};
    }

    geometry_msgs::Point org = EigenToGeometryPoint(axis.col(0));
    geometry_msgs::Point x_d = EigenToGeometryPoint(axis.col(1));
    geometry_msgs::Point y_d = EigenToGeometryPoint(axis.col(2));
    geometry_msgs::Point z_d = EigenToGeometryPoint(axis.col(3));

    msg_x.points.push_back(org);
    msg_x.points.push_back(x_d);
    msg_x.color = ScalarBgrToColorRGBA(BgrColor("red"));
    msg_y.points.push_back(org);
    msg_y.points.push_back(y_d);
    msg_y.color = ScalarBgrToColorRGBA(BgrColor("green"));
    msg_z.points.push_back(org);
    msg_z.points.push_back(z_d);
    msg_z.color = ScalarBgrToColorRGBA(BgrColor("blue"));

    return {msg_x,msg_y,msg_z};
}



}
