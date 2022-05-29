/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "box3d.h"



namespace dynamic_vins{\

using namespace std;
namespace fs=std::filesystem;


Box3D::Box3D(vector<string> &tokens)
{
    ///每行的前3个数字是类别,属性,分数
    class_id=std::stoi(tokens[0]);
    attribution_id = std::stoi(tokens[1]);
    //score = std::stod(tokens[2]);
    score=1.;

    ///3-5个数字是物体包围框底部的中心
    bottom_center<<std::stod(tokens[3]),std::stod(tokens[4]),std::stod(tokens[5]);
    ///6-8数字是物体在x,y,z轴上的大小
    dims<<std::stod(tokens[6]),std::stod(tokens[7]),std::stod(tokens[8]);
    ///9个yaw角(绕着y轴,因为y轴是垂直向下的)
     yaw=std::stod(tokens[9]);

     Eigen::Matrix<double,8,3> corners_norm;
     corners_norm << 0,0,0,  0,0,1,  0,1,1,  0,1,0,  1,0,0,  1,0,1,  1,1,1,  1,1,0;
     Eigen::Vector3d offset(0.5,1,0.5);//预测结果所在的坐标系与相机坐标系之间的偏移
     corners_norm = corners_norm.array().rowwise() - offset.transpose().array();//将每个坐标减去偏移量
     corners = corners_norm.transpose(); //得到矩阵 3x8
    corners = corners.array().colwise() * dims.array();//广播逐点乘法

    ///根据yaw角构造旋转矩阵
    Mat3d R;
    R<<cos(yaw),0, -sin(yaw),   0,1,0,   sin(yaw),0,cos(yaw);

    Eigen::Matrix<double,8,3> result =  corners.transpose() * R;//8x3
    //加上偏移量
    Eigen::Matrix<double,8,3> output= result.array().rowwise() + bottom_center.transpose().array();
    corners = output.transpose();///box的8个顶点在相机坐标系下的坐标
    ///计算3D box投影到图像平面
    for(int i=0;i<8;++i){
        Vec2d p;
        cam0->ProjectPoint(corners.col(i),p);
        corners_2d.col(i) = p;
    }
    Vec2d corner2d_min_pt = corners_2d.rowwise().minCoeff();
    Vec2d corner2d_max_pt = corners_2d.rowwise().maxCoeff();
    box2d.min_pt = cv::Point2f(corner2d_min_pt.x(),corner2d_min_pt.y());
    box2d.max_pt = cv::Point2f(corner2d_max_pt.x(),corner2d_max_pt.y());
    box2d.center_pt = (box2d.min_pt + box2d.max_pt) / 2;

    ///计算包围框中心坐标
    center = (output.row(0)+output.row(6)).transpose()  /2;

}



/**
 * 判断点是否在3D box之内
 * @param point
 * @return
 */
bool Box3D::InsideBox(Eigen::Vector3d &point)
{
        /**
                 .. code-block:: none

                                 front z
                                        /
                                       /
                       p1(x0, y0, z1) + -----------  + p5(x1, y0, z1)
                                     /|            / |
                                    / |           /  |
                    p0(x0, y0, z0) + ---------p4 +   + p6(x1, y1, z1)
                                   |  /      .   |  /
                                   | / origin    | /
                    p3(x0, y1, z0) + ----------- + -------> x right
                                   |             p7(x1, y1, z0)
                                   |
                                   v
                            down y
         输入的点序列:p0:0,0,0, p1: 0,0,1,  p2: 0,1,1,  p3: 0,1,0,  p4: 1,0,0,  p5: 1,0,1,  p6: 1,1,1,  p7: 1,1,0;
         */


        //构建p0p3p4平面法向量,同时也是p1p2p5的法向量
        Vec3d p0p3 = corners.col(3) - corners.col(0);
        Vec3d p0p4 = corners.col(4) - corners.col(0);
        Vec3d p0p3p4_n = p0p3.cross(p0p4);//根据右手定则,法向量指向左边

        Vec3d pxp0 = corners.col(0) - point;
        Vec3d pxp1 = corners.col(1) - point;

        //向量内积公式: a.b = |a| |b| cos(theta),若a.b>0,则角度在0-90,若a.b<0,则角度在90-180度
        double direction_0 = p0p3p4_n.dot(pxp0);
        double direction_1 = p0p3p4_n.dot(pxp1);
        if((direction_0>0 && direction_1>0) ||(direction_0<0 && direction_1<0)){ //方向一致,表明不在box内
            return false;
        }

        //构建p0p1p3平面法向量,同时也是p4p5p7的法向量
        Vec3d p0p1 = corners.col(1) - corners.col(0);
        Vec3d p0p1p3_n = p0p1.cross(p0p3);
        Vec3d pxp4 = corners.col(4) - point;

        //向量内积公式: a.b = |a| |b| cos(theta),若a.b>0,则角度在0-90,若a.b<0,则角度在90-180度
        double direction_2 = p0p1p3_n.dot(pxp0);
        double direction_3 = p0p1p3_n.dot(pxp4);
        if((direction_2>0 && direction_3>0) ||(direction_2<0 && direction_3<0)){ //方向一致,表明不在box内
            return false;
        }

        Vec3d p0p1p4_n = p0p1.cross(p0p4);
        Vec3d pxp3 = corners.col(3) - point;
        double direction_4 = p0p1p4_n.dot(pxp0);
        double direction_5 = p0p1p4_n.dot(pxp3);
        if((direction_4>0 && direction_5>0) ||(direction_4<0 && direction_5<0)){ //方向一致,表明不在box内
            return false;
        }

        return true;
}


/**
 * 根据世界坐标系下的8个3D 角点,构建物体坐标系,返回坐标系3各轴的向量
 * @param corners
 * @return
 */
VecVector3d Box3D::GetCoordinateVectorFromCorners(Mat38d &corners)
{
    /**
         .. code-block:: none

                         front z
                                /
                               /
               p1(x0, y0, z1) + -----------  + p5(x1, y0, z1)
                             /|            / |
                            / |           /  |
            p0(x0, y0, z0) + ---------p4 +   + p6(x1, y1, z1)
                           |  /      .   |  /
                           | / origin    | /
            p3(x0, y1, z0) + ----------- + -------> x right
                           |             p7(x1, y1, z0)
                           |
                           v
                    down y
 输入的点序列:p0:0,0,0, p1: 0,0,1,  p2: 0,1,1,  p3: 0,1,0,  p4: 1,0,0,  p5: 1,0,1,  p6: 1,1,1,  p7: 1,1,0;

     下面定义坐标系的原点位于物体中心, z轴垂直向上, x轴不变,y轴为原来的z轴
 */
    VecVector3d axis_vector(3);
    axis_vector[0] = (corners.col(4) - corners.col(0) ).normalized();//x轴,与上图一致
    axis_vector[1] = (corners.col(1) - corners.col(0) ).normalized();//y轴,上图的z轴
    axis_vector[2] = (corners.col(0) - corners.col(3) ).normalized();//z轴,上图的 -y轴
    return axis_vector;
}

/**
 * 根据世界坐标系下的8个3D 角点,构建物体坐标系,旋转矩阵
 * @param corners
 * @return
 */
Mat3d Box3D::GetCoordinateRotationFromCorners(Mat38d &corners){
    Mat3d R;
    R.col(0) = (corners.col(4) - corners.col(0) ).normalized();
    R.col(1) = (corners.col(1) - corners.col(0) ).normalized();
    R.col(2) = (corners.col(0) - corners.col(3) ).normalized();
    return R;
}


/**
 * 根据物体的位姿,和大小,生成物体的包围框的8个顶点
 * @param R_woi
 * @param P_woi
 * @param dims
 * @return
 */
Mat38d Box3D::GetCornersFromPose(Mat3d &R_woi,Vec3d &P_woi,Vec3d &dims)
{
    /**
         .. code-block:: none

                         front z
                                /
                               /
               p1(x0, y0, z1) + -----------  + p5(x1, y0, z1)
                             /|            / |
                            / |           /  |
            p0(x0, y0, z0) + ---------p4 +   + p6(x1, y1, z1)
                           |  /      .   |  /
                           | / origin    | /
            p3(x0, y1, z0) + ----------- + -------> x right
                           |             p7(x1, y1, z0)
                           |
                           v
                    down y
 输入的点序列:p0:0,0,0, p1: 0,0,1,  p2: 0,1,1,  p3: 0,1,0,  p4: 1,0,0,  p5: 1,0,1,  p6: 1,1,1,  p7: 1,1,0;

     定义坐标系的原点位于物体中心, z轴垂直向上, x轴不变,y轴为原来的z轴
    */

    Mat38d corners;
    corners.col(0) << -dims.x()/2., -dims.y()/2.,dims.z()/2.;
    corners.col(1) << -dims.x()/2., dims.y()/2., dims.z()/2.;
    corners.col(2) <<-dims.x()/2., dims.y()/2., -dims.z()/2.;
    corners.col(3) << -dims.x()/2., -dims.y()/2.,-dims.z()/2.;
    corners.col(4) << dims.x()/2., -dims.y()/2.,dims.z()/2.;
    corners.col(5) << dims.x()/2., dims.y()/2.,dims.z()/2.;
    corners.col(6) << dims.x()/2., dims.y()/2.,-dims.z()/2.;
    corners.col(7) << dims.x()/2., -dims.y()/2.,-dims.z()/2.;

    for (int i = 0; i < 8; ++i) {
        corners.col(i) = R_woi * corners.col(i) + P_woi;
    }

    return corners;
}



/**
 * 根据输入的物体坐标系区域,返回包围框顶点是哪个顶点. 关于物体坐标系的定义,参考 GetCoordinateVectorFromCorners()
 * @param x_d in {-1,1},表示该顶点在物体坐标系x轴的哪个部分,是大于0 (1) 还是小于0 (-1)
 * @param y_d in {-1,1}
 * @param z_d in {-1,1}
 * @return 顶点的索引
 */
 int Box3D::CoordinateDirection(int x_d,int y_d,int z_d){
    if(x_d > 0){
        if(y_d > 0){
            if(z_d > 0){ // x_d>0, y_d>0, z_d>0
                return 5;//p5
            }
            else{ // x_d>0, y_d>0, z_d<0
                return 6;
            }
        }
        else{
            if(z_d > 0){ // x_d>0, y_d<0, z_d>0
                return 4;
            }
            else{ // x_d>0, y_d<0, z_d<0
                return 7;
            }
        }
    }
    else{
        if(y_d > 0){
            if(z_d > 0){ // x_d<0, y_d>0, z_d>0
                return 1;//p5
            }
            else{ // x_d<0, y_d>0, z_d<0
                return 2;
            }
        }
        else{
            if(z_d > 0){ // x_d<0, y_d<0, z_d>0
                return 0;
            }
            else{ // x_d<0, y_d<0, z_d<0
                return 3;
            }
        }
    }


}


/**
 * 为了画出cube,顶点之间的连线
 * @return
 */
vector<std::pair<int,int>> Box3D::GetLineVetexPair(){
    /**
         .. code-block:: none

                         front z
                                /
                               /
               p1(x0, y0, z1) + -----------  + p5(x1, y0, z1)
                             /|            / |
                            / |           /  |
            p0(x0, y0, z0) + ---------p4 +   + p6(x1, y1, z1)
                           |  /      .   |  /
                           | / origin    | /
            p3(x0, y1, z0) + ----------- + -------> x right
                           |             p7(x1, y1, z0)
                           |
                           v
                    down y
 输入的点序列:p0:0,0,0, p1: 0,0,1,  p2: 0,1,1,  p3: 0,1,0,  p4: 1,0,0,  p5: 1,0,1,  p6: 1,1,1,  p7: 1,1,0;

     下面定义坐标系的原点位于物体中心, z轴垂直向上, x轴不变,y轴为原来的z轴
 */
    return {{0,1},{1,5},{5,4},{4,0},
            {3,7},{7,6},{6,2},{2,3},
            {0,3},{4,7},{5,6},{1,2}};
}

/**
 * 将3D角点投影到2D角点
 * @param cam
 * @return
 */
Mat28d Box3D::CornersProjectTo2D(PinHoleCamera &cam)
{
    Mat28d corners_2d;
    for(int i=0;i<8;++i){
        Vec2d p;
        cam.ProjectPoint(corners.col(i),p);
        corners_2d.col(i) = p;
    }
    return corners_2d;
}

/**
 * 获得世界坐标系下的包围框的顶点,
 * @param R_wbi
 * @param P_wbi
 * @param R_bc
 * @param P_bc
 */
Mat38d Box3D::GetCornersInWorld(const Mat3d &R_wbi,const Vec3d &P_wbi,const Mat3d &R_bc,const Vec3d &P_bc){
    Mat38d corners_w;
    for(int i=0;i<8;++i){
        corners_w.col(i) = R_wbi * (R_bc * corners.col(i) + P_bc) + P_wbi;
    }
    return corners_w;
}



/**
 * 在图像上绘制box
 * @param img
 * @param color
 * @param cam
 */
void Box3D::VisCorners2d(cv::Mat &img,const cv::Scalar& color,PinHoleCamera &cam){
    Mat28d corners2d = CornersProjectTo2D(cam);
    vector<std::pair<int,int>> lines = GetLineVetexPair();

    for(auto line : lines){
        cv::line(img,cv::Point2f(corners2d(0,line.first),corners2d(1,line.first)),
                 cv::Point2f(corners2d(0,line.second),corners2d(1,line.second)),color,2);
    }
}




}
