/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "box3d.h"



namespace dynamic_vins{\

using namespace std;
namespace fs=std::filesystem;

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



VecVector3d Box3D::GetCoordinateVectorFromCorners(Eigen::Matrix<double,3,8> &corners)
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
    axis_vector[0] = (corners.col(4) - corners.col(0) ).normalized();//x轴
    axis_vector[1] = (corners.col(1) - corners.col(0) ).normalized();//y轴
    axis_vector[2] = (corners.col(0) - corners.col(3) ).normalized();//z轴
    return axis_vector;
}


Eigen::Matrix<double,2,8> Box3D::CornersProjectTo2D(PinHoleCamera &cam)
{
    Eigen::Matrix<double,2,8> corners_2d;
    for(int i=0;i<8;++i){
        Vec2d p;
        cam.ProjectPoint(corners.col(i),p);
        corners_2d.col(i) = p;
    }
    return corners_2d;
}





}
