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
#include "utils/dataset/nuscenes_utils.h"
#include "utils/dataset/kitti_utils.h"


namespace dynamic_vins{\

using namespace std;
namespace fs=std::filesystem;


Box3D::Ptr Box3D::Box3dFromFCOS3D(vector<string> &tokens,camodocal::CameraPtr &cam)
{
    ///每行的前3个数字是类别,属性,分数
    int class_id = NuScenes::ConvertNuScenesToKitti(std::stoi(tokens[0]));
    string class_name = kitti::GetKittiName(class_id) ;
    int attribution_id = std::stoi(tokens[1]);
    double score = std::stod(tokens[2]);
    //score=1.;

    Box3D::Ptr box3d = std::make_shared<Box3D>(class_id,class_name,attribution_id,score);


    ///3-5个数字是物体包围框底部的中心
    box3d->bottom_center<<std::stod(tokens[3]),std::stod(tokens[4]),std::stod(tokens[5]);
    ///6-8数字是物体在x,y,z轴上的大小. 表示将物体旋转到yaw=0时(包围框的坐标系与相机坐标系对齐),物体在各轴上的大小
    box3d->dims<<std::stod(tokens[6]),std::stod(tokens[7]),std::stod(tokens[8]);

    //cout<<fmt::format("class_id:{} type:{} \n{}",class_id,class_name,EigenToStr(box3d->corners))<<endl;

    /**
                z front (yaw=-0.5*pi)
               /
              /
             0 ------> x right (yaw=0)
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is (0.5, 1.0, 0.5),
    and the yaw is around the y axis, thus the rotation axis=1.
    The yaw is 0 at the positive direction of x axis, and decreases from
    the positive direction of x to the positive direction of z.
     */
    ///9个yaw角(绕着y轴,因为y轴是垂直向下的)
    double yaw=std::stod(tokens[9]);
    //将yaw角限制到[-2pi,0]范围
    while(yaw>0){
        yaw -= (2*M_PI);
    }
    while(yaw< (-2*M_PI)){
        yaw += (2*M_PI);
    }
    //将yaw角限定在[-pi,0]上
    if(yaw < (-M_PI)){
        yaw += M_PI;
    }
    box3d->yaw = yaw;

    ///根据yaw角构造旋转矩阵
    Mat3d R_co = box3d->R_cioi();
    box3d->corners = Box3D::GetCorners(box3d->dims,R_co,box3d->bottom_center);

    for(int i=0;i<8;++i){
        Vec2d p;
        cam->spaceToPlane(box3d->corners.col(i),p);//计算3D box投影到图像平面
        //cam0->ProjectPoint(box3d->corners.col(i),p);
        box3d->corners_2d.col(i) = p;
    }

    Vec2d corner2d_min_pt = box3d->corners_2d.rowwise().minCoeff();
    Vec2d corner2d_max_pt = box3d->corners_2d.rowwise().maxCoeff();
    box3d->box2d.min_pt.x = (float) corner2d_min_pt.x();
    box3d->box2d.min_pt.y = (float) corner2d_min_pt.y();
    box3d->box2d.max_pt.x = (float) corner2d_max_pt.x();
    box3d->box2d.max_pt.y = (float) corner2d_max_pt.y();
    box3d->box2d.center_pt = (box3d->box2d.min_pt + box3d->box2d.max_pt) / 2;

    box3d->center_pt = (box3d->corners.col(0) + box3d->corners.col(6)) / 2;//计算包围框中心坐标

    return box3d;
}




Box3D::Ptr Box3D::Box3dFromKittiTracking(vector<string> &tokens,camodocal::CameraPtr &cam)
{
    /**
     * gt标签的内容
       1    frame        Frame within the sequence where the object appearers
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
     */
    int frame = std::stoi(tokens[0]);
    int track_id = std::stoi(tokens[1]);
    string class_name = tokens[2];
    int class_id = kitti::GetKittiLabelIndex(class_name);
    double score=1.;

    Box3D::Ptr box3d = std::make_shared<Box3D>(class_id,class_name,score);
    box3d->id = track_id;

    double alpha=std::stod(tokens[5]);

    box3d->box2d.min_pt.x =  std::stof(tokens[6]);
    box3d->box2d.min_pt.y =  std::stof(tokens[7]);
    box3d->box2d.max_pt.x =  std::stof(tokens[8]);
    box3d->box2d.max_pt.y =  std::stof(tokens[9]);

    ///注意,这里的维度的与Box3dFromFCOS3D的不同
    box3d->dims<<std::stod(tokens[12]),std::stod(tokens[11]),std::stod(tokens[10]);

    box3d->bottom_center<<std::stod(tokens[13]),std::stod(tokens[14]),std::stod(tokens[15]);


    /**
                z front (yaw=-0.5*pi)
               /
              /
             0 ------> x right (yaw=0)
             |
             |
             v
        down y

    The relative coordinate of bottom center in a CAM box is (0.5, 1.0, 0.5),
    and the yaw is around the y axis, thus the rotation axis=1.
    The yaw is 0 at the positive direction of x axis, and decreases from
    the positive direction of x to the positive direction of z.
     */
    ///9个yaw角(绕着y轴,因为y轴是垂直向下的)
    double yaw = std::stod(tokens[16]);
    //将yaw角限制到[-2pi,0]范围
    while(yaw>0) yaw -= (2*M_PI);
    while(yaw< (-2*M_PI)) yaw += (2*M_PI);
    //将yaw角限定在[-pi,0]上
    if(yaw < (-M_PI)){
        yaw += M_PI;
    }
    box3d->yaw = yaw;

    ///根据yaw角构造旋转矩阵
    Mat3d R_co = box3d->R_cioi();
    box3d->corners = Box3D::GetCorners(box3d->dims,R_co,box3d->bottom_center);

    for(int i=0;i<8;++i){
        Vec2d p;
        cam->spaceToPlane(box3d->corners.col(i),p);//计算3D box投影到图像平面
        //cam0->ProjectPoint(box3d->corners.col(i),p);//计算3D box投影到图像平面
        box3d->corners_2d.col(i) = p;
    }

    Vec2d corner2d_min_pt = box3d->corners_2d.rowwise().minCoeff();
    Vec2d corner2d_max_pt = box3d->corners_2d.rowwise().maxCoeff();
    box3d->box2d.min_pt.x = (float) corner2d_min_pt.x();
    box3d->box2d.min_pt.y = (float) corner2d_min_pt.y();
    box3d->box2d.max_pt.x = (float) corner2d_max_pt.x();
    box3d->box2d.max_pt.y = (float) corner2d_max_pt.y();
    box3d->box2d.center_pt = (box3d->box2d.min_pt + box3d->box2d.max_pt) / 2;

    box3d->center_pt = (box3d->corners.col(0) + box3d->corners.col(6)) / 2;//计算包围框中心坐标

    return box3d;
}


/**
 * 生成包围框的顶点
 * @param dims
 * @param R_xo
 * @param P_xo
 * @return
 */
Mat38d Box3D::GetCorners(Vec3d &dims,Mat3d &R_xo,Vec3d &P_xo){
    ///构造物体坐标系下的角点
    /**
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
    Eigen::Matrix<double,8,3> corners_norm;
    corners_norm << 0,0,0,  0,0,1,  0,1,1,  0,1,0,  1,0,0,  1,0,1,  1,1,1,  1,1,0;
    Eigen::Vector3d offset(0.5,1,0.5);//预测结果所在的坐标系与相机坐标系之间的偏移
    corners_norm = corners_norm.array().rowwise() - offset.transpose().array();//将每个坐标减去偏移量,将坐标系原点设置为包围框底部的中心,
    Mat38d corners = corners_norm.transpose(); //得到矩阵 3x8

    corners = corners.array().colwise() * dims.array();//广播逐点乘法,乘以长度

    for(int i=0;i<8;++i){
        Vec3d v=R_xo * corners.col(i) + P_xo;
        corners.col(i)=v;
    }

    return corners;
}


/**
 * 根据yaw角构造旋转矩阵,并获得物体包围框的坐标系在相机坐标系下的四个点,
 * @return
 */
Mat34d Box3D::GetCoordinateVectorInCamera(double axis_len) const{
    /**
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

    Mat3d R = R_cioi();

    Vec3d x_unit(axis_len,0,0);
    Vec3d y_unit(0,axis_len,0);
    Vec3d z_unit(0,0,axis_len);

    Mat34d matrix;
    matrix.col(0) = bottom_center;
    matrix.col(1) = R * x_unit + bottom_center;
    matrix.col(2) = R * y_unit + bottom_center;
    matrix.col(3) = R * z_unit + bottom_center;


    /*Mat34d matrix;
    matrix.col(0) = (corners.col(1)+corners.col(7))/2.;
    matrix.col(1) = (corners.col(4)+corners.col(5)+corners.col(6)+corners.col(7))/4.;
    matrix.col(2) = (corners.col(1)+corners.col(2)+corners.col(5)+corners.col(6))/4.;
    matrix.col(3) = (corners.col(2)+corners.col(3)+corners.col(6)+corners.col(7))/4.;*/

    return matrix;
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
Mat28d Box3D::CornersProjectTo2D(camodocal::CameraPtr &cam)
{
    Mat28d corners_2d_tmp;
    for(int i=0;i<8;++i){
        Vec2d p;
        cam->spaceToPlane(corners.col(i),p);
        corners_2d_tmp.col(i) = p;
    }
    return corners_2d_tmp;
}



/**
 * 在图像上绘制box
 * @param img
 * @param color
 * @param cam
 */
void Box3D::VisCorners2d(cv::Mat &img,const cv::Scalar& color,camodocal::CameraPtr &cam){
    Mat28d corners2d = CornersProjectTo2D(cam);
    vector<std::pair<int,int>> lines = GetLineVetexPair();

    for(auto line : lines){
        cv::line(img,cv::Point2f(corners2d(0,line.first),corners2d(1,line.first)),
                 cv::Point2f(corners2d(0,line.second),corners2d(1,line.second)),color,2);
    }
}




}
