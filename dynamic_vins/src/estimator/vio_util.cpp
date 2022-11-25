/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "vio_util.h"

#include "utils/log_utils.h"
#include "body.h"
#include "line_detector/line_geometry.h"


namespace dynamic_vins{\



/**
 * 三角化某个点
 * @param Pose0
 * @param Pose1
 * @param point0
 * @param point1
 * @return
 */
Vec3d TriangulatePoint(const Mat34d &Pose0, const Mat34d &Pose1, const Vec2d &point0, const Vec2d &point1){
    Mat4d design_matrix = Mat4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Vec4d triangulated_point;
    triangulated_point =
            design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    return {
        triangulated_point(0) / triangulated_point(3),
        triangulated_point(1) / triangulated_point(3),
        triangulated_point(2) / triangulated_point(3)
    };
}


/**
 * 特征点的三角化
 * @param Pose0
 * @param Pose1
 * @param point0
 * @param point1
 * @param point_3d
 */
void TriangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                      Eigen::Vector2d &point0, Eigen::Vector2d &point1, Eigen::Vector3d &point_3d)
{
    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);//
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);

    Eigen::Vector4d triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

/**
 * 动态特征点的三角化
 * @param Pose0
 * @param Pose1
 * @param point0
 * @param point1
 * @param v
 * @param a
 * @param delta_t
 * @param point_3d
 */
void TriangulateDynamicPoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
                             Eigen::Vector2d &point0, Eigen::Vector2d &point1,
                             Eigen::Vector3d &v, Eigen::Vector3d &a, double delta_t,
                             Eigen::Vector3d &point_3d){
    //构造T_delta
    Eigen::Matrix4d Ma;
    Ma.block<3,3>(0,0) =Sophus::SO3d::exp(a*delta_t).matrix();
    Ma.block<3,1>(0,3)=v*delta_t;
    Ma(3,3)=1;

    Eigen::Matrix<double, 2, 4> Mb;
    Mb.row(0)=point1[0] * Pose1.row(2) - Pose1.row(0);
    Mb.row(1)=point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Matrix<double,2,4> Mc = Mb * Ma;

    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);//
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = Mc.row(0);
    design_matrix.row(3) = Mc.row(1);

    Eigen::Vector4d triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


void TriangulateDynamicPoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1,
                             const Eigen::Vector2d &point0, const  Eigen::Vector2d &point1,
                             const Eigen::Matrix3d &R_woj, const Eigen::Vector3d &P_woj,
                             const Eigen::Matrix3d &R_woi, const Eigen::Vector3d &P_woi,
                             Eigen::Vector3d &point_3d){
    //构造T_delta
    Eigen::Matrix4d Ma;
    Ma.block<3,3>(0,0) =R_woi*R_woj.transpose();
    Ma.block<3,1>(0,3)=R_woi*(-R_woj.transpose() * P_woj) + P_woi;
    Ma(3,3)=1;

    Eigen::Matrix<double, 2, 4> Mb;
    Mb.row(0)=point1[0] * Pose1.row(2) - Pose1.row(0);
    Mb.row(1)=point1[1] * Pose1.row(2) - Pose1.row(1);
    Eigen::Matrix<double,2,4> Mc = Mb * Ma;

    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);//
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = Mc.row(0);
    design_matrix.row(3) = Mc.row(1);

    Eigen::Vector4d triangulated_point;
    triangulated_point = design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


/**
 * 使用RANSAC拟合包围框,当50%的点位于包围框内时,则满足要求
 * @param points 相机坐标系下的3D点
 * @param dims 包围框的长度
 * @return
 */
std::optional<Vec3d> FitBox3DFromObjectFrame(vector<Vec3d> &points,const Vec3d& dims)
{
    Vec3d center_pt = Vec3d::Zero();
    if(points.empty()){
        return {};
    }
    ///计算初始值
    for(auto &p:points){
        center_pt+=p;
    }
    center_pt /= (double)points.size();
    bool is_find=false;

    ///最多迭代10次
    for(int iter=0;iter<10;++iter){
        //计算每个点到中心的距离
        vector<tuple<double,Vec3d>> points_with_dist;
        for(auto &p : points){
            points_with_dist.emplace_back((p-center_pt).norm(),p);
        }
        //根据距离排序,升序
        std::sort(points_with_dist.begin(),points_with_dist.end(),[](tuple<double,Vec3d> &a,tuple<double,Vec3d> &b){
            return std::get<0>(a) < std::get<0>(b);
        });
        //选择前50%的点重新计算中心
        center_pt.setZero();
        double len = points_with_dist.size();
        for(int i=0;i<len/2;++i){
            center_pt += std::get<1>(points_with_dist[i]);
        }
        center_pt /= (len/2);
        //如前50的点位于包围框内,则退出
        if(std::get<0>(points_with_dist[int(len/2)]) <= dims.norm()){
            is_find=true;
            break;
        }
    }

    if(is_find){
        return center_pt;
    }
    else{
        return {};
    }
}





/**
 * 使用RANSAC拟合包围框,根据距离的远近删除点
 * @param points 相机坐标系下的3D点
 * @param dims 包围框的长度
 * @return
 */
std::optional<Vec3d> FitBox3DFromCameraFrame(vector<Vec3d> &points,const Vec3d& dims)
{
    Vec3d center_pt = Vec3d::Zero();
    if(points.empty()){
        return {};
    }
    std::list<Vec3d> points_rest;
    ///计算初始值
    for(auto &p:points){
        center_pt+=p;
        points_rest.push_back(p);
    }
    center_pt /= (double)points.size();
    bool is_find=false;

    double dims_norm = (dims/2.).norm();

    string log_text = "FitBox3DFromCameraFrame: \n";

    ///最多迭代10次
    for(int iter=0;iter<10;++iter){
        //计算每个点到中心的距离
        vector<tuple<double,Vec3d>> points_with_dist;
        for(auto &p : points_rest){
            points_with_dist.emplace_back((p-center_pt).norm(),p);
        }
        //根据距离排序,升序
        std::sort(points_with_dist.begin(),points_with_dist.end(),[](tuple<double,Vec3d> &a,tuple<double,Vec3d> &b){
            return std::get<0>(a) < std::get<0>(b);
        });

        //选择前80%的点重新计算中心
        center_pt.setZero();
        double len = points_with_dist.size();
        int len_used=len*0.8;
        for(int i=0;i<len_used;++i){
            center_pt += std::get<1>(points_with_dist[i]);
        }
        if(len_used<2){
            break;
        }
        center_pt /= len_used;

        log_text += fmt::format("iter:{} center_pt:{}\n",iter, VecToStr(center_pt));

        //如前80%的点位于包围框内,则退出
        if(std::get<0>(points_with_dist[len_used]) <= dims_norm){
            is_find=true;
            break;
        }

        ///只将距离相机中心最近的前50%点用于下一轮的计算
        vector<tuple<double,Vec3d>> points_cam_dist;
        for(auto &p : points_rest){
            points_cam_dist.emplace_back(p.norm(),p);
        }
        //根据距离排序,升序
        std::sort(points_cam_dist.begin(),points_cam_dist.end(),[](tuple<double,Vec3d> &a,tuple<double,Vec3d> &b){
            return std::get<0>(a) < std::get<0>(b);
        });
        points_rest.clear();
        for(int i=0;i<len*0.5;++i){
            points_rest.push_back(std::get<1>(points_cam_dist[i]));
        }

    }

    Debugv(log_text);

    if(is_find){
        return center_pt;
    }
    else{
        return {};
    }
}


/**
 * 选择距离相机最近的前30%的3D点用于计算物体中心
 * @param points
 * @param dims
 * @return
 */
std::optional<Vec3d> FitBox3DSimple(vector<Vec3d> &points,const Vec3d& dims){
    if(points.size() < 4){
        return {};
    }

    vector<tuple<double,Vec3d>> points_cam_dist;
    for(auto &p : points){
        points_cam_dist.emplace_back(p.norm(),p);
    }
    //根据距离排序,升序
    std::sort(points_cam_dist.begin(),points_cam_dist.end(),[](tuple<double,Vec3d> &a,tuple<double,Vec3d> &b){
        return std::get<0>(a) < std::get<0>(b);
    });

    double len = points_cam_dist.size();
    Vec3d center_pt = Vec3d::Zero();
    int size = len*0.3;
    for(int i=0;i<size;++i){
        center_pt += std::get<1>(points_cam_dist[i]);
    }
    center_pt /= double(size);
    return center_pt;
}



/**
 * 根据重投影误差判断哪些点需要被剔除
 * @param removeIndex
 */
void OutliersRejection(std::set<int> &removeIndex,std::list<StaticLandmark>& point_landmarks)
{
    //return;
    int feature_index = -1;
    for (auto &lm : point_landmarks){
        double err = 0;
        int errCnt = 0;
        lm.used_num = lm.feats.size();
        if (lm.used_num < 4)
            continue;
        feature_index ++;
        int imu_i = lm.start_frame, imu_j = imu_i - 1;
        Vec3d pts_i = lm.feats[0].point;
        double depth = lm.depth;

        for (auto &feat : lm.feats){
            imu_j++;
            if (imu_i != imu_j){
                Vec3d pts_j = feat.point;
                double tmp_error = ReprojectionError(body.Rs[imu_i], body.Ps[imu_i], body.ric[0], body.tic[0],
                                                     body.Rs[imu_j], body.Ps[imu_j], body.ric[0], body.tic[0],
                                                     depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
            }
            // need to rewrite projecton factor.........
            if(cfg::is_stereo && feat.is_stereo){
                Vec3d pts_j_right = feat.point_right;
                if(imu_i != imu_j){
                    double tmp_error = ReprojectionError(body.Rs[imu_i], body.Ps[imu_i], body.ric[0], body.tic[0],
                                                         body.Rs[imu_j], body.Ps[imu_j], body.ric[1], body.tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                }
                else{
                    double tmp_error = ReprojectionError(body.Rs[imu_i], body.Ps[imu_i], body.ric[0], body.tic[0],
                                                         body.Rs[imu_j], body.Ps[imu_j], body.ric[1], body.tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                }
            }
        }
        double ave_err = err / errCnt;
        if(ave_err * kFocalLength > 3)
            removeIndex.insert(lm.feature_id);

    }
}



double ReprojectionError(Mat3d &Ri, Vec3d &Pi, Mat3d &rici, Vec3d &tici,
                                    Mat3d &Rj, Vec3d &Pj, Mat3d &ricj, Vec3d &ticj,
                                    double depth, Vec3d &uvi, Vec3d &uvj){
    Vec3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vec3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vec2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}



void TriangulateOneLine(LineLandmark &line){

    int imu_i = line.start_frame, imu_j = imu_i - 1;

    Vec3d t0 = body.Ps[imu_i] + body.Rs[imu_i] * body.tic[0];   // twc = Rwi * tic + twi
    Mat3d R0 = body.Rs[imu_i] * body.ric[0];               // Rwc = Rwi * Ric

    double d = 0, min_cos_theta = 1.0;
    Vec3d tij;
    Mat3d Rij;
    Vec4d obsi;//i时刻的观测
    Vec4d obsj;

    Vec4d pii;//i时刻的观测构建的直线
    Vec3d ni;//pii的法向量

    ///寻找观测平面夹角最小的两帧

    for (auto &feat : line.feats){   // 遍历所有的观测， 注意 start_frame 也会被遍历
        imu_j++;

        if(imu_j == imu_i){   // 第一个观测是start frame 上
            obsi = feat.line_obs;
            Vec3d p1( obsi(0), obsi(1), 1 );
            Vec3d p2( obsi(2), obsi(3), 1 );
            pii = pi_from_ppp(p1, p2,Vec3d( 0, 0, 0 ));//3点构建一个平面
            ni = pii.head(3);
            ni.normalize();
            continue;
        }

        // 非start frame(其他帧)上的观测
        Vec3d t1 = body.Ps[imu_j] + body.Rs[imu_j] * body.tic[0];
        Mat3d R1 = body.Rs[imu_j] * body.ric[0];

        Vec3d t = R0.transpose() * (t1 - t0);   // tij
        Mat3d R = R0.transpose() * R1;          // Rij

        Vec4d obsj_tmp = feat.line_obs;
        ///将直线从j时刻变换到i时刻
        Vec3d p3( obsj_tmp(0), obsj_tmp(1), 1 );
        Vec3d p4( obsj_tmp(2), obsj_tmp(3), 1 );
        p3 = R * p3 + t;
        p4 = R * p4 + t;
        //构建观测j在i时刻的平面
        Vec4d pij = pi_from_ppp(p3, p4,t);
        Vec3d nj = pij.head(3);
        nj.normalize();

        double cos_theta = ni.dot(nj);
        if(cos_theta < min_cos_theta){
            min_cos_theta = cos_theta;
            tij = t;
            Rij = R;
            obsj = obsj_tmp;
            d = t.norm();
        }
        /*             if( d < t.norm() )  // 选择最远的那俩帧进行三角化
                     {
                         d = t.norm();
                         tij = t;
                         Rij = R;
                         obsj = it_per_frame.lineobs;      // 特征的图像坐标
                     }*/

    }

    // if the distance between two frame is lower than 0.1m or the parallax angle is lower than 15deg , do not triangulate.
    // if(d < 0.1 || min_cos_theta > 0.998)
    if(min_cos_theta > 0.998)
        // if( d < 0.2 )
        return;

    // plane pi from jth obs in ith camera frame
    Vec3d p3( obsj(0), obsj(1), 1 );
    Vec3d p4( obsj(2), obsj(3), 1 );
    p3 = Rij * p3 + tij;
    p4 = Rij * p4 + tij;
    Vec4d pij = pi_from_ppp(p3, p4,tij);

    ///根据两个平面，构建PLK坐标
    Vec6d plk = pipi_plk( pii, pij );

    //Vec3d cp = plucker_origin( n, v );
    //if ( cp(2) < 0 )
    {
        //  cp = - cp;
        //  continue;
    }

    //Vector6d line;
    //line.head(3) = cp;
    //line.tail(3) = v;
    //it_per_id.line_plucker = line;

    // plk.normalize();

    ///获得两个3D端点
    auto [valid,pts_1,pts_2] = LineTrimming(plk,line.feats[0].line_obs);

    ///限制直线的长度
    if(!valid || (pts_1-pts_2).norm()>10.){
        return;
    }

    Vec3d w_pts_1 =  body.Rs[imu_i] * (body.ric[0] * pts_1 + body.tic[0]) + body.Ps[imu_i];
    Vec3d w_pts_2 =  body.Rs[imu_i] * (body.ric[0] * pts_2 + body.tic[0]) + body.Ps[imu_i];

    line.ptw1 = w_pts_1;
    line.ptw2 = w_pts_2;

    line.line_plucker = plk;  // plk in camera frame
    line.is_triangulation = true;

}



void TriangulateOneLineStereo(LineLandmark &line){

    int imu_i = line.start_frame;

    Vec3d t0 = body.Ps[imu_i] + body.Rs[imu_i] * body.tic[0];   // twc = Rwi * tic + twi
    Mat3d R0 = body.Rs[imu_i] * body.ric[0];               // Rwc = Rwi * Ric

    // 非start frame(其他帧)上的观测
    Vec3d t1 = body.Ps[imu_i] + body.Rs[imu_i] * body.tic[1];
    Mat3d R1 = body.Rs[imu_i] * body.ric[1];

    Vec3d t = R0.transpose() * (t1 - t0);   // tij
    Mat3d R = R0.transpose() * R1;          // Rij

    const Vec3d& tij = t;
    const Mat3d& Rij = R;

    double d = 0, min_cos_theta = 1.0;

    Vec4d obsi = line.feats[0].line_obs;//i时刻的观测

    Vec4d pii;//i时刻的观测构建的直线
    Vec3d ni;//pii的法向量

    Vec3d p1( obsi(0), obsi(1), 1 );
    Vec3d p2( obsi(2), obsi(3), 1 );
    pii = pi_from_ppp(p1, p2,Vec3d( 0, 0, 0 ));//3点构建一个平面
    ni = pii.head(3);
    ni.normalize();

    Vec4d obsj = line.feats[0].line_obs_right;
    d = t.norm();

    // plane pi from jth obs in ith camera frame
    Vec3d p3( obsj(0), obsj(1), 1 );
    Vec3d p4( obsj(2), obsj(3), 1 );
    p3 = Rij * p3 + tij;
    p4 = Rij * p4 + tij;
    Vec4d pij = pi_from_ppp(p3, p4,tij);

    ///根据两个平面，构建PLK坐标
    Vec6d plk = pipi_plk( pii, pij );

    auto [valid,pts_1,pts_2] = LineTrimming(plk,line.feats[0].line_obs);

    ///限制直线的长度
    if(!valid || (pts_1-pts_2).norm()>50){
        return;
    }

    Vec3d w_pts_1 =  body.Rs[imu_i] * (body.ric[0] * pts_1 + body.tic[0]) + body.Ps[imu_i];
    Vec3d w_pts_2 =  body.Rs[imu_i] * (body.ric[0] * pts_2 + body.tic[0]) + body.Ps[imu_i];

    line.ptw1 = w_pts_1;
    line.ptw2 = w_pts_2;

    // plk.normalize();
    line.line_plucker = plk;  // plk in camera frame
    line.is_triangulation = true;

}






/**
 * PnP求解
 * @param R
 * @param P
 * @param pts2D
 * @param pts3D
 * @return
 */
bool SolvePoseByPnP(Mat3d &R, Vec3d &P,
                    vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D){
    Mat3d R_initial;
    Vec3d P_initial;

    // w_T_cam ---> cam_T_w
    R_initial = R.inverse();
    P_initial = -(R_initial * P);

    if (int(pts2D.size()) < 4){
        printf("feature tracking not enough, please slowly move you device! \n");
        return false;
    }
    //计算初值
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    //调用OpenCV函数求解
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts3D, pts2D, K, D, rvec, t, true);
    //pnp_succ = solvePnPRansac(pts3D, pts2D, K, D, rvec, t, true, 100, 8.0 / focalLength, 0.99, inliers);

    if(!pnp_succ){
        printf("pnp failed ! \n");
        return false;
    }

    cv::Rodrigues(rvec, r);
    Eigen::MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);

    // cam_T_w ---> w_T_cam
    R = R_pnp.transpose();
    P = R * (-T_pnp);
    return true;
}





double CompensatedParallax2(const StaticLandmark &landmark, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const StaticFeature &frame_i = landmark.feats[frame_count - 2 - landmark.start_frame];
    const StaticFeature &frame_j = landmark.feats[frame_count - 1 - landmark.start_frame];

    double ans = 0;
    Vec3d p_j = frame_j.point;

    double u_j = p_j(0);
    double v_j = p_j(1);

    Vec3d p_i = frame_i.point;
    Vec3d p_i_comp;

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i;
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j;

    ans = std::max(ans, sqrt(std::min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}




}