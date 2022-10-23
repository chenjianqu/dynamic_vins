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



#include "feature_manager.h"
#include "vio_util.h"
#include "vio_parameters.h"
#include "body.h"
#include "line_detector/line_geometry.h"


namespace dynamic_vins{\



int FeaturePerId::endFrame()
{
    return start_frame + feats.size() - 1;
}

FeatureManager::FeatureManager(Mat3d _Rs[])
: Rs(_Rs)
{
    for (int i = 0; i < Config::kCamNum; i++)
        ric[i].setIdentity();
}

void FeatureManager::SetRic(Mat3d _ric[])
{
    for (int i = 0; i < Config::kCamNum; i++){
        ric[i] = _ric[i];
    }
}

void FeatureManager::ClearState()
{
    point_landmarks.clear();
}

/**
 * 统计当前地图中被观测4次以上的地点的数量
 * @return
 */
int FeatureManager::GetFeatureCount()
{
    int cnt = 0;
    for (auto &it : point_landmarks){
        it.used_num = it.feats.size();
        if (it.used_num >= 4){
            cnt++;
        }
    }
    return cnt;
}

/**
 *
 * @param frame_count
 * @param image
 * @param td
 * @return
 */
bool FeatureManager::AddFeatureCheckParallax(int frame_count, const std::map<unsigned int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    last_average_parallax = 0;
    new_feature_num = 0;
    long_track_num = 0;

    for (auto &[feature_id,feat_vec] : image){ ///遍历每个观测
        FeaturePerFrame feat(feat_vec[0].second, td);
        assert(feat_vec[0].first == 0);
        if(feat_vec.size() == 2){
            feat.rightObservation(feat_vec[1].second);
            assert(feat_vec[1].first == 1);
        }
        ///判断当前观测是否存在对应的路标
        if (auto it = find_if(point_landmarks.begin(), point_landmarks.end(),
                              [feature_id=feature_id](const FeaturePerId &it){
            return it.feature_id == feature_id;
        });
        ///未存在路标，创建路标
        it == point_landmarks.end()){
            point_landmarks.emplace_back(feature_id, frame_count);
            point_landmarks.back().feats.push_back(feat);
            new_feature_num++;
        }
        ///存在路标，添加观测
        else if (it->feature_id == feature_id){
            it->feats.push_back(feat);
            last_track_num++;
            if(it-> feats.size() >= 4)
                long_track_num++;
        }
    }

    //if (frame_count < 2 || last_track_num < 20)
    //if (frame_count < 2 || last_track_num < 20 || new_feature_num > 0.5 * last_track_num)
    if (frame_count < 2 || last_track_num < 20 || long_track_num < 40 || new_feature_num > 0.5 * last_track_num)
        return true;

    ///计算视差
    for (auto &lm : point_landmarks){
        if (lm.start_frame <= frame_count - 2 && lm.start_frame + int(lm.feats.size()) - 1 >= frame_count - 1){
            parallax_sum += CompensatedParallax2(lm, frame_count);
            parallax_num++;
        }
    }

    if (parallax_num == 0){
        return true;
    }
    else{
        //Debugv("addFeatureCheckParallax parallax_sum: {}, parallax_num: {}", parallax_sum, parallax_num);
        Debugv("addFeatureCheckParallax current parallax: {}", parallax_sum / parallax_num * kFocalLength);
        last_average_parallax = parallax_sum / parallax_num * kFocalLength;
        return parallax_sum / parallax_num >= para::kMinParallax;
    }
}


bool FeatureManager::AddFeatureCheckParallax(int frame_count, FeatureBackground &image, double td){
    ///添加线特征
    for(auto &[line_id,vec_feat]:image.lines){
        if(vec_feat.size()>1){
            LineFeature feat(vec_feat[0].second,vec_feat[1].second);
            auto it = find_if(line_landmarks.begin(), line_landmarks.end(), [id=line_id](const LineLandmark &it){
                return it.feature_id == id;    // 在feature里找id号为feature_id的特征
            });

            if (it == line_landmarks.end()){  // 如果之前没存这个特征，说明是新的
                line_landmarks.emplace_back(line_id, frame_count);//创建线路标
                line_landmarks.back().feats.push_back(feat);
            }
            else if (it->feature_id == line_id){
                it->feats.push_back(feat);
                it->all_obs_cnt++;
            }
        }
        else{
            LineFeature feat(vec_feat[0].second);
            auto it = find_if(line_landmarks.begin(), line_landmarks.end(), [id=line_id](const LineLandmark &it){
                return it.feature_id == id;    // 在feature里找id号为feature_id的特征
            });

            if (it == line_landmarks.end()){  // 如果之前没存这个特征，说明是新的
                line_landmarks.emplace_back(line_id, frame_count);
                line_landmarks.back().feats.push_back(feat);
            }
            else if (it->feature_id == line_id){
                it->feats.push_back(feat);
                it->all_obs_cnt++;
            }
        }
    }

    ///添加点特征
    return AddFeatureCheckParallax(frame_count,image.points,td);
}



vector<pair<Vec3d, Vec3d>> FeatureManager::GetCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vec3d, Vec3d>> corres;
    for (auto &it : point_landmarks)
    {
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vec3d a = Vec3d::Zero(), b = Vec3d::Zero();
            int idx_l = frame_count_l - it.start_frame;
            int idx_r = frame_count_r - it.start_frame;

            a = it.feats[idx_l].point;
            b = it.feats[idx_r].point;
            corres.emplace_back(a, b);
        }
    }
    return corres;
}

void FeatureManager::SetDepth(const Eigen::VectorXd &x)
{
    int feature_index = -1;
    for (auto &lm : point_landmarks){
        lm.used_num = lm.feats.size();
        if (lm.used_num < 4)
            continue;
        lm.depth = 1.0 / x(++feature_index);
        if (lm.depth < 0){
            lm.solve_flag = 2;
        }
        else{
            lm.solve_flag = 1;
        }
    }
}

void FeatureManager::RemoveFailures()
{
    for (auto it = point_landmarks.begin(), it_next = point_landmarks.begin(); it != point_landmarks.end(); it = it_next){
        it_next++;
        if (it->solve_flag == 2)
            point_landmarks.erase(it);
    }
}

void FeatureManager::ClearDepth()
{
    for (auto &lm : point_landmarks)
        lm.depth = -1;
}

Eigen::VectorXd FeatureManager::GetDepthVector()
{
    Eigen::VectorXd dep_vec(GetFeatureCount());
    int feature_index = -1;
    for (auto &lm : point_landmarks){
        lm.used_num = lm.feats.size();
        if (lm.used_num < 4)
            continue;
#if 1
        dep_vec(++feature_index) = 1. / lm.depth;
#else
        dep_vec(++feature_index) = lm->depth;
#endif
    }
    return dep_vec;
}


void FeatureManager::TriangulatePoint(Mat34d &Pose0, Mat34d &Pose1,
                                      Vec2d &point0, Vec2d &point1, Vec3d &point_3d){
    Mat4d design_matrix = Mat4d::Zero();
    design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
    Vec4d triangulated_point;
    triangulated_point =
            design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    point_3d(0) = triangulated_point(0) / triangulated_point(3);
    point_3d(1) = triangulated_point(1) / triangulated_point(3);
    point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

/**
 * PnP求解
 * @param R
 * @param P
 * @param pts2D
 * @param pts3D
 * @return
 */
bool FeatureManager::SolvePoseByPnP(Mat3d &R, Vec3d &P,
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


/**
 * 使用PnP求解得到当前帧的位姿
 * @param frameCnt
 * @param Ps
 * @param Rs
 * @param tic
 * @param ric
 */
void FeatureManager::InitFramePoseByPnP(int frameCnt, Vec3d Ps[], Mat3d Rs[], Vec3d tic[], Mat3d ric[])
{
    if(frameCnt > 0){
        ///构建3D-2D匹配对
        vector<cv::Point2f> pts2D;
        vector<cv::Point3f> pts3D;
        for (auto &lm : point_landmarks){
            if (lm.depth > 0){
                int index = frameCnt - lm.start_frame;
                if((int)lm.feats.size() >= index + 1){
                    Vec3d ptsInCam = ric[0] * (lm.feats[0].point * lm.depth) + tic[0];
                    Vec3d ptsInWorld = Rs[lm.start_frame] * ptsInCam + Ps[lm.start_frame];

                    cv::Point3f point3d(ptsInWorld.x(), ptsInWorld.y(), ptsInWorld.z());
                    cv::Point2f point2d(lm.feats[index].point.x(), lm.feats[index].point.y());
                    pts3D.push_back(point3d);
                    pts2D.push_back(point2d);
                }
            }
        }
        Debugv("InitFramePoseByPnP pts2D.size:{}", pts2D.size());
        ///使用上一帧的位姿作为初值
        Mat3d RCam;
        Vec3d PCam;
        // trans to w_T_cam
        RCam = Rs[frameCnt - 1] * ric[0];
        PCam = Rs[frameCnt - 1] * tic[0] + Ps[frameCnt - 1];
        ///求解
        if(SolvePoseByPnP(RCam, PCam, pts2D, pts3D)){
            // trans to w_T_imu
            Rs[frameCnt] = RCam * ric[0].transpose();
            Ps[frameCnt] = -RCam * ric[0].transpose() * tic[0] + PCam;
        }
    }
}

/**
 * 三角化背景特征点
 * @param frameCnt
 * @param Ps
 * @param Rs
 * @param tic
 * @param ric
 */
void FeatureManager::triangulate(int frameCnt, Vec3d Ps[], Mat3d Rs[], Vec3d tic[], Mat3d ric[])
{
    for (auto &lm : point_landmarks){
        if (lm.depth > 0)
            continue;

        if(Config::is_stereo && lm.feats[0].is_stereo){
            int imu_i = lm.start_frame;

            Mat34d leftPose = body.GetCamPose34d(imu_i,0);
            Mat34d rightPose = body.GetCamPose34d(imu_i,1);

            Vec2d point0, point1;
            Vec3d point3d;
            point0 = lm.feats[0].point.head(2);
            point1 = lm.feats[0].pointRight.head(2);

            TriangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Vec3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
            double depth = localPoint.z();
            if (depth > 0)
                lm.depth = depth;
            else
                lm.depth = para::kInitDepth;
            continue;
        }
        else if(lm.feats.size() > 1)
        {
            int imu_i = lm.start_frame;
            Mat34d leftPose = body.GetCamPose34d(imu_i,0);
            imu_i++;
            Mat34d rightPose = body.GetCamPose34d(imu_i,0);

            Vec2d point0, point1;
            Vec3d point3d;
            point0 = lm.feats[0].point.head(2);
            point1 = lm.feats[1].point.head(2);
            TriangulatePoint(leftPose, rightPose, point0, point1, point3d);
            Vec3d localPoint;
            localPoint = leftPose.leftCols<3>() * point3d + leftPose.rightCols<1>();
            double depth = localPoint.z();
            if (depth > 0)
                lm.depth = depth;
            else
                lm.depth = para::kInitDepth;
            continue;
        }

        lm.used_num = lm.feats.size();
        if (lm.used_num < 4)
            continue;

        //以下代码好像执行不到

        int imu_i = lm.start_frame, imu_j = imu_i - 1;

        Eigen::MatrixXd svd_A(2 * lm.feats.size(), 4);
        int svd_idx = 0;

        Mat34d P0;
        Vec3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Mat3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Mat3d::Identity();
        P0.rightCols<1>() = Vec3d::Zero();

        for (auto &feat : lm.feats){
            imu_j++;

            Vec3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Mat3d R1 = Rs[imu_j] * ric[0];
            Vec3d t = R0.transpose() * (t1 - t0);
            Mat3d R = R0.transpose() * R1;
            Mat34d P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Vec3d f = feat.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Vec4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->depth = -b / A;
        //it_per_id->depth = svd_V[2] / svd_V[3];

        lm.depth = svd_method;
        //it_per_id->depth = kInitDepth;

        if (lm.depth < 0.1){
            lm.depth = para::kInitDepth;
        }

    }
}



void FeatureManager::TriangulateLine(Vec3d Ps[], Vec3d tic[], Mat3d ric[])
{
    for (auto &landmark : line_landmarks){        // 遍历每个特征，对新特征进行三角化

        landmark.used_num = landmark.feats.size();    // 已经有多少帧看到了这个特征
        if (!(landmark.used_num >= kLineMinObs && landmark.start_frame < kWinSize - 2))   // 看到的帧数少于2， 或者 这个特征最近倒数第二帧才看到， 那都不三角化
            continue;
        if (landmark.is_triangulation)       // 如果已经三角化了
            continue;

        int imu_i = landmark.start_frame, imu_j = imu_i - 1;

        Vec3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];   // twc = Rwi * tic + twi
        Mat3d R0 = Rs[imu_i] * ric[0];               // Rwc = Rwi * Ric

        double d = 0, min_cos_theta = 1.0;
        Vec3d tij;
        Mat3d Rij;
        Vec4d obsi,obsj;  // obs from two frame are used to do triangulation

        // plane pi from ith obs in ith camera frame
        Vec4d pii;
        Vec3d ni;      // normal vector of plane
        for (auto &it_per_frame : landmark.feats){   // 遍历所有的观测， 注意 start_frame 也会被遍历
            imu_j++;

            if(imu_j == imu_i){   // 第一个观测是start frame 上
                obsi = it_per_frame.line_obs;
                Vec3d p1( obsi(0), obsi(1), 1 );
                Vec3d p2( obsi(2), obsi(3), 1 );
                pii = pi_from_ppp(p1, p2,Vec3d( 0, 0, 0 ));
                ni = pii.head(3);
                ni.normalize();
                continue;
            }

            // 非start frame(其他帧)上的观测
            Vec3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Mat3d R1 = Rs[imu_j] * ric[0];

            Vec3d t = R0.transpose() * (t1 - t0);   // tij
            Mat3d R = R0.transpose() * R1;          // Rij

            Vec4d obsj_tmp = it_per_frame.line_obs;

            // plane pi from jth obs in ith camera frame
            Vec3d p3( obsj_tmp(0), obsj_tmp(1), 1 );
            Vec3d p4( obsj_tmp(2), obsj_tmp(3), 1 );
            p3 = R * p3 + t;
            p4 = R * p4 + t;
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
            continue;

        // plane pi from jth obs in ith camera frame
        Vec3d p3( obsj(0), obsj(1), 1 );
        Vec3d p4( obsj(2), obsj(3), 1 );
        p3 = Rij * p3 + tij;
        p4 = Rij * p4 + tij;
        Vec4d pij = pi_from_ppp(p3, p4,tij);

        Vec6d plk = pipi_plk( pii, pij );
        Vec3d n = plk.head(3);
        Vec3d v = plk.tail(3);

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
        landmark.line_plucker = plk;  // plk in camera frame
        landmark.is_triangulation = true;

        //  used to debug
        Vec3d pc, nc, vc;
        nc = landmark.line_plucker.head(3);
        vc = landmark.line_plucker.tail(3);

        ///对偶的plucker矩阵
        Mat4d Lc;
        Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

        Vec4d obs_startframe = landmark.feats[0].line_obs;   // 第一次观测到这帧
        Vec3d p11 = Vec3d(obs_startframe(0), obs_startframe(1), 1.0);
        Vec3d p21 = Vec3d(obs_startframe(2), obs_startframe(3), 1.0);
        Vec2d ln = ( p11.cross(p21) ).head(2);     // 直线的垂直方向
        ln = ln / ln.norm();

        Vec3d p12 = Vec3d(p11(0) + ln(0), p11(1) + ln(1), 1.0);  // 直线垂直方向上移动一个单位
        Vec3d p22 = Vec3d(p21(0) + ln(0), p21(1) + ln(1), 1.0);
        Vec3d cam = Vec3d( 0, 0, 0 );

        Vec4d pi1 = pi_from_ppp(cam, p11, p12);
        Vec4d pi2 = pi_from_ppp(cam, p21, p22);

        Vec4d e1 = Lc * pi1;
        Vec4d e2 = Lc * pi2;
        e1 = e1/e1(3);
        e2 = e2/e2(3);

        Vec3d pts_1(e1(0),e1(1),e1(2));
        Vec3d pts_2(e2(0),e2(1),e2(2));

        Vec3d w_pts_1 =  Rs[imu_i] * (ric[0] * pts_1 + tic[0]) + Ps[imu_i];
        Vec3d w_pts_2 =  Rs[imu_i] * (ric[0] * pts_2 + tic[0]) + Ps[imu_i];
        landmark.ptw1 = w_pts_1;
        landmark.ptw2 = w_pts_2;
    }

    //    removeLineOutlier(Ps,tic,ric);
}




/**
 *  @brief  stereo line triangulate
 */
void FeatureManager::TriangulateLine(double baseline)
{
    for (auto &landmark : line_landmarks) {       // 遍历每个特征，对新特征进行三角化
        landmark.used_num = landmark.feats.size();    // 已经有多少帧看到了这个特征
        if (landmark.is_triangulation)  // 已经三角化了 或者 少于两帧看到 或者 右目没有看到
            continue;
        //查看是否存在双目观测
        /*bool is_stereo=false;
        for(auto &feat:landmark.feats){
            if(feat.is_stereo)
                is_stereo=true;
        }
        if(is_stereo==false)
            continue;*/
        if(!landmark.feats[0].is_stereo){
            continue;
        }

        int imu_i = landmark.start_frame;
        LineFeature feat = landmark.feats.front();

        // plane pi from ith left obs in ith left camera frame
        Vec3d p1( feat.line_obs(0), feat.line_obs(1), 1 );
        Vec3d p2( feat.line_obs(2), feat.line_obs(3), 1 );
        Vec4d pii = pi_from_ppp(p1, p2,Vec3d( 0, 0, 0 ));

        // plane pi from ith right obs in ith left camera frame
        Vec3d p3( feat.line_obs_right(0) + baseline, feat.line_obs_right(1), 1 );
        Vec3d p4( feat.line_obs_right(2) + baseline, feat.line_obs_right(3), 1 );
        Vec4d pij = pi_from_ppp(p3, p4,Vec3d(baseline, 0, 0));

        Vec6d plk = pipi_plk( pii, pij );
        Vec3d n = plk.head(3);
        Vec3d v = plk.tail(3);

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
        landmark.line_plucker = plk;  // plk in camera frame
        landmark.is_triangulation = true;
    }

    RemoveLineOutlier();
}




int FeatureManager::GetLineFeatureCount()
{
    int cnt = 0;
    for (auto &it : line_landmarks){
        it.used_num = it.feats.size();
        if (it.used_num >= kLineMinObs   && it.start_frame < kWinSize - 2 && it.is_triangulation)
            cnt++;
    }
    return cnt;
}


Eigen::MatrixXd FeatureManager::GetLineOrthVectorInCamera()
{
    Eigen::MatrixXd lineorth_vec(GetLineFeatureCount(),4);
    int feature_index = -1;
    for (auto &it_per_id : line_landmarks)
    {
        it_per_id.used_num = it_per_id.feats.size();
        if (!(it_per_id.used_num >= kLineMinObs && it_per_id.start_frame < kWinSize - 2 && it_per_id.is_triangulation))
            continue;

        lineorth_vec.row(++feature_index) = plk_to_orth(it_per_id.line_plucker);

    }
    return lineorth_vec;
}


Eigen::MatrixXd FeatureManager::GetLineOrthVector(Vec3d Ps[], Vec3d tic[], Mat3d ric[])
{
    Eigen::MatrixXd lineorth_vec(GetLineFeatureCount(),4);
    int feature_index = -1;
    for (auto &it_per_id : line_landmarks){
        it_per_id.used_num = it_per_id.feats.size();
        if (!(it_per_id.used_num >= kLineMinObs && it_per_id.start_frame < kWinSize - 2 && it_per_id.is_triangulation))
            continue;

        int imu_i = it_per_id.start_frame;

        Eigen::Vector3d twc = Ps[imu_i] + Rs[imu_i] * tic[0];   // twc = Rwi * tic + twi
        Eigen::Matrix3d Rwc = Rs[imu_i] * ric[0];               // Rwc = Rwi * Ric

        Vec6d line_w = plk_to_pose(it_per_id.line_plucker, Rwc, twc);  // transfrom to world frame
        // line_w.normalize();
        lineorth_vec.row(++feature_index) = plk_to_orth(line_w);
        //lineorth_vec.row(++feature_index) = plk_to_orth(it_per_id.line_plucker);
    }
    return lineorth_vec;
}



void FeatureManager::SetLineOrth(Eigen::MatrixXd x,Vec3d P[], Mat3d R[], Vec3d tic[], Mat3d ric[])
{
    int feature_index = -1;
    for (auto &it_per_id : line_landmarks)
    {
        it_per_id.used_num = it_per_id.feats.size();
        if (!(it_per_id.used_num >= kLineMinObs  && it_per_id.start_frame < kWinSize - 2 && it_per_id.is_triangulation))
            continue;

        Vec4d line_orth_w = x.row(++feature_index);
        Vec6d line_w = orth_to_plk(line_orth_w);

        int imu_i = it_per_id.start_frame;

        Eigen::Vector3d twc = P[imu_i] + R[imu_i] * tic[0];   // twc = Rwi * tic + twi
        Eigen::Matrix3d Rwc = R[imu_i] * ric[0];               // Rwc = Rwi * Ric

        it_per_id.line_plucker = plk_from_pose(line_w, Rwc, twc); // transfrom to camera frame
        //it_per_id.line_plucker = line_w; // transfrom to camera frame

        /*
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
         */
    }
}




void FeatureManager::RemoveLineOutlier()
{
    for (auto landmark = line_landmarks.begin(), it_next = line_landmarks.begin();
         landmark != line_landmarks.end(); landmark = it_next)
    {
        it_next++;
        landmark->used_num = landmark->feats.size();
        // TODO: 右目没看到
        if (landmark->is_triangulation || landmark->used_num < 2)  // 已经三角化了 或者 少于两帧看到 或者 右目没有看到
            continue;

        int imu_i = landmark->start_frame, imu_j = imu_i - 1;

        // 计算初始帧上线段对应的3d端点
        Vec3d pc, nc, vc;
        nc = landmark->line_plucker.head(3);
        vc = landmark->line_plucker.tail(3);

        Mat4d Lc;
        Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

        Vec4d obs_startframe = landmark->feats[0].line_obs;   // 第一次观测到这帧
        Vec3d p11 = Vec3d(obs_startframe(0), obs_startframe(1), 1.0);
        Vec3d p21 = Vec3d(obs_startframe(2), obs_startframe(3), 1.0);
        Vec2d ln = ( p11.cross(p21) ).head(2);     // 直线的垂直方向
        ln = ln / ln.norm();

        Vec3d p12 = Vec3d(p11(0) + ln(0), p11(1) + ln(1), 1.0);  // 直线垂直方向上移动一个单位
        Vec3d p22 = Vec3d(p21(0) + ln(0), p21(1) + ln(1), 1.0);
        Vec3d cam = Vec3d( 0, 0, 0 );

        Vec4d pi1 = pi_from_ppp(cam, p11, p12);
        Vec4d pi2 = pi_from_ppp(cam, p21, p22);

        Vec4d e1 = Lc * pi1;
        Vec4d e2 = Lc * pi2;
        e1 = e1/e1(3);
        e2 = e2/e2(3);

        if(e1(2) < 0 || e2(2) < 0){
            line_landmarks.erase(landmark);
            continue;
        }

        if((e1-e2).norm() > 10){
            line_landmarks.erase(landmark);
            continue;
        }
        /*
                // 点到直线的距离不能太远啊
                Vec3d Q = plucker_origin(nc,vc);
                if(Q.norm() > 5.0)
                {
                    linefeature.erase(it_per_id);
                    continue;
                }
        */

    }
}



void FeatureManager::RemoveLineOutlier(Vec3d Ps[], Vec3d tic[], Mat3d ric[])
{

    for (auto landmark = line_landmarks.begin(), it_next = line_landmarks.begin();
         landmark != line_landmarks.end(); landmark = it_next)
    {
        it_next++;
        landmark->used_num = landmark->feats.size();
        if (!(landmark->used_num >= kLineMinObs && landmark->start_frame < kWinSize - 2 && landmark->is_triangulation))
            continue;

        int imu_i = landmark->start_frame, imu_j = imu_i - 1;

        Vec3d twc = Ps[imu_i] + Rs[imu_i] * tic[0];   // twc = Rwi * tic + twi
        Mat3d Rwc = Rs[imu_i] * ric[0];               // Rwc = Rwi * Ric

        // 计算初始帧上线段对应的3d端点
        Vec3d pc, nc, vc;
        nc = landmark->line_plucker.head(3);
        vc = landmark->line_plucker.tail(3);

        //       double  d = nc.norm()/vc.norm();
        //       if (d > 5.0)
        {
            //           std::cerr <<"remove a large distant line \n";
            //           linefeature.erase(it_per_id);
            //           continue;
        }

        Mat4d Lc;
        Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

        Vec4d obs_startframe = landmark->feats[0].line_obs;   // 第一次观测到这帧
        Vec3d p11 = Vec3d(obs_startframe(0), obs_startframe(1), 1.0);
        Vec3d p21 = Vec3d(obs_startframe(2), obs_startframe(3), 1.0);
        Vec2d ln = ( p11.cross(p21) ).head(2);     // 直线的垂直方向
        ln = ln / ln.norm();

        Vec3d p12 = Vec3d(p11(0) + ln(0), p11(1) + ln(1), 1.0);  // 直线垂直方向上移动一个单位
        Vec3d p22 = Vec3d(p21(0) + ln(0), p21(1) + ln(1), 1.0);
        Vec3d cam = Vec3d( 0, 0, 0 );

        Vec4d pi1 = pi_from_ppp(cam, p11, p12);
        Vec4d pi2 = pi_from_ppp(cam, p21, p22);

        Vec4d e1 = Lc * pi1;
        Vec4d e2 = Lc * pi2;
        e1 = e1/e1(3);
        e2 = e2/e2(3);

        if(e1(2) < 0 || e2(2) < 0){
            line_landmarks.erase(landmark);
            continue;
        }
        if((e1-e2).norm() > 10){
            line_landmarks.erase(landmark);
            continue;
        }

        /*
                // 点到直线的距离不能太远啊
                Vec3d Q = plucker_origin(nc,vc);
                if(Q.norm() > 5.0)
                {
                    linefeature.erase(it_per_id);
                    continue;
                }
        */
        // 并且平均投影误差不能太大啊
        Vec6d line_w = plk_to_pose(landmark->line_plucker, Rwc, twc);  // transfrom to world frame

        int i = 0;
        double allerr = 0;
        Vec3d tij;
        Mat3d Rij;
        Vec4d obs;

        for (auto &it_per_frame : landmark->feats) {   // 遍历所有的观测， 注意 start_frame 也会被遍历
            imu_j++;
            obs = it_per_frame.line_obs;
            Vec3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Mat3d R1 = Rs[imu_j] * ric[0];

            double err =  LineReprojectionError(obs, R1, t1, line_w);

            //            if(err > 0.0000001)
            //                i++;
            //            allerr += err;    // 计算平均投影误差

            if(allerr < err)    // 记录最大投影误差，如果最大的投影误差比较大，那就说明有outlier
                allerr = err;
            }
        //        allerr = allerr / i;
        if (allerr > 3.0 / 500.0){
            line_landmarks.erase(landmark);
        }
    }
}



/**
 * 剔除特征点
 * @param outlierIndex
 */
void FeatureManager::RemoveOutlier(std::set<int> &outlierIndex){
    std::set<int>::iterator itSet;
    for (auto it = point_landmarks.begin(), it_next = point_landmarks.begin(); it != point_landmarks.end(); it = it_next){
        it_next++;
        int index = it->feature_id;
        itSet = outlierIndex.find(index);
        if(itSet != outlierIndex.end()){
            point_landmarks.erase(it);
        }
    }
}

void FeatureManager::RemoveBackShiftDepth(const Mat3d& marg_R, const Vec3d& marg_P, Mat3d new_R, Vec3d new_P){
    for (auto it = point_landmarks.begin(), it_next = point_landmarks.begin(); it != point_landmarks.end(); it = it_next){
        it_next++;

        if (it->start_frame != 0){
            it->start_frame--;
        }
        else{
            Vec3d uv_i = it->feats[0].point;
            it->feats.erase(it->feats.begin());
            if (it->feats.size() < 2){
                point_landmarks.erase(it);
                continue;
            }
            else{
                Vec3d pts_i = uv_i * it->depth;
                Vec3d w_pts_i = marg_R * pts_i + marg_P;
                Vec3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->depth = dep_j;
                else
                    it->depth = para::kInitDepth;
            }
        }
        /*
        // remove tracking-lost feature after marginalize
        if (it->endFrame() < kWindowSize - 1)
        {
            feature.erase(it);
        }
        */
    }


    for (auto it = line_landmarks.begin(), it_next = line_landmarks.begin();it != line_landmarks.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0) {    // 如果特征不是在这帧上初始化的，那就不用管，只要管id--
            it->start_frame--;
        }
        else{
            /*
                        //  used to debug
                        Vector3d pc, nc, vc;
                        nc = it->line_plucker.head(3);
                        vc = it->line_plucker.tail(3);

                        Matrix4d Lc;
                        Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

                        Vector4d obs_startframe = it->linefeature_per_frame[0].lineobs;   // 第一次观测到这帧
                        Vector3d p11 = Vector3d(obs_startframe(0), obs_startframe(1), 1.0);
                        Vector3d p21 = Vector3d(obs_startframe(2), obs_startframe(3), 1.0);
                        Vector2d ln = ( p11.cross(p21) ).head(2);     // 直线的垂直方向
                        ln = ln / ln.norm();

                        Vector3d p12 = Vector3d(p11(0) + ln(0), p11(1) + ln(1), 1.0);  // 直线垂直方向上移动一个单位
                        Vector3d p22 = Vector3d(p21(0) + ln(0), p21(1) + ln(1), 1.0);
                        Vector3d cam = Vector3d( 0, 0, 0 );

                        Vector4d pi1 = pi_from_ppp(cam, p11, p12);
                        Vector4d pi2 = pi_from_ppp(cam, p21, p22);

                        Vector4d e1 = Lc * pi1;
                        Vector4d e2 = Lc * pi2;
                        e1 = e1/e1(3);
                        e2 = e2/e2(3);

                        Vector3d pts_1(e1(0),e1(1),e1(2));
                        Vector3d pts_2(e2(0),e2(1),e2(2));

                        Vector3d w_pts_1 =  marg_R * pts_1 + marg_P;
                        Vector3d w_pts_2 =  marg_R * pts_2 + marg_P;

                        std::cout<<"-------------------------------\n";
                        std::cout << w_pts_1 << "\n" <<w_pts_2 <<"\n\n";
                        Vector4d obs_startframe = it->linefeature_per_frame[0].lineobs;   // 第一次观测到这帧
                        */
            //-----------------
            it->feats.erase(it->feats.begin());  // 移除观测
            if (it->feats.size() < 2){  // 如果观测到这个帧的图像少于两帧，那这个特征不要了
                line_landmarks.erase(it);
                continue;
            }
            else  // 如果还有很多帧看到它，而我们又把这个特征的初始化帧给marg掉了，那就得把这个特征转挂到下一帧上去, 这里 marg_R, new_R 都是相应时刻的相机坐标系到世界坐标系的变换
            {
                it->removed_cnt++;
                // transpose this line to the new pose
                Mat3d Rji = new_R.transpose() * marg_R;     // Rcjw * Rwci
                Vec3d tji = new_R.transpose() * (marg_P - new_P);
                Vec6d plk_j = plk_to_pose(it->line_plucker, Rji, tji);
                it->line_plucker = plk_j;
            }
            //-----------------------
            /*
                        //  used to debug
                        nc = it->line_plucker.head(3);
                        vc = it->line_plucker.tail(3);

                        Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

                        obs_startframe = it->linefeature_per_frame[0].lineobs;   // 第一次观测到这帧
                        p11 = Vector3d(obs_startframe(0), obs_startframe(1), 1.0);
                        p21 = Vector3d(obs_startframe(2), obs_startframe(3), 1.0);
                        ln = ( p11.cross(p21) ).head(2);     // 直线的垂直方向
                        ln = ln / ln.norm();

                        p12 = Vector3d(p11(0) + ln(0), p11(1) + ln(1), 1.0);  // 直线垂直方向上移动一个单位
                        p22 = Vector3d(p21(0) + ln(0), p21(1) + ln(1), 1.0);
                        cam = Vector3d( 0, 0, 0 );

                        pi1 = pi_from_ppp(cam, p11, p12);
                        pi2 = pi_from_ppp(cam, p21, p22);

                        e1 = Lc * pi1;
                        e2 = Lc * pi2;
                        e1 = e1/e1(3);
                        e2 = e2/e2(3);

                        pts_1 = Vector3d(e1(0),e1(1),e1(2));
                        pts_2 = Vector3d(e2(0),e2(1),e2(2));

                        w_pts_1 =  new_R * pts_1 + new_P;
                        w_pts_2 =  new_R * pts_2 + new_P;

                        std::cout << w_pts_1 << "\n" <<w_pts_2 <<"\n";
            */
        }
    }

}



void FeatureManager::RemoveBack()
{
    for (auto it = point_landmarks.begin(), it_next = point_landmarks.begin(); it != point_landmarks.end(); it = it_next){
        it_next++;
        if (it->start_frame != 0){
            it->start_frame--;
        }
        else{
            it->feats.erase(it->feats.begin());
            if (it->feats.empty())
                point_landmarks.erase(it);
        }
    }

    for (auto it = line_landmarks.begin(), it_next = line_landmarks.begin();it != line_landmarks.end(); it = it_next){
        it_next++;

        // 如果这个特征不是在窗口里最老关键帧上观测到的，由于窗口里移除掉了一个帧，所有其他特征对应的初始化帧id都要减1左移
        // 例如： 窗口里有 0,1,2,3,4 一共5个关键帧，特征f2在第2帧上三角化的， 移除掉第0帧以后， 第2帧在窗口里的id就左移变成了第1帧，这是很f2的start_frame对应减1
        if (it->start_frame != 0)
            it->start_frame--;
        else{
            it->feats.erase(it->feats.begin());  // 删掉特征ft在这个图像帧上的观测量
            if (it->feats.empty())                       // 如果没有其他图像帧能看到这个特征ft了，那就直接删掉它
                line_landmarks.erase(it);
        }
    }
}


void FeatureManager::RemoveFront(int frame_count)
{
    for (auto it = point_landmarks.begin(), it_next = point_landmarks.begin(); it != point_landmarks.end(); it = it_next){
        it_next++;
        if (it->start_frame == frame_count){
            it->start_frame--;
        }
        else{
            int j = kWinSize - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feats.erase(it->feats.begin() + j);
            if (it->feats.empty())
                point_landmarks.erase(it);
        }
    }

    for (auto it = line_landmarks.begin(), it_next = line_landmarks.begin(); it != line_landmarks.end(); it = it_next){
        it_next++;
        if (it->start_frame == frame_count) {  // 由于要删去的是第frame_count-1帧，最新这一帧frame_count的id就变成了i-1
            it->start_frame--;
        }
        else{
            int j = kWinSize - 1 - it->start_frame;    // j指向第i-1帧
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feats.erase(it->feats.begin() + j);   // 删掉特征ft在这个图像帧上的观测量
            if (it->feats.empty())                            // 如果没有其他图像帧能看到这个特征ft了，那就直接删掉它
                line_landmarks.erase(it);
        }
    }
}



double FeatureManager::CompensatedParallax2(const FeaturePerId &landmark, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = landmark.feats[frame_count - 2 - landmark.start_frame];
    const FeaturePerFrame &frame_j = landmark.feats[frame_count - 1 - landmark.start_frame];

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

