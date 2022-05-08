/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
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



#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>

#include <eigen3/Eigen/Dense>
#include <ros/console.h>
#include <ros/assert.h>

#include "utils/def.h"
#include "utils/parameters.h"
#include "feature_queue.h"


namespace dynamic_vins{\


class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
        is_stereo = false;
    }
    void rightObservation(const Eigen::Matrix<double, 7, 1> &_point)
    {
        pointRight.x() = _point(0);
        pointRight.y() = _point(1);
        pointRight.z() = _point(2);
        uvRight.x() = _point(3);
        uvRight.y() = _point(4);
        velocityRight.x() = _point(5); 
        velocityRight.y() = _point(6); 
        is_stereo = true;
    }
    double cur_td;
    Vec3d point, pointRight;
    Vec2d uv, uvRight;
    Vec2d velocity, velocityRight;
    bool is_stereo;
};

class FeaturePerId
{
  public:
    const int feature_id;
    int start_frame;
    vector<FeaturePerFrame> feature_per_frame;
    int used_num;
    double estimated_depth;
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail;

    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }

    int endFrame();
};

class FeatureManager
{
  public:
    FeatureManager(Mat3d _Rs[]);

    void SetRic(Mat3d _ric[]);
    void ClearState();
    int GetFeatureCount();
    bool AddFeatureCheckParallax(int frame_count, const FeatureBackground &image, double td);
    vector<pair<Vec3d, Vec3d>> GetCorresponding(int frame_count_l, int frame_count_r);
    //void updateDepth(const VectorXd &x);
    void SetDepth(const Eigen::VectorXd &x);
    void RemoveFailures();
    void ClearDepth();
    Eigen::VectorXd GetDepthVector();
    void triangulate(int frameCnt, Vec3d Ps[], Mat3d Rs[], Vec3d tic[], Mat3d ric[]);
    void TriangulatePoint(Mat34d &Pose0, Mat34d &Pose1,
                          Vec2d &point0, Vec2d &point1, Vec3d &point_3d);
    void initFramePoseByPnP(int frameCnt, Vec3d Ps[], Mat3d Rs[], Vec3d tic[], Mat3d ric[]);
    static bool SolvePoseByPnP(Mat3d &R_initial, Vec3d &P_initial,
                        vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);
    void RemoveBackShiftDepth(const Mat3d& marg_R, const Vec3d& marg_P, Mat3d new_R, Vec3d new_P);
    void RemoveBack();
    void RemoveFront(int frame_count);
    void RemoveOutlier(std::set<int> &outlierIndex);

    std::list<FeaturePerId> feature;
    int last_track_num;

  private:
    static double CompensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Mat3d *Rs;
    Mat3d ric[2];

    double last_average_parallax;
    int new_feature_num;
    int long_track_num;
};


}

#endif