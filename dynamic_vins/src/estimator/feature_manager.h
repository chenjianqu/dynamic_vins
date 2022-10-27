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
#include "semantic_feature.h"
#include "line_landmark.h"


namespace dynamic_vins{\


class FeatureManager
{
  public:
    FeatureManager(Mat3d _Rs[]);

    void SetRic(Mat3d _ric[]);
    void ClearState();
    int GetFeatureCount();

    bool AddFeatureCheckParallax(int frame_count, const std::map<unsigned int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    bool AddFeatureCheckParallax(int frame_count, FeatureBackground &image, double td);

    vector<pair<Vec3d, Vec3d>> GetCorresponding(int frame_count_l, int frame_count_r);
    //void updateDepth(const VectorXd &x);
    void SetDepth(const Eigen::VectorXd &x);
    void RemoveFailures();

    void ClearDepth();
    Eigen::VectorXd GetDepthVector();

    int GetLineFeatureCount();

    Eigen::MatrixXd GetLineOrthVector(Vec3d Ps[], Vec3d tic[], Mat3d ric[]);

    Eigen::MatrixXd GetLineOrthVectorInCamera();


    void SetLineOrth(Eigen::MatrixXd x,Vec3d P[], Mat3d R[], Vec3d tic[], Mat3d ric[]);

    void TriangulatePoint(int frameCnt, Vec3d Ps[], Mat3d Rs[], Vec3d tic[], Mat3d ric[]);

    static void TriangulatePoint(Mat34d &Pose0, Mat34d &Pose1,
                          Vec2d &point0, Vec2d &point1, Vec3d &point_3d);

    void TriangulateLineMono();

    void TriangulateLineStereo(double baseline);  // stereo line

    void InitFramePoseByPnP(int frameCnt, Vec3d Ps[], Mat3d Rs[], Vec3d tic[], Mat3d ric[]);

    void RemoveBackShiftDepth(const Mat3d& marg_R, const Vec3d& marg_P, Mat3d new_R, Vec3d new_P);
    void RemoveBack();
    void RemoveFront(int frame_count);

    void RemoveOutlier(std::set<int> &outlierIndex);

    void RemoveLineOutlierByLength();

    void RemoveLineOutlier();

    static bool SolvePoseByPnP(Mat3d &R_initial, Vec3d &P_initial,
                               vector<cv::Point2f> &pts2D, vector<cv::Point3f> &pts3D);

    std::list<FeaturePerId> point_landmarks;
    std::list<LineLandmark> line_landmarks;
    int last_track_num;

private:
    static double CompensatedParallax2(const FeaturePerId &landmark, int frame_count);
    const Mat3d *Rs;
    Mat3d ric[2];

    double last_average_parallax;
    int new_feature_num;
    int long_track_num;
};


}

#endif
