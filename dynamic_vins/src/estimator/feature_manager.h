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
#include "estimator/basic/frontend_feature.h"
#include "estimator/basic/line_landmark.h"

namespace dynamic_vins{\

class FeatureManager
{
  public:
    FeatureManager();

    void ClearState();
    int GetFeatureCount();

    bool AddFeatureCheckParallax(
            int frame_count,
            const std::map<unsigned int, std::vector<std::pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
            double td);

    bool AddFeatureCheckParallax(int frame_count, FeatureBackground &image, double td);

    vector<pair<Vec3d, Vec3d>> GetCorresponding(int frame_count_l, int frame_count_r);
    //void updateDepth(const VectorXd &x);
    void SetDepth(const Eigen::VectorXd &x);
    void RemoveFailures();

    void ClearDepth();
    Eigen::VectorXd GetDepthVector();

    int GetLineFeatureCount();

    Eigen::MatrixXd GetLineOrthVector();

    Eigen::MatrixXd GetLineOrthVectorInCamera();

    void SetLineOrth(Eigen::MatrixXd &x);

    void TriangulatePoints();

    void TriangulateLineMono();

    void TriangulateLineStereo(double baseline);  // stereo line

    void RemoveBackShiftDepth(const Mat3d& marg_R, const Vec3d& marg_P, Mat3d new_R, Vec3d new_P);

    void RemoveBack();

    void RemoveFront(int frame_count);

    void RemoveOutlier(std::set<int> &outlierIndex);

    void RemoveLineOutlierByLength();

    void RemoveLineOutlier();

public:
    std::list<StaticLandmark> point_landmarks;
    std::list<LineLandmark> line_landmarks;
    int last_track_num;

private:
    double last_average_parallax;
    int new_feature_num;
    int long_track_num;
};


}

#endif
