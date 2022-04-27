/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>


namespace dynamic_vins{\


class MotionEstimator{
  public:

    bool solveRelativeRT(const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> &corres, Eigen::Matrix3d &R, Eigen::Vector3d &T);

  private:
    double testTriangulation(const std::vector<cv::Point2f> &l,
                             const std::vector<cv::Point2f> &r,
                             cv::Mat_<double> R, cv::Mat_<double> t);
    void decomposeE(cv::Mat E,
                    cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                    cv::Mat_<double> &t1, cv::Mat_<double> &t2);
};

}
