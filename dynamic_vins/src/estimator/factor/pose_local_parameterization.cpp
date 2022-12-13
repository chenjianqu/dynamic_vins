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

#include "pose_local_parameterization.h"
#include "utils/def.h"
#include "utils/parameters.h"

namespace dynamic_vins{\


bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}



/**
 * 只关注在平面上的运动
 * @param x
 * @param delta
 * @param x_plus_delta
 * @return
 */
bool PoseConstraintLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);

    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    //由于使用和IMU和vision-only两种情况下的世界坐标系定义不同
    if(cfg::use_imu){
        Vec3d dp(delta[0],delta[1],0);

        Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

        Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

        p = _p + dp;
        q = (_q * dq).normalized();
    }
    else{
        Vec3d dp(delta[0],0,delta[2]);

        Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

        Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
        Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

        p = _p + dp;
        q = (_q * dq).normalized();
    }


    return true;
}


bool PoseConstraintLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}



bool SpeedConstraintLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Vec6d> _v(x);

    //Eigen::Map<const Eigen::Vector3d> dp(delta);
    Vec6d dv;
    if(cfg::use_imu){
        dv(0,0) = delta[0];
        dv(1,0) = delta[1];
        dv(2,0) = 0;
        dv(3,0) = 0;
        dv(4,0) = 0;
        dv(5,0) = delta[5];
    }
    else{
        dv(0,0) = delta[0];
        dv(1,0) = 0;
        dv(2,0) = delta[2];
        dv(3,0) = delta[3];
        dv(4,0) = delta[4];
        dv(5,0) = delta[5];
    }

    Eigen::Map<Vec6d> v(x_plus_delta);

    v = _v + dv;

    return true;
}


bool SpeedConstraintLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> j(jacobian);
    j.setZero();

    return true;
}


}
