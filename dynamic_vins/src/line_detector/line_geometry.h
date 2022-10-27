//
// Created by hyj on 17-12-8.
//

/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#ifndef VINS_ESTIMATOR_LINE_GEOMETRY_H
#define VINS_ESTIMATOR_LINE_GEOMETRY_H

#include <eigen3/Eigen/Dense>

#include "utils/def.h"

namespace dynamic_vins{\


Vec4d line_to_orth(Vec6d line);
Vec6d orth_to_line(Vec4d orth);
Vec4d plk_to_orth(Vec6d plk);
Vec6d orth_to_plk(Vec4d orth);

Vec4d pi_from_ppp(Vec3d x1, Vec3d x2, Vec3d x3);
Vec6d pipi_plk( Vec4d pi1, Vec4d pi2);
Vec3d plucker_origin(Vec3d n, Vec3d v);
Mat3d skew_symmetric( Vec3d v );

Vec3d poit_from_pose( Mat3d Rcw, Vec3d tcw, Vec3d pt_c );
Vec3d point_to_pose( Mat3d Rcw, Vec3d tcw , Vec3d pt_w );
Vec6d line_to_pose(Vec6d line_w, Mat3d Rcw, Vec3d tcw);
Vec6d line_from_pose(Vec6d line_c, Mat3d Rcw, Vec3d tcw);

Vec6d plk_to_pose( Vec6d plk_w, Mat3d Rcw, Vec3d tcw );
Vec6d plk_from_pose( Vec6d plk_c, Mat3d Rcw, Vec3d tcw );


double LineReprojectionError( Vec4d obs, Mat3d Rwc, Vec3d twc, Vec6d line_w );

tuple<bool,Vec3d,Vec3d> LineTrimming(const Vec6d &plucker,const Vec4d &line_obs);


}

#endif //VINS_ESTIMATOR_LINE_GEOMETRY_H
