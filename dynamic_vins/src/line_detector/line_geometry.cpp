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

#include "line_geometry.h"

namespace dynamic_vins{\


Vec4d line_to_orth(Vec6d line)
{
    Vec4d orth;
    Vec3d p = line.head(3);
    Vec3d v = line.tail(3);
    Vec3d n = p.cross(v);

    Vec3d u1 = n/n.norm();
    Vec3d u2 = v/v.norm();
    Vec3d u3 = u1.cross(u2);

    orth[0] = atan2( u2(2),u3(2) );
    orth[1] = asin( -u1(2) );
    orth[2] = atan2( u1(1),u1(0) );

    Vec2d w( n.norm(), v.norm() );
    w = w/w.norm();
    orth[3] = asin( w(1) );

    return orth;

}
Vec6d orth_to_line(Vec4d orth)
{
    Vec6d line;

    Vec3d theta = orth.head(3);
    double phi = orth[3];

    // todo:: SO3
    double s1 = sin(theta[0]);
    double c1 = cos(theta[0]);
    double s2 = sin(theta[1]);
    double c2 = cos(theta[1]);
    double s3 = sin(theta[2]);
    double c3 = cos(theta[2]);

    Mat3d R;
    R <<
    c2 * c3,   s1 * s2 * c3 - c1 * s3,   c1 * s2 * c3 + s1 * s3,
    c2 * s3,   s1 * s2 * s3 + c1 * c3,   c1 * s2 * s3 - s1 * c3,
    -s2,                  s1 * c2,                  c1 * c2;

    double w1 = cos(phi);
    double w2 = sin(phi);
    double d = w1/w2;      // 原点到直线的距离

    line.head(3) = -R.col(2) * d;
    line.tail(3) = R.col(1);

    return line;


}

Vec4d plk_to_orth(Vec6d plk)
{
    Vec4d orth;
    Vec3d n = plk.head(3);
    Vec3d v = plk.tail(3);

    Vec3d u1 = n/n.norm();
    Vec3d u2 = v/v.norm();
    Vec3d u3 = u1.cross(u2);

    // todo:: use SO3
    orth[0] = atan2( u2(2),u3(2) );
    orth[1] = asin( -u1(2) );
    orth[2] = atan2( u1(1),u1(0) );

    Vec2d w( n.norm(), v.norm() );
    w = w/w.norm();
    orth[3] = asin( w(1) );
    return orth;
}


Vec6d orth_to_plk(Vec4d orth)
{
    Vec6d plk;

    Vec3d theta = orth.head(3);
    double phi = orth[3];

    double s1 = sin(theta[0]);
    double c1 = cos(theta[0]);
    double s2 = sin(theta[1]);
    double c2 = cos(theta[1]);
    double s3 = sin(theta[2]);
    double c3 = cos(theta[2]);

    ///计算U矩阵
    Mat3d R;
    R <<
    c2 * c3,   s1 * s2 * c3 - c1 * s3,   c1 * s2 * c3 + s1 * s3,
    c2 * s3,   s1 * s2 * s3 + c1 * c3,   c1 * s2 * s3 - s1 * c3,
    -s2,                  s1 * c2,                  c1 * c2;

    double w1 = cos(phi);
    double w2 = sin(phi);
    double d = w1/w2;      // 原点到直线的距离

    Vec3d u1 = R.col(0);
    Vec3d u2 = R.col(1);

    Vec3d n = w1 * u1;
    Vec3d v = w2 * u2;

    plk.head(3) = n;
    plk.tail(3) = v;

    //Vec3d Q = -R.col(2) * d;
    //plk.head(3) = Q.cross(v);
    //plk.tail(3) = v;

    return plk;


}

/*
 三点确定一个平面 a(x-x0)+b(y-y0)+c(z-z0)=0  --> ax + by + cz + d = 0   d = -(ax0 + by0 + cz0)
 平面通过点（x0,y0,z0）以及垂直于平面的法线（a,b,c）来得到
 (a,b,c)^T = vector(AO) cross vector(BO)
 d = O.dot(cross(AO,BO))
 */
Vec4d pi_from_ppp(Vec3d x1, Vec3d x2, Vec3d x3) {
    Vec4d pi;
    pi << ( x1 - x3 ).cross( x2 - x3 ), - x3.dot( x1.cross( x2 ) ); // d = - x3.dot( (x1-x3).cross( x2-x3 ) ) = - x3.dot( x1.cross( x2 ) )

    return pi;
}

// 两平面相交得到直线的plucker 坐标
Vec6d pipi_plk( Vec4d pi1, Vec4d pi2){
    Vec6d plk;
    Mat4d dp = pi1 * pi2.transpose() - pi2 * pi1.transpose();

    plk << dp(0,3), dp(1,3), dp(2,3), - dp(1,2), dp(0,2), - dp(0,1);
    return plk;
}

// 获取光心到直线的垂直点
Vec3d plucker_origin(Vec3d n, Vec3d v) {
    return v.cross(n) / v.dot(v);
}

Mat3d skew_symmetric( Vec3d v ) {
    Mat3d S;
    S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
    return S;
}



Vec3d point_to_pose( Mat3d Rcw, Vec3d tcw , Vec3d pt_w ) {
    return Rcw * pt_w + tcw;
}

// 从相机坐标系到世界坐标系
Vec3d poit_from_pose( Mat3d Rcw, Vec3d tcw, Vec3d pt_c ) {

    Mat3d Rwc = Rcw.transpose();
    Vec3d twc = -Rwc*tcw;
    return point_to_pose( Rwc, twc, pt_c );
}

Vec6d line_to_pose(Vec6d line_w, Mat3d Rcw, Vec3d tcw) {
    Vec6d line_c;

    Vec3d cp_w, dv_w;
    cp_w = line_w.head(3);
    dv_w = line_w.tail(3);

    Vec3d cp_c = point_to_pose( Rcw, tcw, cp_w );
    Vec3d dv_c = Rcw* dv_w;

    line_c.head(3) = cp_c;
    line_c.tail(3) = dv_c;

    return line_c;
}

Vec6d line_from_pose(Vec6d line_c, Mat3d Rcw, Vec3d tcw) {
    Mat3d Rwc = Rcw.transpose();
    Vec3d twc = -Rwc*tcw;
    return line_to_pose( line_c, Rwc, twc );
}

/// 世界坐标系到相机坐标系下
Vec6d plk_to_pose( Vec6d plk_w, Mat3d Rcw, Vec3d tcw ) {
    Vec3d nw = plk_w.head(3);
    Vec3d vw = plk_w.tail(3);

    Vec3d nc = Rcw * nw + skew_symmetric(tcw) * Rcw * vw;
    Vec3d vc = Rcw * vw;

    Vec6d plk_c;
    plk_c.head(3) = nc;
    plk_c.tail(3) = vc;
    return plk_c;
}

///将线从相机坐标系转换到世界坐标系
Vec6d plk_from_pose( Vec6d plk_c, Mat3d Rcw, Vec3d tcw ) {

    Mat3d Rwc = Rcw.transpose();
    Vec3d twc = -Rwc*tcw;
    return plk_to_pose( plk_c, Rwc, twc);
}


double LineReprojectionError( Vec4d obs, Mat3d Rwc, Vec3d twc, Vec6d line_w ) {

    double error = 0;

    Vec3d n_w, d_w;
    n_w = line_w.head(3);
    d_w = line_w.tail(3);

    Vec3d p1, p2;
    p1 << obs[0], obs[1], 1;
    p2 << obs[2], obs[3], 1;

    Vec6d line_c = plk_from_pose(line_w,Rwc,twc);
    Vec3d nc = line_c.head(3);
    double sql = nc.head(2).norm();
    nc /= sql;

    error += fabs( nc.dot(p1) );
    error += fabs( nc.dot(p2) );

    return error / 2.0;
}


/**
 * 获得3D直线的两个端点
 * @param plucker
 * @return 是否有效，两个3D点在相机坐标系下的坐标
 */
tuple<bool,Vec3d,Vec3d> LineTrimming(const Vec6d &plucker,const Vec4d &line_obs){
    Vec3d nc = plucker.head(3);
    Vec3d vc = plucker.tail(3);

    //plucker矩阵
    Mat4d Lc;
    Lc << skew_symmetric(nc), vc, -vc.transpose(), 0;

    Vec3d p11 = Vec3d(line_obs(0), line_obs(1), 1.0);
    Vec3d p21 = Vec3d(line_obs(2), line_obs(3), 1.0);
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

    Vec3d pts_1(e1(0),e1(1),e1(2));//直线在相机坐标系下的端点
    Vec3d pts_2(e2(0),e2(1),e2(2));

    bool valid = e1[2]>=0 && e2[2]>=0;

    return {valid,pts_1,pts_2};
}



}
