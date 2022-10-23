
#include "line_projection_factor.h"

#include "line_parameterization.h"
#include "line_detector/line_geometry.h"
#include "utils/def.h"

namespace dynamic_vins{\


Eigen::Matrix2d lineProjectionFactor::sqrt_info;
double lineProjectionFactor::sum_t;

lineProjectionFactor::lineProjectionFactor(const Eigen::Vector4d &_obs_i) : obs_i(_obs_i)
{
};


/*
  parameters[0]:  Twi
  parameters[1]:  Tbc
  parameters[2]:  line_orth
*/
bool lineProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d tic(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector4d line_orth( parameters[2][0],parameters[2][1],parameters[2][2],parameters[2][3]);
    Vec6d line_w = orth_to_plk(line_orth);

    ///将直线从世界坐标系转换到相机坐标系
    Mat3d Rwb(Qi);
    Vec3d twb(Pi);
    Vec6d line_b = plk_from_pose(line_w, Rwb, twb);
    Mat3d Rbc(qic);
    Vec3d tbc(tic);
    Vec6d line_c = plk_from_pose(line_b, Rbc, tbc);

    //由于此时直线在归一化平面上，因此直线的投影矩阵K为单位阵

    ///误差计算
    Vec3d nc = line_c.head(3);
    double l_norm = nc(0) * nc(0) + nc(1) * nc(1);
    double l_sqrtnorm = sqrt( l_norm );
    double l_trinorm = l_norm * l_sqrtnorm;

    double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
    double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);

    Eigen::Map<Vec2d> residual(residuals);
    residual(0) = e1/l_sqrtnorm;
    residual(1) = e2/l_sqrtnorm;

    //    sqrt_info.setIdentity();
    residual = sqrt_info * residual;

    if (jacobians){
        ///误差对投影直线的导数
        Eigen::Matrix<double, 2, 3> jaco_e_l(2, 3);
        jaco_e_l << (obs_i(0)/l_sqrtnorm - nc(0) * e1 / l_trinorm ), (obs_i(1)/l_sqrtnorm - nc(1) * e1 / l_trinorm ), 1.0/l_sqrtnorm,
        (obs_i(2)/l_sqrtnorm - nc(0) * e2 / l_trinorm ), (obs_i(3)/l_sqrtnorm - nc(1) * e2 / l_trinorm ), 1.0/l_sqrtnorm;

        jaco_e_l = sqrt_info * jaco_e_l;

        Eigen::Matrix<double, 3, 6> jaco_l_Lc(3, 6);
        jaco_l_Lc.setZero();
        jaco_l_Lc.block(0,0,3,3) = Mat3d::Identity();

        Eigen::Matrix<double, 2, 6> jaco_e_Lc;
        jaco_e_Lc = jaco_e_l * jaco_l_Lc;

        ///对位姿求导
        if (jacobians[0]){
            //std::cout <<"jacobian_pose_i"<<"\n";
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double,6,6> invTbc;
            invTbc << Rbc.transpose(), -Rbc.transpose()*skew_symmetric(tbc),
            Mat3d::Zero(),  Rbc.transpose();

            Vec3d nw = line_w.head(3);
            Vec3d dw = line_w.tail(3);
            Eigen::Matrix<double, 6, 6> jaco_Lc_pose;
            jaco_Lc_pose.setZero();
            ///对位置的雅可比
            jaco_Lc_pose.block(0,0,3,3) = Rwb.transpose() * skew_symmetric(dw);   // Lc_t
            ///对方向的雅可比
            jaco_Lc_pose.block(0,3,3,3) = skew_symmetric( Rwb.transpose() * (nw + skew_symmetric(dw) * twb) );  // Lc_theta
            jaco_Lc_pose.block(3,3,3,3) = skew_symmetric( Rwb.transpose() * dw);

            jaco_Lc_pose = invTbc * jaco_Lc_pose;

            jacobian_pose_i.leftCols<6>() = jaco_e_Lc * jaco_Lc_pose;

            jacobian_pose_i.rightCols<1>().setZero();            //最后一列设成0
        }

        ///对外参求导
        if (jacobians[1]){
            //std::cout <<"jacobian_ex_pose"<<"\n";
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[1]);

            Vec3d nb = line_b.head(3);
            Vec3d db = line_b.tail(3);
            Eigen::Matrix<double, 6, 6> jaco_Lc_ex;
            jaco_Lc_ex.setZero();
            jaco_Lc_ex.block(0,0,3,3) = Rbc.transpose() * skew_symmetric(db);   // Lc_t
            jaco_Lc_ex.block(0,3,3,3) = skew_symmetric( Rbc.transpose() * (nb + skew_symmetric(db) * tbc) );  // Lc_theta
            jaco_Lc_ex.block(3,3,3,3) = skew_symmetric( Rbc.transpose() * db);

            jacobian_ex_pose.leftCols<6>() = jaco_e_Lc * jaco_Lc_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }

        ///对正交表示求导
        if (jacobians[2]){
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jacobian_lineOrth(jacobians[2]);

            Mat3d Rwc = Rwb * Rbc;
            Vec3d twc = Rwb * tbc + twb;
            Eigen::Matrix<double,6,6> invTwc;
            invTwc << Rwc.transpose(), -Rwc.transpose() * skew_symmetric(twc),
            Mat3d::Zero(),  Rwc.transpose();
            //std::cout<<invTwc<<"\n";

            Vec3d nw = line_w.head(3);
            Vec3d vw = line_w.tail(3);
            Vec3d u1 = nw/nw.norm();
            Vec3d u2 = vw/vw.norm();
            Vec3d u3 = u1.cross(u2);
            Vec2d w( nw.norm(), vw.norm() );
            w = w/w.norm();

            Eigen::Matrix<double, 6, 4> jaco_Lw_orth;
            jaco_Lw_orth.setZero();
            jaco_Lw_orth.block(3,0,3,1) = w[1] * u3;
            jaco_Lw_orth.block(0,1,3,1) = -w[0] * u3;
            jaco_Lw_orth.block(0,2,3,1) = w(0) * u2;
            jaco_Lw_orth.block(3,2,3,1) = -w(1) * u1;
            jaco_Lw_orth.block(0,3,3,1) = -w(1) * u1;
            jaco_Lw_orth.block(3,3,3,1) = w(0) * u2;

            jacobian_lineOrth = jaco_e_Lc * invTwc * jaco_Lw_orth;
        }

    }

    // check jacobian
    /*
        std::cout << "---------- check jacobian ----------\n";
        if(jacobians[0])
        std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jacobians[0]) << std::endl
                  << std::endl;
        if(jacobians[1])
        std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jacobians[1]) << std::endl
                  << std::endl;
        if(jacobians[2])
        std::cout << Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>>(jacobians[2]) << std::endl
                  << std::endl;
        const double eps = 1e-6;
        Eigen::Matrix<double, 2, 16> num_jacobian;
        for (int k = 0; k < 16; k++)
        {
            Vec3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
            Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

            Vec3d tic(parameters[1][0], parameters[1][1], parameters[1][2]);
            Eigen::Quaterniond qic(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

            Eigen::Vector4d line_orth( parameters[2][0],parameters[2][1],parameters[2][2],parameters[2][3]);
            ceres::LocalParameterization *local_parameterization_line = new LineOrthParameterization();

            int a = k / 3, b = k % 3;
            Vec3d delta = Vec3d(b == 0, b == 1, b == 2) * eps;

            if (a == 0)
                Pi += delta;
            else if (a == 1)
                Qi = Qi * Utility::deltaQ(delta);
            else if (a == 2)
                tic += delta;
            else if (a == 3)
                qic = qic * Utility::deltaQ(delta);
            else if (a == 4) {           // line orth的前三个元素
                Eigen::Vector4d line_new;
                Eigen::Vector4d delta_l;
                delta_l<< delta, 0.0;
                local_parameterization_line->Plus(line_orth.data(),delta_l.data(),line_new.data());
                line_orth = line_new;
            }
            else if (a == 5) {           // line orth的最后一个元素
                Eigen::Vector4d line_new;
                Eigen::Vector4d delta_l;
                delta_l.setZero();
                delta_l[3]= delta.x();
                local_parameterization_line->Plus(line_orth.data(),delta_l.data(),line_new.data());
                line_orth = line_new;
            }

            Vec6d line_w = orth_to_plk(line_orth);

            Mat3d Rwb(Qi);
            Vec3d twb(Pi);
            Vec6d line_b = plk_from_pose(line_w, Rwb, twb);

            Mat3d Rbc(qic);
            Vec3d tbc(tic);
            Vec6d line_c = plk_from_pose(line_b, Rbc, tbc);

            // 直线的投影矩阵K为单位阵
            Vec3d nc = line_c.head(3);
            double l_norm = nc(0) * nc(0) + nc(1) * nc(1);
            double l_sqrtnorm = sqrt( l_norm );
            double l_trinorm = l_norm * l_sqrtnorm;

            double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
            double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);
            Vec2d tmp_residual;
            tmp_residual(0) = e1/l_sqrtnorm;
            tmp_residual(1) = e2/l_sqrtnorm;
            tmp_residual = sqrt_info * tmp_residual;

            num_jacobian.col(k) = (tmp_residual - residual) / eps;

        }
        std::cout <<"num_jacobian:\n"<< num_jacobian <<"\n"<< std::endl;
    */

    return true;
}


//////////////////////////////////////////////////
Eigen::Matrix2d lineProjectionFactor_incamera::sqrt_info;
lineProjectionFactor_incamera::lineProjectionFactor_incamera(const Eigen::Vector4d &_obs_i) : obs_i(_obs_i)
{
};

/*
  parameters[0]:  Twi
  parameters[1]:  Twj
  parameters[2]:  Tbc
  parameters[3]:  line_orth
*/
bool lineProjectionFactor_incamera::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Vec3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vec3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vec3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    Eigen::Vector4d line_orth( parameters[3][0],parameters[3][1],parameters[3][2],parameters[3][3]);
    Vec6d line_ci = orth_to_plk(line_orth);

    Mat3d Rbc(qic);
    Vec3d tbc(tic);
    Vec6d line_bi = plk_to_pose(line_ci, Rbc, tbc);

    Mat3d Rwbi = Qi.toRotationMatrix();
    Vec3d twbi(Pi);
    Vec6d line_w = plk_to_pose(line_bi, Rwbi, twbi);

    Mat3d Rwbj = Qj.toRotationMatrix();
    Vec3d twbj(Pj);
    Vec6d line_bj = plk_from_pose(line_w, Rwbj, twbj);

    Vec6d line_cj = plk_from_pose(line_bj, Rbc, tbc);

    // 直线的投影矩阵K为单位阵
    Vec3d nc = line_cj.head(3);
    double l_norm = nc(0) * nc(0) + nc(1) * nc(1);
    double l_sqrtnorm = sqrt( l_norm );
    double l_trinorm = l_norm * l_sqrtnorm;

    double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
    double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);
    Eigen::Map<Vec2d> residual(residuals);
    residual(0) = e1/l_sqrtnorm;
    residual(1) = e2/l_sqrtnorm;

    sqrt_info.setIdentity();
    residual = sqrt_info * residual;
    //std::cout<< residual <<std::endl;
    if (jacobians)
    {

        Eigen::Matrix<double, 2, 3> jaco_e_l(2, 3);
        jaco_e_l << (obs_i(0)/l_sqrtnorm - nc(0) * e1 / l_trinorm ), (obs_i(1)/l_sqrtnorm - nc(1) * e1 / l_trinorm ), 1.0/l_sqrtnorm,
        (obs_i(2)/l_sqrtnorm - nc(0) * e2 / l_trinorm ), (obs_i(3)/l_sqrtnorm - nc(1) * e2 / l_trinorm ), 1.0/l_sqrtnorm;

        jaco_e_l = sqrt_info * jaco_e_l;

        Eigen::Matrix<double, 3, 6> jaco_l_Lc(3, 6);
        jaco_l_Lc.setZero();
        jaco_l_Lc.block(0,0,3,3) = Mat3d::Identity();

        Eigen::Matrix<double, 2, 6> jaco_e_Lc;
        jaco_e_Lc = jaco_e_l * jaco_l_Lc;
        //std::cout <<jaco_e_Lc<<"\n\n";
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            /*
                        Matrix6d invTbc;
                        invTbc << Rbc.transpose(), -Rbc.transpose()*skew_symmetric(tbc),
                                Mat3d::Zero(),  Rbc.transpose();

                        Matrix6d invTwbj;
                        invTwbj << Rwbj.transpose(), -Rwbj.transpose()*skew_symmetric(twbj),
                             en::Mat3d::Zero(),  Rwbj.transpose();
            */

            Mat3d Rwcj = Rwbj * Rbc;
            Vec3d twcj = Rwbj * tbc + twbj;
            Eigen::Matrix<double,6,6> invTwcj;
            invTwcj << Rwcj.transpose(), -Rwcj.transpose()*skew_symmetric(twcj),
            Mat3d::Zero(),  Rwcj.transpose();

            Vec3d nbi = line_bi.head(3);
            Vec3d dbi = line_bi.tail(3);
            Eigen::Matrix<double, 6, 6> jaco_Lc_pose;
            jaco_Lc_pose.setZero();
            jaco_Lc_pose.block(0,0,3,3) = - skew_symmetric(Rwbi * dbi);   // Lc_t
            jaco_Lc_pose.block(0,3,3,3) = -Rwbi * skew_symmetric( nbi) - skew_symmetric(twbi) * Rwbi * skew_symmetric(dbi);  // Lc_theta
            jaco_Lc_pose.block(3,3,3,3) = -Rwbi * skew_symmetric(dbi);

            //jaco_Lc_pose = invTbc * invTwbj * jaco_Lc_pose;
            jaco_Lc_pose = invTwcj * jaco_Lc_pose;
            jacobian_pose_i.leftCols<6>() = jaco_e_Lc * jaco_Lc_pose;
            jacobian_pose_i.rightCols<1>().setZero();            //最后一列设成0
        }

        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double,6,6> invTbc;
            invTbc << Rbc.transpose(), -Rbc.transpose()*skew_symmetric(tbc),
            Mat3d::Zero(),  Rbc.transpose();

            Vec3d nw = line_w.head(3);
            Vec3d dw = line_w.tail(3);
            Eigen::Matrix<double, 6, 6> jaco_Lc_pose;
            jaco_Lc_pose.setZero();
            jaco_Lc_pose.block(0,0,3,3) = Rwbj.transpose() * skew_symmetric(dw);   // Lc_t
            jaco_Lc_pose.block(0,3,3,3) = skew_symmetric( Rwbj.transpose() * (nw + skew_symmetric(dw) * twbj) );  // Lc_theta
            jaco_Lc_pose.block(3,3,3,3) = skew_symmetric( Rwbj.transpose() * dw);

            jaco_Lc_pose = invTbc * jaco_Lc_pose;
            jacobian_pose_j.leftCols<6>() = jaco_e_Lc * jaco_Lc_pose;

            jacobian_pose_j.rightCols<1>().setZero();            //最后一列设成0
        }

        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);

            Mat3d Rbjbi = Rwbj.transpose() * Rwbi;
            Mat3d Rcjci = Rbc.transpose() * Rbjbi * Rbc;
            Vec3d tcjci = Rbc * ( Rwbj.transpose() * (Rwbi * tbc + twbi - twbj) - tbc);

            Vec3d nci = line_ci.head(3);
            Vec3d dci = line_ci.tail(3);
            Eigen::Matrix<double, 6, 6> jaco_Lc_ex;
            jaco_Lc_ex.setZero();
            jaco_Lc_ex.block(0,0,3,3) = -Rbc.transpose() * Rbjbi * skew_symmetric( Rbc * dci) + Rbc.transpose() * skew_symmetric(Rbjbi * Rbc * dci);   // Lc_t
            Mat3d tmp = skew_symmetric(tcjci) * Rcjci;
            jaco_Lc_ex.block(0,3,3,3) = -Rcjci * skew_symmetric(nci) + skew_symmetric(Rcjci * nci)
                    -tmp * skew_symmetric(dci) + skew_symmetric(tmp * dci);    // Lc_theta
                    jaco_Lc_ex.block(3,3,3,3) = -Rcjci * skew_symmetric(dci) + skew_symmetric(Rcjci * dci);

                    jacobian_ex_pose.leftCols<6>() = jaco_e_Lc * jaco_Lc_ex;
                    jacobian_ex_pose.rightCols<1>().setZero();
        }
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jacobian_lineOrth(jacobians[3]);

            Mat3d Rbjbi = Rwbj.transpose() * Rwbi;
            Mat3d Rcjci = Rbc.transpose() * Rbjbi * Rbc;
            Vec3d tcjci = Rbc * ( Rwbj.transpose() * (Rwbi * tbc + twbi - twbj) - tbc);

            Eigen::Matrix<double,6,6> Tcjci;
            Tcjci << Rcjci, skew_symmetric(tcjci) * Rcjci,
            Mat3d::Zero(),  Rcjci;

            Vec3d nci = line_ci.head(3);
            Vec3d vci = line_ci.tail(3);
            Vec3d u1 = nci/nci.norm();
            Vec3d u2 = vci/vci.norm();
            Vec3d u3 = u1.cross(u2);
            Vec2d w( nci.norm(), vci.norm() );
            w = w/w.norm();

            Eigen::Matrix<double, 6, 4> jaco_Lc_orth;
            jaco_Lc_orth.setZero();
            jaco_Lc_orth.block(3,0,3,1) = w[1] * u3;
            jaco_Lc_orth.block(0,1,3,1) = -w[0] * u3;
            jaco_Lc_orth.block(0,2,3,1) = w(0) * u2;
            jaco_Lc_orth.block(3,2,3,1) = -w(1) * u1;
            jaco_Lc_orth.block(0,3,3,1) = -w(1) * u1;
            jaco_Lc_orth.block(3,3,3,1) = w(0) * u2;

            jacobian_lineOrth = jaco_e_Lc * Tcjci * jaco_Lc_orth;
        }

    }
    /*
        // check jacobian
        if(jacobians[0])
        std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jacobians[0]) << std::endl
                  << std::endl;
        if(jacobians[1])
        std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jacobians[1]) << std::endl
                  << std::endl;
        if(jacobians[2])
        std::cout << Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>>(jacobians[2]) << std::endl
                  << std::endl;
        if(jacobians[3])
        std::cout << Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>>(jacobians[3]) << std::endl
                  << std::endl;
        const double eps = 1e-6;
        Eigen::Matrix<double, 2, 22> num_jacobian;// 3 * 6 + 4
        for (int k = 0; k < 22; k++)
        {

            Vec3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
            Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

            Vec3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
            Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

            Vec3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
            Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

            Eigen::Vector4d line_orth( parameters[3][0],parameters[3][1],parameters[3][2],parameters[3][3]);
            ceres::LocalParameterization *local_parameterization_line = new LineOrthParameterization();

            int a = k / 3, b = k % 3;
            Vec3d delta = Vec3d(b == 0, b == 1, b == 2) * eps;

            if (a == 0)
                Pi += delta;
            else if (a == 1)
                Qi = Qi * deltaQ(delta);
            else if (a == 2)
                Pj += delta;
            else if (a == 3)
                Qj = Qj * deltaQ(delta);
            else if (a == 4)
                tic += delta;
            else if (a == 5)
                qic = qic * deltaQ(delta);
            else if (a == 6) {           // line orth的前三个元素
                Eigen::Vector4d line_new;
                Eigen::Vector4d delta_l;
                delta_l<< delta, 0.0;
                local_parameterization_line->Plus(line_orth.data(),delta_l.data(),line_new.data());
                line_orth = line_new;
            }
            else if (a == 7) {           // line orth的最后一个元素
                Eigen::Vector4d line_new;
                Eigen::Vector4d delta_l;
                delta_l.setZero();
                delta_l[3]= delta.x();
                local_parameterization_line->Plus(line_orth.data(),delta_l.data(),line_new.data());
                line_orth = line_new;
            }

            Vec6d line_ci = orth_to_plk(line_orth);
            Mat3d Rbc(qic);
            Vec3d tbc(tic);
            Vec6d line_bi = plk_to_pose(line_ci, Rbc, tbc);

            Mat3d Rwbi = Qi.toRotationMatrix();
            Vec3d twbi(Pi);
            Vec6d line_w = plk_to_pose(line_bi, Rwbi, twbi);

            Mat3d Rwbj = Qj.toRotationMatrix();
            Vec3d twbj(Pj);
            Vec6d line_bj = plk_from_pose(line_w, Rwbj, twbj);

            Vec6d line_cj = plk_from_pose(line_bj, Rbc, tbc);

            // 直线的投影矩阵K为单位阵
            Vec3d nc = line_cj.head(3);

            double l_norm = nc(0) * nc(0) + nc(1) * nc(1);
            double l_sqrtnorm = sqrt( l_norm );

            double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
            double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);
            Vec2d tmp_residual;
            tmp_residual(0) = e1/l_sqrtnorm;
            tmp_residual(1) = e2/l_sqrtnorm;
            tmp_residual = sqrt_info * tmp_residual;

            num_jacobian.col(k) = (tmp_residual - residual) / eps;

        }
        std::cout <<"num_jacobian:\n"<< num_jacobian <<"\n"<< std::endl;
    */

    return true;
}


Eigen::Matrix2d lineProjectionFactor_instartframe::sqrt_info;
lineProjectionFactor_instartframe::lineProjectionFactor_instartframe(const Eigen::Vector4d &_obs_i) : obs_i(_obs_i)
{
};

/*
  parameters[0]:  Twi
  parameters[1]:  Twj
  parameters[2]:  Tbc
  parameters[3]:  line_orth
*/
bool lineProjectionFactor_instartframe::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{

    Eigen::Vector4d line_orth( parameters[0][0],parameters[0][1],parameters[0][2],parameters[0][3]);
    Vec6d line_ci = orth_to_plk(line_orth);

    // 直线的投影矩阵K为单位阵
    Vec3d nc = line_ci.head(3);
    double l_norm = nc(0) * nc(0) + nc(1) * nc(1);
    double l_sqrtnorm = sqrt( l_norm );
    double l_trinorm = l_norm * l_sqrtnorm;

    double e1 = obs_i(0) * nc(0) + obs_i(1) * nc(1) + nc(2);
    double e2 = obs_i(2) * nc(0) + obs_i(3) * nc(1) + nc(2);
    Eigen::Map<Vec2d> residual(residuals);
    residual(0) = e1/l_sqrtnorm;
    residual(1) = e2/l_sqrtnorm;

    sqrt_info.setIdentity();
    residual = sqrt_info * residual;
    //std::cout<< residual <<std::endl;
    if (jacobians)
    {

        Eigen::Matrix<double, 2, 3> jaco_e_l(2, 3);
        jaco_e_l << (obs_i(0)/l_sqrtnorm - nc(0) * e1 / l_trinorm ), (obs_i(1)/l_sqrtnorm - nc(1) * e1 / l_trinorm ), 1.0/l_sqrtnorm,
        (obs_i(2)/l_sqrtnorm - nc(0) * e2 / l_trinorm ), (obs_i(3)/l_sqrtnorm - nc(1) * e2 / l_trinorm ), 1.0/l_sqrtnorm;

        jaco_e_l = sqrt_info * jaco_e_l;

        Eigen::Matrix<double, 3, 6> jaco_l_Lc(3, 6);
        jaco_l_Lc.setZero();
        jaco_l_Lc.block(0,0,3,3) = Mat3d::Identity();

        Eigen::Matrix<double, 2, 6> jaco_e_Lc;
        jaco_e_Lc = jaco_e_l * jaco_l_Lc;
        //std::cout <<jaco_e_Lc<<"\n\n";
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> jacobian_lineOrth(jacobians[0]);


            Vec3d nci = line_ci.head(3);
            Vec3d vci = line_ci.tail(3);
            Vec3d u1 = nci/nci.norm();
            Vec3d u2 = vci/vci.norm();
            Vec3d u3 = u1.cross(u2);
            Vec2d w( nci.norm(), vci.norm() );
            w = w/w.norm();

            Eigen::Matrix<double, 6, 4> jaco_Lci_orth;
            jaco_Lci_orth.setZero();
            jaco_Lci_orth.block(3,0,3,1) = w[1] * u3;
            jaco_Lci_orth.block(0,1,3,1) = -w[0] * u3;
            jaco_Lci_orth.block(0,2,3,1) = w(0) * u2;
            jaco_Lci_orth.block(3,2,3,1) = -w(1) * u1;
            jaco_Lci_orth.block(0,3,3,1) = -w(1) * u1;
            jaco_Lci_orth.block(3,3,3,1) = w(0) * u2;

            jacobian_lineOrth = jaco_e_Lc  * jaco_Lci_orth;
        }

    }

    return true;
}

}
