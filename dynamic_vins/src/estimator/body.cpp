/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "body.h"

namespace dynamic_vins{ \

BodyState body;

    void BodyState::SetOptimizeParameters() {
        for (int i = 0; i <= kWinSize; i++)
        {
            para_Pose[i][0] = Ps[i].x();
            para_Pose[i][1] = Ps[i].y();
            para_Pose[i][2] = Ps[i].z();
            Quaterniond q{Rs[i]};
            para_Pose[i][3] = q.x();
            para_Pose[i][4] = q.y();
            para_Pose[i][5] = q.z();
            para_Pose[i][6] = q.w();

            if(cfg::is_use_imu)
            {
                para_SpeedBias[i][0] = Vs[i].x();
                para_SpeedBias[i][1] = Vs[i].y();
                para_SpeedBias[i][2] = Vs[i].z();

                para_SpeedBias[i][3] = Bas[i].x();
                para_SpeedBias[i][4] = Bas[i].y();
                para_SpeedBias[i][5] = Bas[i].z();

                para_SpeedBias[i][6] = Bgs[i].x();
                para_SpeedBias[i][7] = Bgs[i].y();
                para_SpeedBias[i][8] = Bgs[i].z();
            }
        }

        for (int i = 0; i < cfg::kCamNum; i++)
        {
            para_ex_pose[i][0] = tic[i].x();
            para_ex_pose[i][1] = tic[i].y();
            para_ex_pose[i][2] = tic[i].z();
            Quaterniond q{ric[i]};
            para_ex_pose[i][3] = q.x();
            para_ex_pose[i][4] = q.y();
            para_ex_pose[i][5] = q.z();
            para_ex_pose[i][6] = q.w();
        }

        para_Td[0][0] = td;

    }

    void BodyState::GetOptimizationParameters(Vec3d &origin_R0,Vec3d &origin_P0) {

        if(cfg::is_use_imu){
            Vec3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                          para_Pose[0][3],
                                                          para_Pose[0][4],
                                                          para_Pose[0][5]).toRotationMatrix());
            double y_diff = origin_R0.x() - origin_R00.x();
            //TODO
            Mat3d rot_diff = Utility::ypr2R(Vec3d(y_diff, 0, 0));
            if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
            {
                Debugv("euler singular point!");
                rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                                    para_Pose[0][3],
                                                    para_Pose[0][4],
                                                    para_Pose[0][5]).toRotationMatrix().transpose();
            }

            for (int i = 0; i <= kWinSize; i++)
            {
                Rs[i] = rot_diff *
                        Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5])
                        .normalized().toRotationMatrix();

                Ps[i] = rot_diff * Vec3d(para_Pose[i][0] - para_Pose[0][0],
                                              para_Pose[i][1] - para_Pose[0][1],
                                              para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


                Vs[i] = rot_diff * Vec3d(para_SpeedBias[i][0],
                                              para_SpeedBias[i][1],
                                              para_SpeedBias[i][2]);

                Bas[i] = Vec3d(para_SpeedBias[i][3],
                                    para_SpeedBias[i][4],
                                    para_SpeedBias[i][5]);

                Bgs[i] = Vec3d(para_SpeedBias[i][6],
                                    para_SpeedBias[i][7],
                                    para_SpeedBias[i][8]);

            }
        }
        else
        {
            for (int i = 0; i <= kWinSize; i++)
            {
                Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5])
                        .normalized().toRotationMatrix();

                Ps[i] = Vec3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
            }
        }

        if(cfg::is_use_imu){
            for (int i = 0; i < cfg::kCamNum; i++){
                tic[i] = Vec3d(para_ex_pose[i][0],
                                    para_ex_pose[i][1],
                                    para_ex_pose[i][2]);
                ric[i] = Quaterniond(para_ex_pose[i][6],
                                          para_ex_pose[i][3],
                                          para_ex_pose[i][4],
                                          para_ex_pose[i][5]).normalized().toRotationMatrix();
            }

            td = para_Td[0][0];
        }
    }


}
