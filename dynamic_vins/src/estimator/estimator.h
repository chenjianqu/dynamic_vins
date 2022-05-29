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


#pragma once
 
#include <thread>
#include <mutex>

#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "utils/parameters.h"
#include "feature_manager.h"

#include "front_end/background_tracker.h"
#include "vio_util.h"
#include "instance_manager.h"
#include "utils/def.h"
#include "landmark.h"
#include "vio_parameters.h"
#include "feature_queue.h"

#include "factor/imu_factor.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"
#include "factor/marginalization_factor.h"

namespace dynamic_vins{\

using std::set;
using std::queue;

class Estimator
{
  public:
    using Ptr=std::shared_ptr<Estimator>;

    explicit Estimator(const string& config_path);
    ~Estimator();
    void SetParameter();

    void InputIMU(double t, const Vec3d &linearAcceleration, const Vec3d &angularVelocity);

    void ProcessMeasurements();
    void ChangeSensorType(int use_imu, int use_stereo);

    void GetPoseInWorldFrame(Eigen::Matrix4d &T);
    void GetPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void PredictPtsInNextFrame();

    void ClearState();

    void SetOutputPose(const Mat3d &R,const Vec3d &P,const Mat3d &R_bc,const Vec3d &P_bc){
        std::unique_lock<std::mutex> lk(out_pose_mutex);
        R_out=R;
        P_out=P;
        R_bc_out=R_bc;
        P_bc_out=P_bc;
    }
    std::tuple<Mat3d,Vec3d,Mat3d,Vec3d> GetOutputPose(){
        std::unique_lock<std::mutex> lk(out_pose_mutex);
        return {R_out,P_out,R_bc_out,P_bc_out};
    }

    Mat3d ric[2];
    Vec3d tic[2];
    Vec3d Ps[(kWinSize + 1)];
    Vec3d Vs[(kWinSize + 1)];
    Mat3d Rs[(kWinSize + 1)];
    Vec3d Bas[(kWinSize + 1)];
    Vec3d Bgs[(kWinSize + 1)];
    double td{};
    double headers[(kWinSize + 1)]{};
    int frame{};

    vector<Vec3d> key_poses;
    SolverFlag solver_flag;
    MarginFlag margin_flag;

    double para_ex_pose[2][kSizePose]{};
    double para_Pose[kWinSize + 1][kSizePose]{};
    double para_SpeedBias[kWinSize + 1][kSizeSpeedBias]{};
    double para_Feature[kNumFeat][kSizeFeature]{};
    double para_Retrive_Pose[kSizePose]{};
    double para_Td[1][1]{};
    double para_Tr[1][1]{};

    InstanceManager insts_manager;
    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    SemanticFeature feature_frame;

private:

    void SetMarginalizationInfo();

    void ProcessImage(SemanticFeature & image, double header);
    void ProcessIMU(double t, double dt, const Vec3d &linear_acceleration, const Vec3d &angular_velocity);

    void SlideWindow();
    void SlideWindowNew();
    void SlideWindowOld();
    void Optimization();
    void Vector2double();
    void Double2vector();

    bool InitialStructure();
    bool VisualInitialAlign();
    bool RelativePose(Mat3d &relative_R, Vec3d &relative_T, int &l);

    bool FailureDetection();
    bool GetIMUInterval(double t0, double t1, vector<pair<double, Vec3d>> &acc_vec,
                        vector<pair<double, Vec3d>> &gyr_vec);

    void OutliersRejection(set<int> &removeIndex);
    static double ReprojectionError(Mat3d &Ri, Vec3d &Pi, Mat3d &rici, Vec3d &tici,
                                    Mat3d &Rj, Vec3d &Pj, Mat3d &ricj, Vec3d &ticj,
                                    double depth, Vec3d &uvi, Vec3d &uvj);
    void UpdateLatestStates();
    void FastPredictIMU(double t, const Vec3d& linear_acceleration,const Vec3d& angular_velocity);

    void InitFirstIMUPose(vector<pair<double, Vec3d>> &accVector);

    void AddInstanceParameterBlock(ceres::Problem &problem);
    int AddResidualBlock(ceres::Problem &problem, ceres::LossFunction *loss_function);


    bool IMUAvailable(double t){
        if(!acc_buf.empty() && t <= acc_buf.back().first)
            return true;
        else
            return false;
    }

    string LogCurrentPose(){
        string result;
        for(int i=0; i <= kWinSize; ++i)
            result+= fmt::format("{} t:({}) q:({})\n", i, VecToStr(Ps[i]),
                                 QuaternionToStr(Eigen::Quaterniond(Rs[i])));
        return result;
    }

    void initFirstPose(Vec3d p, Mat3d r){
        Ps[0] = p;
        Rs[0] = r;
        initP = p;
        initR = r;
    }


    std::mutex process_mutex;
    std::mutex buf_mutex;
    std::mutex propogate_mutex;
    queue<pair<double, Vec3d>> acc_buf;
    queue<pair<double, Vec3d>> gyr_buf;
    double prev_time{}, cur_time{};
    bool openExEstimation{};

    Vec3d g;

    Mat3d back_R0, last_R, last_R0;
    Vec3d back_P0, last_P, last_P0;

    IntegrationBase *pre_integrations[(kWinSize + 1)] {nullptr};
    Vec3d acc_0, gyr_0;

    vector<double> dt_buf[(kWinSize + 1)];
    vector<Vec3d> linear_acceleration_buf[(kWinSize + 1)];
    vector<Vec3d> angular_velocity_buf[(kWinSize + 1)];

    int sum_of_outlier{}, sum_of_back{}, sum_of_front{}, sum_of_invalid{};
    int input_image_cnt{};

    bool first_imu{};
    bool is_valid{}, is_key{};
    bool failure_occur{};

    vector<Vec3d> point_cloud;
    vector<Vec3d> margin_cloud;
    double initial_timestamp{};

    int loop_window_index{};

    MarginalizationInfo *last_marg_info{};
    vector<double *> last_marg_para_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration{};

    Vec3d initP;
    Mat3d initR;

    double latest_time{};
    Vec3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;

    bool is_init_first_pose{};
    bool initThreadFlag;

    std::mutex out_pose_mutex;
    Mat3d R_out,R_bc_out;
    Vec3d P_out,P_bc_out;

};


}
