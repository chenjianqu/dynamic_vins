/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
 
#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "../parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"

#include "dynamic.h"
#include "instance_manager.h"
#include "../utils.h"

class Estimator
{
  public:
    using Ptr=std::shared_ptr<Estimator>;

    Estimator();
    ~Estimator();
    void SetParameter();

    // interface
    void initFirstPose(Vec3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header);

    void PushBack(double time, FeatureMap &feats);
    void PushBack(double time, FeatureMap &feats, InstancesFeatureMap &insts);

    void ProcessMeasurements();
    void ChangeSensorType(int use_imu, int use_stereo);

    // internal
    void ClearState();
    bool initialStructure();
    bool visualInitialAlign();
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);
    void slideWindow();
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection();
    bool getIMUInterval(double t0, double t1, vector<pair<double, Vec3d>> &accVector, 
                                              vector<pair<double, Vec3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates();
    void fastPredictIMU(double t, Vec3d linear_acceleration, Vec3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Vec3d>> &accVector);

    std::mutex mProcess;
    std::mutex mBuf;
    std::mutex mPropagate;
    queue<pair<double, Vec3d>> accBuf;
    queue<pair<double, Vec3d>> gyrBuf;
    queue<FeatureFrame> featureBuf;
    queue<InstancesFeatureMap> instancesBuf;
    double prevTime{}, curTime{};
    bool openExEstimation{};

    std::thread trackThread;
    std::thread processThread;

    SolverFlag solver_flag;
    MarginFlag   margin_flag;
    Vector3d g;

    Matrix3d ric[2];
    Vector3d tic[2];

    Vector3d        Ps[(kWindowSize + 1)];
    Vector3d        Vs[(kWindowSize + 1)];
    Matrix3d        Rs[(kWindowSize + 1)];
    Vector3d        Bas[(kWindowSize + 1)];
    Vector3d        Bgs[(kWindowSize + 1)];
    double td{};

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double headers[(kWindowSize + 1)]{};

    IntegrationBase *pre_integrations[(kWindowSize + 1)] {nullptr};
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(kWindowSize + 1)];
    vector<Vector3d> linear_acceleration_buf[(kWindowSize + 1)];
    vector<Vector3d> angular_velocity_buf[(kWindowSize + 1)];

    int frame_count{};
    int sum_of_outlier{}, sum_of_back{}, sum_of_front{}, sum_of_invalid{};
    int inputImageCnt{};

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu{};
    bool is_valid{}, is_key{};
    bool failure_occur{};

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp{};

    double para_Pose[kWindowSize + 1][kSizePose]{};
    double para_SpeedBias[kWindowSize + 1][kSizeSpeedBias]{};
    double para_Feature[kNumFeat][kSizeFeature]{};
    double para_ex_pose[2][kSizePose]{};
    double para_Retrive_Pose[kSizePose]{};
    double para_Td[1][1]{};
    double para_Tr[1][1]{};

    int loop_window_index{};

    MarginalizationInfo *last_marginalization_info{};
    vector<double *> last_marginalization_parameter_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration{};

    Vec3d initP;
    Eigen::Matrix3d initR;

    double latest_time{};
    Vec3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;

    bool initFirstPoseFlag{};
    bool initThreadFlag;

    InstanceManager insts_manager;

private:
    string logCurrentPose();


};
