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

#include <sys/types.h>
#include <dirent.h>
#include <cstdio>

#include "estimator.h"
#include "utility/visualization.h"
#include "utils.h"

namespace dynamic_vins{\


Estimator::Estimator(): f_manager{Rs}
{
    ClearState();
    insts_manager.set_estimator(this);

    /*    std::string path_test="/home/chen/slam/expriments/DynamicVINS/FlowTrack/1_vel";
        DIR* dir= opendir(path_test.c_str());//打开当前目录
        if(access(path_test.c_str(),F_OK)==-1) //文件夹不存在
            return;
        struct dirent *ptr;
        while((ptr=readdir(dir)) != nullptr){
            saved_name.insert(ptr->d_name);
        }*/
}



Estimator::~Estimator()
{
    printf("join thread \n");
}



void Estimator::PushBack(double time, FeatureMap &feats, InstancesFeatureMap &insts)
{
    inputImageCnt++;
    if(inputImageCnt % 2 == 0){
        mBuf.lock();
        featureBuf.emplace(std::move(feats),time);//放入特征队列中
        instancesBuf.push(std::move(insts));
        mBuf.unlock();
    }
}


void Estimator::PushBack(double time, FeatureMap &feats)
{
    inputImageCnt++;
    if(inputImageCnt % 2 == 0){
        mBuf.lock();
        featureBuf.emplace(std::move(feats),time);//放入特征队列中
        mBuf.unlock();
    }
}


string Estimator::logCurrentPose(){
    string result;
    for(int i=0; i <= kWindowSize; ++i)
        result+= fmt::format("{} t:({}) q:({})\n", i, VecToStr(Ps[i]), QuaternionToStr(Eigen::Quaterniond(Rs[i])));
    return result;
}


void Estimator::optimization()
{
    TicToc t_whole, t_prepare;
    vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    //loss_function = NULL;
    //loss_function = new ceres::CauchyLoss(1.0 / kFocalLength);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    for (int i = 0; i < frame_count + 1; i++){
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_Pose[i], kSizePose, local_parameterization);
        if(cfg::is_use_imu)
            problem.AddParameterBlock(para_SpeedBias[i], kSizeSpeedBias);
    }
    if(!cfg::is_use_imu)
        problem.SetParameterBlockConstant(para_Pose[0]);

    for (int i = 0; i < cfg::kCamNum; i++)
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(para_ex_pose[i], kSizePose, local_parameterization);
        if ((cfg::ESTIMATE_EXTRINSIC && frame_count == kWindowSize && Vs[0].norm() > 0.2) || openExEstimation)
        {
            //ROS_INFO("estimate extinsic param");
            openExEstimation = true;
        }
        else
        {
            //ROS_INFO("fix extinsic param");
            problem.SetParameterBlockConstant(para_ex_pose[i]);
        }
    }

    if(cfg::slam == SlamType::kDynamic){
        insts_manager.SetOptimizationParameters();
        insts_manager.AddInstanceParameterBlock(problem);
    }

    problem.AddParameterBlock(para_Td[0], 1);

    if (!cfg::is_estimate_td || Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(para_Td[0]);

    if (last_marginalization_info && last_marginalization_info->valid)
    {
        // construct new marginlization_factor
        auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, nullptr,
                                 last_marginalization_parameter_blocks);
    }
    if(cfg::is_use_imu)
    {
        for (int i = 0; i < frame_count; i++)
        {
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            auto* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, nullptr, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
        }
    }

    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                auto *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_ex_pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if(cfg::is_stereo && it_per_frame.is_stereo)
            {
                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    auto *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_ex_pose[0], para_ex_pose[1], para_Feature[feature_index], para_Td[0]);
                }
                else
                {
                    auto *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                 it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                    problem.AddResidualBlock(f, loss_function, para_ex_pose[0], para_ex_pose[1], para_Feature[feature_index], para_Td[0]);
                }

            }
            f_m_cnt++;
        }
    }

    if(cfg::slam == SlamType::kDynamic)
        insts_manager.AddResidualBlock(problem, loss_function);

    DebugV("optimization 开始优化 visual measurement count: {}", f_m_cnt);

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = cfg::KNumIter;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (margin_flag == MarginFlag::kMarginOld)
        options.max_solver_time_in_seconds = cfg::kMaxSolverTime * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = cfg::kMaxSolverTime;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    DebugV("Iterations: {}", summary.iterations.size());
    InfoV("优化完成");

    if(cfg::slam == SlamType::kDynamic)
        insts_manager.GetOptimizationParameters();


    string msg="相机位姿 优化前：\n";
    msg += logCurrentPose();

    double2vector();

    msg+="相机位姿 优化后：\n";
    msg += logCurrentPose();
    DebugV(msg);

    if(frame_count < kWindowSize)
        return;

    TicToc t_whole_marginalization;
    if (margin_flag == MarginFlag::kMarginOld)
    {
        auto *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info && last_marginalization_info->valid)
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                last_marginalization_parameter_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
            auto *residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                              last_marginalization_parameter_blocks,
                                                              drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if(cfg::is_use_imu)
        {
            if (pre_integrations[1]->sum_dt < 10.0)
            {
                auto* imu_factor = new IMUFactor(pre_integrations[1]);
                auto *residual_block_info = new ResidualBlockInfo(imu_factor, nullptr,
                                                                  vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                  vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual_block_info);
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature)
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size();
                if (it_per_id.used_num < 4)
                    continue;

                ++feature_index;

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
                if (imu_i != 0)
                    continue;

                Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Vector3d pts_j = it_per_frame.point;
                        auto *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                        it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        auto *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                          vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_ex_pose[0], para_Feature[feature_index], para_Td[0]},
                                                                          vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if(cfg::is_stereo && it_per_frame.is_stereo)
                    {
                        Vector3d pts_j_right = it_per_frame.pointRight;
                        if(imu_i != imu_j)
                        {
                            auto *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                         it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            auto *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                              vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_ex_pose[0], para_ex_pose[1], para_Feature[feature_index], para_Td[0]},
                                                                              vector<int>{0, 4});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                        else
                        {
                            auto *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocityRight,
                                                                         it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                            auto *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                              vector<double *>{para_ex_pose[0], para_ex_pose[1], para_Feature[feature_index], para_Td[0]},
                                                                              vector<int>{2});
                            marginalization_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        DebugV("pre marginalization {} ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        DebugV("marginalization {} ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= kWindowSize; i++)
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
            if(cfg::is_use_imu)
                addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < cfg::kCamNum; i++)
            addr_shift[reinterpret_cast<long>(para_ex_pose[i])] = para_ex_pose[i];

        addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];

        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;

    }
    else
    {
        if (last_marginalization_info &&
        std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[kWindowSize - 1]))
        {

            auto *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[kWindowSize - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[kWindowSize - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                auto *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                  last_marginalization_parameter_blocks,
                                                                  drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            InfoV("begin marginalization");
            marginalization_info->preMarginalize();
            InfoV("end pre marginalization, {} ms", t_pre_margin.toc());

            TicToc t_margin;
            InfoV("begin marginalization");
            marginalization_info->marginalize();
            InfoV("end marginalization, {} ms", t_margin.toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= kWindowSize; i++)
            {
                if (i == kWindowSize - 1)
                    continue;
                else if (i == kWindowSize)
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];
                    if(cfg::is_use_imu)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];
                    if(cfg::is_use_imu)
                        addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < cfg::kCamNum; i++)
                addr_shift[reinterpret_cast<long>(para_ex_pose[i])] = para_ex_pose[i];

            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];


            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

            delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;

        }
    }
    //printf("whole marginalization costs: %f \n", t_whole_marginalization.toc());
    //printf("whole time for ceres: %f \n", t_whole.toc());
}






void Estimator::ClearState()
{
    mProcess.lock();
    while(!accBuf.empty())
        accBuf.pop();
    while(!gyrBuf.empty())
        gyrBuf.pop();
    while(!featureBuf.empty())
        featureBuf.pop();


    prevTime = -1;
    curTime = 0;
    openExEstimation = false;
    initP = Vec3d(0, 0, 0);
    initR = Eigen::Matrix3d::Identity();
    inputImageCnt = 0;
    initFirstPoseFlag = false;

    for (int i = 0; i < kWindowSize + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];

        pre_integrations[i] = nullptr;

    }

    for (int i = 0; i < cfg::kCamNum; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = SolverFlag::kInitial;
    initial_timestamp = 0;
    all_image_frame.clear();

    delete tmp_pre_integration;
    delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = false;

    mProcess.unlock();
}

void Estimator::SetParameter()
{
    mProcess.lock();
    for (int i = 0; i < cfg::kCamNum; i++){
        tic[i] = cfg::TIC[i];
        ric[i] = cfg::RIC[i];
        std::stringstream ss;
        ss << "setParameter extrinsic cam " << i << endl  << ric[i] << endl << tic[i].transpose() << endl;
        InfoV(ss.str());
    }
    f_manager.setRic(ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();

    ProjectionSpeedFactor::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();
    ProjectionSpeedSimpleFactor::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();

    ProjInst21Factor::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();
    ProjInst21SimpleFactor::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();
    ProjInst22Factor::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();
    ProjInst12Factor::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();
    ProjInst12FactorSimple::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();

    //ProjectionSpeedFactor::sqrt_info
    td = cfg::TD;
    g = cfg::G;
    std::stringstream ss;
    ss << "setParameter set g " << g.transpose() << endl;
    InfoV(ss.str());

    mProcess.unlock();
}

void Estimator::ChangeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    mProcess.lock();
    if(!use_imu && !use_stereo)
        printf("at least use two sensors! \n");
    else
    {
        if(cfg::is_use_imu != use_imu)
        {
            cfg::is_use_imu = use_imu;
            if(cfg::is_use_imu)
            {
                // reuse imu; restart system
                restart = true;
            }
            else
            {
                if (last_marginalization_info != nullptr)
                    delete last_marginalization_info;

                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }

        cfg::is_stereo = use_stereo;
        printf("use imu %d use stereo %d\n", cfg::is_use_imu, cfg::is_stereo);
    }
    mProcess.unlock();
    if(restart)
    {
        ClearState();
        SetParameter();
    }
}


void Estimator::InputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity)
{
    mBuf.lock();
    accBuf.push(make_pair(t, linearAcceleration));
    gyrBuf.push(make_pair(t, angularVelocity));
    //printf("input imu with time %f \n", t);
    mBuf.unlock();

    if (solver_flag == SolverFlag::kNonLinear)
    {
        mPropagate.lock();
        fastPredictIMU(t, linearAcceleration, angularVelocity);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        mPropagate.unlock();
    }
}



bool Estimator::getIMUInterval(double t0, double t1, vector<pair<double, Vec3d>> &accVector,
                               vector<pair<double, Vec3d>> &gyrVector){
    if(accBuf.empty())
    {
        printf("not receive imu\n");
        return false;
    }
    //printf("get imu from %f %f\n", t0, t1);
    //printf("imu fornt time %f   imu end time %f\n", accBuf.front().first, accBuf.back().first);
    if(t1 <= accBuf.back().first)
    {
        while (accBuf.front().first <= t0)
        {
            accBuf.pop();
            gyrBuf.pop();
        }
        while (accBuf.front().first < t1)
        {
            accVector.push_back(accBuf.front());
            accBuf.pop();
            gyrVector.push_back(gyrBuf.front());
            gyrBuf.pop();
        }
        accVector.push_back(accBuf.front());
        gyrVector.push_back(gyrBuf.front());
    }
    else
    {
        printf("wait for imu\n");
        return false;
    }
    return true;
}

bool Estimator::IMUAvailable(double t)
{
    if(!accBuf.empty() && t <= accBuf.back().first)
        return true;
    else
        return false;
}


void Estimator::initFirstIMUPose(vector<pair<double, Vec3d>> &accVector)
{
    printf("init first imu pose\n");
    initFirstPoseFlag = true;
    //return;
    Vec3d averAcc(0, 0, 0);
    int n = (int)accVector.size();
    for(size_t i = 0; i < accVector.size(); i++)
    {
        averAcc = averAcc + accVector[i].second;
    }
    averAcc = averAcc / n;
    printf("averge acc %f %f %f\n", averAcc.x(), averAcc.y(), averAcc.z());
    Matrix3d R0 = Utility::g2R(averAcc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Vec3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vector3d(5, 0, 0);
}

void Estimator::initFirstPose(Vec3d p, Eigen::Matrix3d r)
{
    Ps[0] = p;
    Rs[0] = r;
    initP = p;
    initR = r;
}


void Estimator::processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu)
    {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count])
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0)
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

bool Estimator::initialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vector3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25)
        {
            ROS_INFO("IMU excitation not enouth!");
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame){
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.emplace_back(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()});
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l))
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame_count + 1, Q, T, l,
                      relative_R, relative_T,
                      sfm_f, sfm_tracked_points))
    {
        ROS_DEBUG("global SFM failed!");
        margin_flag = MarginFlag::kMarginOld;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vector3d>::iterator it;
    frame_it = all_image_frame.begin( );
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == headers[i])
        {
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * cfg::RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > headers[i])
        {
            i++;
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_inital = - R_inital * T[i];
        cv::eigen2cv(R_inital, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_inital, t);

        frame_it->second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points)
        {
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second)
            {
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end())
                {
                    Vector3d world_pts = it->second;
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2));
                    pts_3_vector.push_back(pts_3);
                    Vector2d img_pts = i_p.second.head<2>();
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2);
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if(pts_3_vector.size() < 6)
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r);
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        T_pnp = R_pnp * (-T_pnp);
        frame_it->second.R = R_pnp * cfg::RIC[0].transpose();
        frame_it->second.T = T_pnp;
    }
    if (visualInitialAlign())
        return true;
    else
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result)
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++)
    {
        Matrix3d Ri = all_image_frame[headers[i]].R;
        Vector3d Pi = all_image_frame[headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= kWindowSize; i++)
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * cfg::TIC[0] - (s * Ps[0] - Rs[0] * cfg::TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Vec3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose());

    f_manager.clearDepth();
    f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < kWindowSize; i++)
    {
        vector<pair<Vector3d, Vector3d>> corres;
        corres = f_manager.getCorresponding(i, kWindowSize);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i;
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::vector2double()
{
    for (int i = 0; i <= kWindowSize; i++)
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


    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    /*    printf("相机位姿优化前:");
        for (int i = 0; i <= kWindowSize; i++)
            printf("%d:(%.3lf,%.3lf,%.3lf) ",i,Ps[i].x(),Ps[i].y(),Ps[i].z());
        printf("\n");*/

    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = false;
    }

    if(cfg::is_use_imu)
    {
        Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                         para_Pose[0][3],
                                                         para_Pose[0][4],
                                                         para_Pose[0][5]).toRotationMatrix());
        double y_diff = origin_R0.x() - origin_R00.x();
        //TODO
        Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
        if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
        {
            ROS_DEBUG("euler singular point!");
            rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                           para_Pose[0][3],
                                           para_Pose[0][4],
                                           para_Pose[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i <= kWindowSize; i++)
        {
            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                        para_Pose[i][1] - para_Pose[0][1],
                                        para_Pose[i][2] - para_Pose[0][2]) + origin_P0;


            Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                        para_SpeedBias[i][1],
                                        para_SpeedBias[i][2]);

            Bas[i] = Vector3d(para_SpeedBias[i][3],
                              para_SpeedBias[i][4],
                              para_SpeedBias[i][5]);

            Bgs[i] = Vector3d(para_SpeedBias[i][6],
                              para_SpeedBias[i][7],
                              para_SpeedBias[i][8]);

        }
    }
    else
    {
        for (int i = 0; i <= kWindowSize; i++)
        {
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = Vector3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(cfg::is_use_imu)
    {
        for (int i = 0; i < cfg::kCamNum; i++)
        {
            tic[i] = Vector3d(para_ex_pose[i][0],
                              para_ex_pose[i][1],
                              para_ex_pose[i][2]);
            ric[i] = Quaterniond(para_ex_pose[i][6],
                                 para_ex_pose[i][3],
                                 para_ex_pose[i][4],
                                 para_ex_pose[i][5]).normalized().toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);

    if(cfg::is_use_imu)
        td = para_Td[0][0];


    /*    printf("相机位姿优化后:");
        for (int i = 0; i <= kWindowSize; i++){
            printf("%d:",i);
            printf("(%.3lf",Ps[i].x());
            printf(",%.3lf,",Ps[i].y());
            printf("%.3lf) ",Ps[i].z());
            //        printf("%d:(%.3lf,%.3lf,%.3lf) ",i,Ps[i].x(),Ps[i].y(),Ps[i].z());
        }
        printf("\n");*/

}

bool Estimator::failureDetection()
{
    return false;
    if (f_manager.last_track_num < 2)
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[kWindowSize].norm() > 2.5)
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[kWindowSize].norm());
        return true;
    }
    if (Bgs[kWindowSize].norm() > 1.0)
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[kWindowSize].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[kWindowSize];
    if ((tmp_P - last_P).norm() > 5)
    {
        //ROS_INFO(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1)
    {
        //ROS_INFO(" big z translation");
        //return true;
    }
    Matrix3d tmp_R = Rs[kWindowSize];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50)
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false;
}

void Estimator::slideWindow()
{
    TicToc t_margin;
    if (margin_flag == MarginFlag::kMarginOld)
    {
        double t_0 = headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == kWindowSize)
        {
            for (int i = 0; i < kWindowSize; i++)
            {
                headers[i] = headers[i + 1];
                Rs[i].swap(Rs[i + 1]);
                Ps[i].swap(Ps[i + 1]);
                if(cfg::is_use_imu)
                {
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);

                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    Vs[i].swap(Vs[i + 1]);
                    Bas[i].swap(Bas[i + 1]);
                    Bgs[i].swap(Bgs[i + 1]);
                }
            }
            headers[kWindowSize] = headers[kWindowSize - 1];
            Ps[kWindowSize] = Ps[kWindowSize - 1];
            Rs[kWindowSize] = Rs[kWindowSize - 1];

            if(cfg::is_use_imu)
            {
                Vs[kWindowSize] = Vs[kWindowSize - 1];
                Bas[kWindowSize] = Bas[kWindowSize - 1];
                Bgs[kWindowSize] = Bgs[kWindowSize - 1];

                delete pre_integrations[kWindowSize];
                pre_integrations[kWindowSize] = new IntegrationBase{acc_0, gyr_0, Bas[kWindowSize], Bgs[kWindowSize]};

                dt_buf[kWindowSize].clear();
                linear_acceleration_buf[kWindowSize].clear();
                angular_velocity_buf[kWindowSize].clear();
            }

            if (true || solver_flag == SolverFlag::kInitial)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            slideWindowOld();
        }
    }
    else
    {
        if (frame_count == kWindowSize)
        {
            headers[frame_count - 1] = headers[frame_count];
            Ps[frame_count - 1] = Ps[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];

            if(cfg::is_use_imu)
            {
                for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
                {
                    double tmp_dt = dt_buf[frame_count][i];
                    Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
                    Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

                    pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame_count - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame_count - 1] = Vs[frame_count];
                Bas[frame_count - 1] = Bas[frame_count];
                Bgs[frame_count - 1] = Bgs[frame_count];

                delete pre_integrations[kWindowSize];
                pre_integrations[kWindowSize] = new IntegrationBase{acc_0, gyr_0, Bas[kWindowSize], Bgs[kWindowSize]};

                dt_buf[kWindowSize].clear();
                linear_acceleration_buf[kWindowSize].clear();
                angular_velocity_buf[kWindowSize].clear();
            }
            slideWindowNew();
        }
    }
}

void Estimator::slideWindowNew()
{
    sum_of_front++;
    f_manager.removeFront(frame_count);
}

void Estimator::slideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == SolverFlag::kNonLinear ? true : false;
    if (shift_depth)
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.removeBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.removeBack();
}


void Estimator::getPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame_count];
    T.block<3, 1>(0, 3) = Ps[frame_count];
}

void Estimator::getPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::predictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame_count < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    getPoseInWorldFrame(curT);
    getPoseInWorldFrame(frame_count - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Vec3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame_count)
            {
                double depth = it_per_id.estimated_depth;
                Vector3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vector3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vector3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vector3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    //featureTracker->setPrediction(predictPts);
    //printf("e output %d predict pts\n",(int)predictPts.size());
}

double Estimator::reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                    Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj,
                                    double depth, Vector3d &uvi, Vector3d &uvj){
    Vector3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vector3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::outliersRejection(set<int> &removeIndex)
{
    //return;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature)
    {
        double err = 0;
        int errCnt = 0;
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (it_per_id.used_num < 4)
            continue;
        feature_index ++;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
        Vector3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vector3d pts_j = it_per_frame.point;
                double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                     Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                     depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", kFocalLength / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(cfg::is_stereo && it_per_frame.is_stereo)
            {

                Vector3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", kFocalLength / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = reprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", kFocalLength / 1.5 * tmp_error);
                }
            }
        }
        double ave_err = err / errCnt;
        if(ave_err * kFocalLength > 3)
            removeIndex.insert(it_per_id.feature_id);

    }
}

void Estimator::fastPredictIMU(double t, Vec3d linear_acceleration, Vec3d angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Vec3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - g;
    Vec3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);
    Vec3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - g;
    Vec3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;
    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}


void Estimator::updateLatestStates(){
    mPropagate.lock();
    latest_time = headers[frame_count] + td;
    latest_P = Ps[frame_count];
    latest_Q = Rs[frame_count];
    latest_V = Vs[frame_count];
    latest_Ba = Bas[frame_count];
    latest_Bg = Bgs[frame_count];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    mBuf.lock();
    queue<pair<double, Vec3d>> tmp_accBuf = accBuf;
    queue<pair<double, Vec3d>> tmp_gyrBuf = gyrBuf;
    mBuf.unlock();
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Vec3d acc = tmp_accBuf.front().second;
        Vec3d gyr = tmp_gyrBuf.front().second;
        fastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    mPropagate.unlock();
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header){
    InfoV("processImage Adding feature points:{}", image.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, td))
        margin_flag = MarginFlag::kMarginOld;
    else
        margin_flag = MarginFlag::kMarginSecondNew;

    DebugV("processImage margin_flag:{}", margin_flag == MarginFlag::kMarginSecondNew ? "Non-keyframe" : "Keyframe");
    DebugV("processImage frame_count {}", frame_count);
    DebugV("processImage all feature size: {}", f_manager.getFeatureCount());

    headers[frame_count] = header;

    ImageFrame img_frame(image, header);
    img_frame.pre_integration = tmp_pre_integration;
    all_image_frame.insert({header, img_frame});
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if(cfg::ESTIMATE_EXTRINSIC == 2){
        InfoV("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0){
            auto cor = f_manager.getCorresponding(frame_count - 1, frame_count);
            if (Matrix3d calib_ric;initial_ex_rotation.CalibrationExRotation(cor, pre_integrations[frame_count]->delta_q, calib_ric)){
                DebugV("initial extrinsic rotation calib success");
                DebugV("initial extrinsic rotation:\n{}", EigenToStr(calib_ric));
                ric[0] = calib_ric;
                cfg::RIC[0] = calib_ric;
                cfg::ESTIMATE_EXTRINSIC = 1;
            }
        }
    }


    if(cfg::slam == SlamType::kDynamic){
        insts_manager.PredictCurrentPose();

        insts_manager.Triangulate(frame_count);
        insts_manager.InitialInstance();
    }



    if (solver_flag == SolverFlag::kInitial)
    {
        // monocular + IMU initilization
        if (!cfg::is_stereo && cfg::is_use_imu){
            if (frame_count == kWindowSize){
                bool result = false;
                if(cfg::ESTIMATE_EXTRINSIC != 2 && (header - initial_timestamp) > 0.1){
                    result = initialStructure();
                    initial_timestamp = header;
                }
                if(result){
                    optimization();
                    updateLatestStates();
                    solver_flag = SolverFlag::kNonLinear;
                    slideWindow();
                    InfoV("Initialization finish!");
                }
                else{
                    slideWindow();
                }
            }
        }
        // stereo + IMU initilization
        else if(cfg::is_stereo && cfg::is_use_imu)
        {
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            if (frame_count == kWindowSize){
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++){
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int j = 0; j <= kWindowSize; j++)
                    pre_integrations[j]->repropagate(Vector3d::Zero(), Bgs[j]);
                optimization();
                updateLatestStates();
                solver_flag = SolverFlag::kNonLinear;
                slideWindow();
                InfoV("Initialization finish!");
            }
        }

        // stereo only initilization
        else if(cfg::is_stereo && !cfg::is_use_imu){
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);
            f_manager.triangulate(frame_count, Ps, Rs, tic, ric);
            optimization();
            if(frame_count == kWindowSize)
            {
                optimization();
                updateLatestStates();
                solver_flag = SolverFlag::kNonLinear;
                slideWindow();
                InfoV("Initialization finish!");
            }
        }

        if(frame_count < kWindowSize){
            frame_count++;
            int prev_frame = frame_count - 1;
            Ps[frame_count] = Ps[prev_frame];
            Vs[frame_count] = Vs[prev_frame];
            Rs[frame_count] = Rs[prev_frame];
            Bas[frame_count] = Bas[prev_frame];
            Bgs[frame_count] = Bgs[prev_frame];
        }

    }
    else
    {
        TicToc t_solve;

        if(!cfg::is_use_imu)
            f_manager.initFramePoseByPnP(frame_count, Ps, Rs, tic, ric);

        f_manager.triangulate(frame_count, Ps, Rs, tic, ric);

        optimization();

        set<int> removeIndex;
        outliersRejection(removeIndex);
        f_manager.removeOutlier(removeIndex);

        DebugV("solver costs:{} ms", t_solve.toc());

        if (failureDetection())
        {
            WarnV("failure detection!");
            failure_occur = true;
            ClearState();
            SetParameter();
            WarnV("system reboot!");
            return;
        }

        if(cfg::slam == SlamType::kDynamic){
            //insts_manager.SetWindowPose();
            insts_manager.SlideWindow();
        }

        slideWindow();

        //printf("外点剔除\n");

        if(cfg::slam == SlamType::kDynamic)
            insts_manager.OutliersRejection();

        f_manager.removeFailures();

        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= kWindowSize; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[kWindowSize];
        last_P = Ps[kWindowSize];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        updateLatestStates();
    }


    if(cfg::slam == SlamType::kDynamic){
        insts_manager.SetInstanceCurrentPoint3d();
    }

}

void Estimator::ProcessMeasurements(){

    while (cfg::ok.load(std::memory_order_seq_cst))
    {
        if(!featureBuf.empty())
        {
            FeatureFrame feature_frame;
            InstancesFeatureMap curr_insts;
            vector<pair<double, Vec3d>> accVector, gyrVector;

            feature_frame = featureBuf.front();
            curTime = feature_frame.time + td;

            if(cfg::is_use_imu && !IMUAvailable(curTime)){
                std::cerr<<"wait for imu ..."<<endl;
                std::this_thread::sleep_for(5ms);
                continue;
            }

            if(cfg::slam == SlamType::kDynamic)
                curr_insts=instancesBuf.front();

            WarnV("----------Time : {} ----------", feature_frame.time);

            static TicToc tt;
            tt.tic();

            mBuf.lock();
            if(cfg::is_use_imu)
                getIMUInterval(prevTime, curTime, accVector, gyrVector);

            featureBuf.pop();
            if(cfg::slam == SlamType::kDynamic)
                instancesBuf.pop();

            mBuf.unlock();

            if(cfg::is_use_imu)
            {
                if(!initFirstPoseFlag)
                    initFirstIMUPose(accVector);
                for(size_t i = 0; i < accVector.size(); i++)
                {
                    double dt;
                    if(i == 0)
                        dt = accVector[i].first - prevTime;
                    else if (i == accVector.size() - 1)
                        dt = curTime - accVector[i - 1].first;
                    else
                        dt = accVector[i].first - accVector[i - 1].first;
                    processIMU(accVector[i].first, dt, accVector[i].second, gyrVector[i].second);
                }
            }
            mProcess.lock();

            InfoV("get input time: {} ms", tt.toc_then_tic());
            DebugV("solver_flag:{}", solver_flag == SolverFlag::kInitial ? "INITIAL" : "NO-LINEAR");

            if(cfg::slam == SlamType::kDynamic)
                insts_manager.PushBack(frame_count, curr_insts);

            processImage(feature_frame.features, feature_frame.time);
            prevTime = curTime;

            printStatistics(*this, 0);

            std_msgs::Header header;
            header.frame_id = "world";
            header.stamp = ros::Time(feature_frame.time);


            if(cfg::slam == SlamType::kDynamic){
                //printInstanceData(*this);
                pubInstancePointCloud(*this,header);
            }


            pubOdometry(*this, header);
            pubKeyPoses(*this, header);
            pubCameraPose(*this, header);
            pubPointCloud(*this, header);
            pubKeyframe(*this);
            pubTF(*this, header);
            mProcess.unlock();

            static unsigned int estimator_cnt=0;
            estimator_cnt++;
            auto output_msg=fmt::format("cnt:{} ___________process time: {} ms_____________\n",estimator_cnt,tt.toc_then_tic());
            InfoV(output_msg);
            cerr<<output_msg<<endl;
        }
        std::this_thread::sleep_for(2ms);
    }

    WarnV("ProcessMeasurements 线程退出");

}


}

