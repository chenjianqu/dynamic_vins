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



void Estimator::Optimization()
{
    TicToc t_whole, t_prepare;
    Vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    //loss_function = NULL;
    //loss_function = new ceres::CauchyLoss(1.0 / kFocalLength);
    //ceres::LossFunction* loss_function = new ceres::HuberLoss(1.0);

    for (int i = 0; i < frame + 1; i++){
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
        if ((cfg::is_estimate_ex && frame == kWinSize && Vs[0].norm() > 0.2) || openExEstimation)
            openExEstimation = true;
        else
            problem.SetParameterBlockConstant(para_ex_pose[i]);
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
        for (int i = 0; i < frame; i++)
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

        Vec3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vec3d pts_j = it_per_frame.point;
                auto *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_ex_pose[0], para_Feature[feature_index], para_Td[0]);
            }

            if(cfg::is_stereo && it_per_frame.is_stereo)
            {
                Vec3d pts_j_right = it_per_frame.pointRight;
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

    Debugv("optimization 开始优化 visual measurement count: {}", f_m_cnt);

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

    Debugv("Iterations: {}", summary.iterations.size());
    Infov("优化完成");

    if(cfg::slam == SlamType::kDynamic)
        insts_manager.GetOptimizationParameters();


    string msg="相机位姿 优化前：\n";
    msg += LogCurrentPose();

    Double2vector();

    msg+="相机位姿 优化后：\n";
    msg += LogCurrentPose();
    Debugv(msg);

    if(frame < kWinSize)
        return;

    TicToc t_whole_marginalization;
    if (margin_flag == MarginFlag::kMarginOld)
    {
        auto *marginalization_info = new MarginalizationInfo();
        Vector2double();

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

                Vec3d pts_i = it_per_id.feature_per_frame[0].point;

                for (auto &it_per_frame : it_per_id.feature_per_frame)
                {
                    imu_j++;
                    if(imu_i != imu_j)
                    {
                        Vec3d pts_j = it_per_frame.point;
                        auto *f_td = new ProjectionTwoFrameOneCamFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                        it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td);
                        auto *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                          vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_ex_pose[0], para_Feature[feature_index], para_Td[0]},
                                                                          vector<int>{0, 3});
                        marginalization_info->addResidualBlockInfo(residual_block_info);
                    }
                    if(cfg::is_stereo && it_per_frame.is_stereo)
                    {
                        Vec3d pts_j_right = it_per_frame.pointRight;
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
        Debugv("pre marginalization {} ms", t_pre_margin.Toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        Debugv("marginalization {} ms", t_margin.Toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= kWinSize; i++)
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
        std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[kWinSize - 1]))
        {

            auto *marginalization_info = new MarginalizationInfo();
            Vector2double();
            if (last_marginalization_info && last_marginalization_info->valid)
            {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++)
                {
                    assert(last_marginalization_parameter_blocks[i] != para_SpeedBias[kWinSize - 1]);
                    if (last_marginalization_parameter_blocks[i] == para_Pose[kWinSize - 1])
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
            Infov("begin marginalization");
            marginalization_info->preMarginalize();
            Infov("end pre marginalization, {} ms", t_pre_margin.Toc());

            TicToc t_margin;
            Infov("begin marginalization");
            marginalization_info->marginalize();
            Infov("end marginalization, {} ms", t_margin.Toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= kWinSize; i++)
            {
                if (i == kWinSize - 1)
                    continue;
                else if (i == kWinSize)
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
    process_mutex.lock();
    while(!acc_buf.empty())acc_buf.pop();
    while(!gyr_buf.empty())gyr_buf.pop();
    while(!feature_buf.empty())feature_buf.pop();

    prev_time = -1;
    cur_time = 0;
    openExEstimation = false;
    initP = Vec3d(0, 0, 0);
    initR = Mat3d::Identity();
    input_image_cnt = 0;
    is_init_first_pose = false;

    for (int i = 0; i < kWinSize + 1; i++){
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();
        delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < cfg::kCamNum; i++){
        tic[i] = Vec3d::Zero();
        ric[i] = Mat3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame = 0;
    solver_flag = SolverFlag::kInitial;
    initial_timestamp = 0;
    all_image_frame.clear();

    delete tmp_pre_integration;
    delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();
    f_manager.ClearState();
    failure_occur = false;

    process_mutex.unlock();
}

void Estimator::SetParameter()
{
    process_mutex.lock();
    for (int i = 0; i < cfg::kCamNum; i++){
        tic[i] = cfg::TIC[i];
        ric[i] = cfg::RIC[i];
        Infov("SetParameter Extrinsic Cam {}:\n{} \n{}",i,EigenToStr(ric[i]),VecToStr(tic[i]));
    }
    f_manager.SetRic(ric);
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

    td = cfg::TD;
    g = cfg::G;

    Infov("SetParameter Set g:{}", VecToStr(g));
    process_mutex.unlock();
}

void Estimator::ChangeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    process_mutex.lock();
    if(!use_imu && !use_stereo){
        printf("at least use two sensors! \n");
    }
    else{
        if(cfg::is_use_imu != use_imu){
            cfg::is_use_imu = use_imu;
            if(cfg::is_use_imu){
                restart = true;
            }
            else{
                delete last_marginalization_info;
                tmp_pre_integration = nullptr;
                last_marginalization_info = nullptr;
                last_marginalization_parameter_blocks.clear();
            }
        }
        cfg::is_stereo = use_stereo;
        printf("use imu %d use stereo %d\n", cfg::is_use_imu, cfg::is_stereo);
    }
    process_mutex.unlock();

    if(restart){
        ClearState();
        SetParameter();
    }
}


void Estimator::InputIMU(double t, const Vec3d &linear_acc, const Vec3d &angular_val)
{
    buf_mutex.lock();
    acc_buf.push({t, linear_acc});
    gyr_buf.push({t, angular_val});
    buf_mutex.unlock();
    if (solver_flag == SolverFlag::kNonLinear){
        propogate_mutex.lock();
        FastPredictIMU(t, linear_acc, angular_val);
        pubLatestOdometry(latest_P, latest_Q, latest_V, t);
        propogate_mutex.unlock();
    }
}


bool Estimator::GetIMUInterval(double t0, double t1, vector<pair<double, Vec3d>> &acc_vec,
                               vector<pair<double, Vec3d>> &gyr_vec){
    if(acc_buf.empty()){
        printf("not receive imu\n");
        return false;
    }
    if(t1 <= acc_buf.back().first){
        while (acc_buf.front().first <= t0){
            acc_buf.pop();
            gyr_buf.pop();
        }
        while (acc_buf.front().first < t1){
            acc_vec.push_back(acc_buf.front());
            acc_buf.pop();
            gyr_vec.push_back(gyr_buf.front());
            gyr_buf.pop();
        }
        acc_vec.push_back(acc_buf.front());
        gyr_vec.push_back(gyr_buf.front());
    }
    else{
        printf("wait for imu\n");
        return false;
    }
    return true;
}



void Estimator::InitFirstIMUPose(vector<pair<double, Vec3d>> &acc_vec)
{
    printf("init first imu pose\n");
    is_init_first_pose = true;
    //return;
    Vec3d aver_acc(0, 0, 0);
    int n = (int)acc_vec.size();
    for(size_t i = 0; i < acc_vec.size(); i++)
        aver_acc = aver_acc + acc_vec[i].second;
    aver_acc = aver_acc / n;
    printf("averge acc %f %f %f\n", aver_acc.x(), aver_acc.y(), aver_acc.z());
    Mat3d R0 = Utility::g2R(aver_acc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Vec3d{-yaw, 0, 0}) * R0;
    Rs[0] = R0;
    cout << "init R0 " << endl << Rs[0] << endl;
    //Vs[0] = Vec3d(5, 0, 0);
}


void Estimator::ProcessIMU(double t, double dt, const Vec3d &linear_acceleration, const Vec3d &angular_velocity)
{
    if (!first_imu){
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame]){
        pre_integrations[frame] = new IntegrationBase{acc_0, gyr_0, Bas[frame], Bgs[frame]};
    }
    if (frame != 0){
        pre_integrations[frame]->push_back(dt, linear_acceleration, angular_velocity);
        //if(solver_flag != NON_LINEAR)
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);
        dt_buf[frame].push_back(dt);
        linear_acceleration_buf[frame].push_back(linear_acceleration);
        angular_velocity_buf[frame].push_back(angular_velocity);
        int j = frame;
        Vec3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vec3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vec3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vec3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

bool Estimator::InitialStructure()
{
    TicToc t_sfm;
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it;
        Vec3d sum_g;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vec3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            sum_g += tmp_g;
        }
        Vec3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1);
        double var = 0;
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++)
        {
            double dt = frame_it->second.pre_integration->sum_dt;
            Vec3d tmp_g = frame_it->second.pre_integration->delta_v / dt;
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g);
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));
        //Warnv("IMU variation %f!", var);
        if(var < 0.25) Warnv("IMU excitation not enouth!");
    }
    // global sfm
    Quaterniond Q[frame + 1];
    Vec3d T[frame + 1];
    map<int, Vec3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (auto &it_per_id : f_manager.feature)
    {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (auto &it_per_frame : it_per_id.feature_per_frame){
            imu_j++;
            Vec3d pts_j = it_per_frame.point;
            tmp_feature.observation.emplace_back(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()});
        }
        sfm_f.push_back(tmp_feature);
    }
    Mat3d relative_R;
    Vec3d relative_T;
    int l;
    if (!RelativePose(relative_R, relative_T, l)){
        Warnv("Not enough features or parallax; Move device around");
        return false;
    }
    GlobalSFM sfm;
    if(!sfm.construct(frame + 1, Q, T, l,
                      relative_R, relative_T,
                      sfm_f, sfm_tracked_points)){
        Warnv("global SFM failed!");
        margin_flag = MarginFlag::kMarginOld;
        return false;
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it;
    map<int, Vec3d>::iterator it;
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
        Mat3d R_inital = (Q[i].inverse()).toRotationMatrix();
        Vec3d P_inital = - R_inital * T[i];
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
                    Vec3d world_pts = it->second;
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
            Debugv("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1))
        {
            Debugv("solve pnp fail!");
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
    if (VisualInitialAlign()){
        return true;
    }
    else{
        Warnv("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::VisualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if(!result){
        Debugv("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame; i++)
    {
        Mat3d Ri = all_image_frame[headers[i]].R;
        Vec3d Pi = all_image_frame[headers[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= kWinSize; i++)
    {
        pre_integrations[i]->repropagate(Vec3d::Zero(), Bgs[i]);
    }
    for (int i = frame; i >= 0; i--)
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

    Mat3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Vec3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Mat3d rot_diff = R0 * Rs[0].transpose();
    Mat3d rot_diff = R0;
    for (int i = 0; i <= frame; i++)
    {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    Debugv("g0:{}",VecToStr(g));
    Debugv("my R0 :{}", VecToStr(Utility::R2ypr(Rs[0])));

    f_manager.ClearDepth();
    f_manager.triangulate(frame, Ps, Rs, tic, ric);

    return true;
}

bool Estimator::RelativePose(Mat3d &relative_R, Vec3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < kWinSize; i++)
    {
        vector<pair<Vec3d, Vec3d>> corres;
        corres = f_manager.GetCorresponding(i, kWinSize);
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
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T)){
                l = i;
                Debugv("Average_parallax {} choose l {} and newest frame to triangulate the whole structure",average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::Vector2double()
{
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


    VectorXd dep = f_manager.GetDepthVector();
    for (int i = 0; i < f_manager.GetFeatureCount(); i++)
        para_Feature[i][0] = dep(i);

    para_Td[0][0] = td;
}

void Estimator::Double2vector()
{
    /*    printf("相机位姿优化前:");
        for (int i = 0; i <= kWindowSize; i++)
            printf("%d:(%.3lf,%.3lf,%.3lf) ",i,Ps[i].x(),Ps[i].y(),Ps[i].z());
        printf("\n");*/

    Vec3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vec3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = false;
    }

    if(cfg::is_use_imu)
    {
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
            Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

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
            Rs[i] = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();

            Ps[i] = Vec3d(para_Pose[i][0], para_Pose[i][1], para_Pose[i][2]);
        }
    }

    if(cfg::is_use_imu)
    {
        for (int i = 0; i < cfg::kCamNum; i++)
        {
            tic[i] = Vec3d(para_ex_pose[i][0],
                              para_ex_pose[i][1],
                              para_ex_pose[i][2]);
            ric[i] = Quaterniond(para_ex_pose[i][6],
                                 para_ex_pose[i][3],
                                 para_ex_pose[i][4],
                                 para_ex_pose[i][5]).normalized().toRotationMatrix();
        }
    }

    VectorXd dep = f_manager.GetDepthVector();
    for (int i = 0; i < f_manager.GetFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.SetDepth(dep);

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

bool Estimator::FailureDetection()
{
    return false;
    if (f_manager.last_track_num < 2){
        Infov(" little feature %d", f_manager.last_track_num);
    }
    if (Bas[kWinSize].norm() > 2.5){
        Infov(" big IMU acc bias estimation %f", Bas[kWinSize].norm());
        return true;
    }
    if (Bgs[kWinSize].norm() > 1.0){
        Infov(" big IMU gyr bias estimation %f", Bgs[kWinSize].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        Infov(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vec3d tmp_P = Ps[kWinSize];
    if ((tmp_P - last_P).norm() > 5){
        //Infov(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1){
        //Infov(" big z translation");
        //return true;
    }
    Mat3d tmp_R = Rs[kWinSize];
    Mat3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50){
        Infov(" big delta_angle ");
    }
    return false;
}

void Estimator::SlideWindow()
{
    TicToc t_margin;
    if (margin_flag == MarginFlag::kMarginOld)
    {
        double t_0 = headers[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame == kWinSize)
        {
            for (int i = 0; i < kWinSize; i++)
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
            headers[kWinSize] = headers[kWinSize - 1];
            Ps[kWinSize] = Ps[kWinSize - 1];
            Rs[kWinSize] = Rs[kWinSize - 1];

            if(cfg::is_use_imu)
            {
                Vs[kWinSize] = Vs[kWinSize - 1];
                Bas[kWinSize] = Bas[kWinSize - 1];
                Bgs[kWinSize] = Bgs[kWinSize - 1];

                delete pre_integrations[kWinSize];
                pre_integrations[kWinSize] = new IntegrationBase{acc_0, gyr_0, Bas[kWinSize], Bgs[kWinSize]};

                dt_buf[kWinSize].clear();
                linear_acceleration_buf[kWinSize].clear();
                angular_velocity_buf[kWinSize].clear();
            }

            if (true || solver_flag == SolverFlag::kInitial)
            {
                map<double, ImageFrame>::iterator it_0;
                it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }
            SlideWindowOld();
        }
    }
    else
    {
        if (frame == kWinSize)
        {
            headers[frame - 1] = headers[frame];
            Ps[frame - 1] = Ps[frame];
            Rs[frame - 1] = Rs[frame];

            if(cfg::is_use_imu)
            {
                for (unsigned int i = 0; i < dt_buf[frame].size(); i++)
                {
                    double tmp_dt = dt_buf[frame][i];
                    Vec3d tmp_linear_acceleration = linear_acceleration_buf[frame][i];
                    Vec3d tmp_angular_velocity = angular_velocity_buf[frame][i];

                    pre_integrations[frame - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

                    dt_buf[frame - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame - 1].push_back(tmp_angular_velocity);
                }

                Vs[frame - 1] = Vs[frame];
                Bas[frame - 1] = Bas[frame];
                Bgs[frame - 1] = Bgs[frame];

                delete pre_integrations[kWinSize];
                pre_integrations[kWinSize] = new IntegrationBase{acc_0, gyr_0, Bas[kWinSize], Bgs[kWinSize]};

                dt_buf[kWinSize].clear();
                linear_acceleration_buf[kWinSize].clear();
                angular_velocity_buf[kWinSize].clear();
            }
            SlideWindowNew();
        }
    }
}

void Estimator::SlideWindowNew()
{
    sum_of_front++;
    f_manager.RemoveFront(frame);
}

void Estimator::SlideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == SolverFlag::kNonLinear ? true : false;
    if (shift_depth)
    {
        Mat3d R0, R1;
        Vec3d P0, P1;
        R0 = back_R0 * ric[0];
        R1 = Rs[0] * ric[0];
        P0 = back_P0 + back_R0 * tic[0];
        P1 = Ps[0] + Rs[0] * tic[0];
        f_manager.RemoveBackShiftDepth(R0, P0, R1, P1);
    }
    else
        f_manager.RemoveBack();
}


void Estimator::GetPoseInWorldFrame(Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[frame];
    T.block<3, 1>(0, 3) = Ps[frame];
}

void Estimator::GetPoseInWorldFrame(int index, Eigen::Matrix4d &T)
{
    T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = Rs[index];
    T.block<3, 1>(0, 3) = Ps[index];
}

void Estimator::PredictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame < 2)
        return;
    // predict next pose. Assume constant velocity motion
    Eigen::Matrix4d curT, prevT, nextT;
    GetPoseInWorldFrame(curT);
    GetPoseInWorldFrame(frame - 1, prevT);
    nextT = curT * (prevT.inverse() * curT);
    map<int, Vec3d> predictPts;

    for (auto &it_per_id : f_manager.feature)
    {
        if(it_per_id.estimated_depth > 0)
        {
            int firstIndex = it_per_id.start_frame;
            int lastIndex = it_per_id.start_frame + it_per_id.feature_per_frame.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)it_per_id.feature_per_frame.size() >= 2 && lastIndex == frame)
            {
                double depth = it_per_id.estimated_depth;
                Vec3d pts_j = ric[0] * (depth * it_per_id.feature_per_frame[0].point) + tic[0];
                Vec3d pts_w = Rs[firstIndex] * pts_j + Ps[firstIndex];
                Vec3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vec3d pts_cam = ric[0].transpose() * (pts_local - tic[0]);
                int ptsIndex = it_per_id.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    //featureTracker->setPrediction(predictPts);
    //printf("e output %d predict pts\n",(int)predictPts.size());
}

double Estimator::ReprojectionError(Mat3d &Ri, Vec3d &Pi, Mat3d &rici, Vec3d &tici,
                                    Mat3d &Rj, Vec3d &Pj, Mat3d &ricj, Vec3d &ticj,
                                    double depth, Vec3d &uvi, Vec3d &uvj){
    Vec3d pts_w = Ri * (rici * (depth * uvi) + tici) + Pi;
    Vec3d pts_cj = ricj.transpose() * (Rj.transpose() * (pts_w - Pj) - ticj);
    Vector2d residual = (pts_cj / pts_cj.z()).head<2>() - uvj.head<2>();
    double rx = residual.x();
    double ry = residual.y();
    return sqrt(rx * rx + ry * ry);
}

void Estimator::OutliersRejection(set<int> &removeIndex)
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
        Vec3d pts_i = it_per_id.feature_per_frame[0].point;
        double depth = it_per_id.estimated_depth;
        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;
            if (imu_i != imu_j)
            {
                Vec3d pts_j = it_per_frame.point;
                double tmp_error = ReprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                     Rs[imu_j], Ps[imu_j], ric[0], tic[0],
                                                     depth, pts_i, pts_j);
                err += tmp_error;
                errCnt++;
                //printf("tmp_error %f\n", kFocalLength / 1.5 * tmp_error);
            }
            // need to rewrite projecton factor.........
            if(cfg::is_stereo && it_per_frame.is_stereo)
            {

                Vec3d pts_j_right = it_per_frame.pointRight;
                if(imu_i != imu_j)
                {
                    double tmp_error = ReprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
                                                         Rs[imu_j], Ps[imu_j], ric[1], tic[1],
                                                         depth, pts_i, pts_j_right);
                    err += tmp_error;
                    errCnt++;
                    //printf("tmp_error %f\n", kFocalLength / 1.5 * tmp_error);
                }
                else
                {
                    double tmp_error = ReprojectionError(Rs[imu_i], Ps[imu_i], ric[0], tic[0],
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

void Estimator::FastPredictIMU(double t, Vec3d linear_acceleration, Vec3d angular_velocity)
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


void Estimator::UpdateLatestStates(){
    propogate_mutex.lock();
    latest_time = headers[frame] + td;
    latest_P = Ps[frame];
    latest_Q = Rs[frame];
    latest_V = Vs[frame];
    latest_Ba = Bas[frame];
    latest_Bg = Bgs[frame];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    buf_mutex.lock();
    queue<pair<double, Vec3d>> tmp_accBuf = acc_buf;
    queue<pair<double, Vec3d>> tmp_gyrBuf = gyr_buf;
    buf_mutex.unlock();
    while(!tmp_accBuf.empty())
    {
        double t = tmp_accBuf.front().first;
        Vec3d acc = tmp_accBuf.front().second;
        Vec3d gyr = tmp_gyrBuf.front().second;
        FastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    propogate_mutex.unlock();
}

void Estimator::ProcessImage(const FeatureMap &image, const double header){
    Infov("processImage Adding feature points:{}", image.size());
    if (f_manager.AddFeatureCheckParallax(frame, image, td))
        margin_flag = MarginFlag::kMarginOld;
    else
        margin_flag = MarginFlag::kMarginSecondNew;

    Debugv("processImage margin_flag:{}", margin_flag == MarginFlag::kMarginSecondNew ? "Non-keyframe" : "Keyframe");
    Debugv("processImage frame_count {}", frame);
    Debugv("processImage all feature size: {}", f_manager.GetFeatureCount());

    headers[frame] = header;

    ImageFrame img_frame(image, header);
    img_frame.pre_integration = tmp_pre_integration;
    all_image_frame.insert({header, img_frame});
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame], Bgs[frame]};

    if(cfg::is_estimate_ex == 2){
        Infov("calibrating extrinsic param, rotation movement is needed");
        if (frame != 0){
            auto cor = f_manager.GetCorresponding(frame - 1, frame);
            if (Mat3d calib_ric;initial_ex_rotation.CalibrationExRotation(cor, pre_integrations[frame]->delta_q, calib_ric)){
                Debugv("initial extrinsic rotation calib success");
                Debugv("initial extrinsic rotation:\n{}", EigenToStr(calib_ric));
                ric[0] = calib_ric;
                cfg::RIC[0] = calib_ric;
                cfg::is_estimate_ex = 1;
            }
        }
    }


    if(cfg::slam == SlamType::kDynamic){
        insts_manager.PredictCurrentPose();

        insts_manager.Triangulate(frame);
        insts_manager.InitialInstance();
    }



    if (solver_flag == SolverFlag::kInitial)
    {
        // monocular + IMU initilization
        if (!cfg::is_stereo && cfg::is_use_imu){
            if (frame == kWinSize){
                bool result = false;
                if(cfg::is_estimate_ex != 2 && (header - initial_timestamp) > 0.1){
                    result = InitialStructure();
                    initial_timestamp = header;
                }
                if(result){
                    Optimization();
                    UpdateLatestStates();
                    solver_flag = SolverFlag::kNonLinear;
                    SlideWindow();
                    Infov("Initialization finish!");
                }
                else{
                    SlideWindow();
                }
            }
        }
        // stereo + IMU initilization
        else if(cfg::is_stereo && cfg::is_use_imu)
        {
            f_manager.initFramePoseByPnP(frame, Ps, Rs, tic, ric);
            f_manager.triangulate(frame, Ps, Rs, tic, ric);
            if (frame == kWinSize){
                map<double, ImageFrame>::iterator frame_it;
                int i = 0;
                for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++){
                    frame_it->second.R = Rs[i];
                    frame_it->second.T = Ps[i];
                    i++;
                }
                solveGyroscopeBias(all_image_frame, Bgs);
                for (int j = 0; j <= kWinSize; j++)
                    pre_integrations[j]->repropagate(Vec3d::Zero(), Bgs[j]);
                Optimization();
                UpdateLatestStates();
                solver_flag = SolverFlag::kNonLinear;
                SlideWindow();
                Infov("Initialization finish!");
            }
        }

        // stereo only initilization
        else if(cfg::is_stereo && !cfg::is_use_imu){
            f_manager.initFramePoseByPnP(frame, Ps, Rs, tic, ric);
            f_manager.triangulate(frame, Ps, Rs, tic, ric);
            Optimization();
            if(frame == kWinSize)
            {
                Optimization();
                UpdateLatestStates();
                solver_flag = SolverFlag::kNonLinear;
                SlideWindow();
                Infov("Initialization finish!");
            }
        }

        if(frame < kWinSize){
            frame++;
            int prev_frame = frame - 1;
            Ps[frame] = Ps[prev_frame];
            Vs[frame] = Vs[prev_frame];
            Rs[frame] = Rs[prev_frame];
            Bas[frame] = Bas[prev_frame];
            Bgs[frame] = Bgs[prev_frame];
        }

    }
    else
    {
        TicToc t_solve;

        if(!cfg::is_use_imu)
            f_manager.initFramePoseByPnP(frame, Ps, Rs, tic, ric);

        f_manager.triangulate(frame, Ps, Rs, tic, ric);

        Optimization();

        set<int> removeIndex;
        OutliersRejection(removeIndex);
        f_manager.RemoveOutlier(removeIndex);

        Debugv("solver costs:{} ms", t_solve.Toc());

        if (FailureDetection())
        {
            Warnv("failure detection!");
            failure_occur = true;
            ClearState();
            SetParameter();
            Warnv("system reboot!");
            return;
        }

        if(cfg::slam == SlamType::kDynamic){
            //insts_manager.SetWindowPose();
            insts_manager.SlideWindow();
        }

        SlideWindow();

        //printf("外点剔除\n");

        if(cfg::slam == SlamType::kDynamic)
            insts_manager.OutliersRejection();

        f_manager.RemoveFailures();

        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= kWinSize; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[kWinSize];
        last_P = Ps[kWinSize];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
        UpdateLatestStates();
    }


    if(cfg::slam == SlamType::kDynamic){
        insts_manager.SetInstanceCurrentPoint3d();
    }

}

void Estimator::ProcessMeasurements(){
    static TicToc tt;
    while (cfg::ok.load(std::memory_order_seq_cst))
    {
        if(!feature_buf.empty())
        {
            FeatureFrame feature_frame = feature_buf.front();
            cur_time = feature_frame.time + td;
            if(cfg::is_use_imu && !IMUAvailable(cur_time)){
                std::cerr<<"wait for imu ..."<<endl;
                std::this_thread::sleep_for(5ms);
                continue;
            }
            InstancesFeatureMap curr_insts;
            if(cfg::slam == SlamType::kDynamic)
                curr_insts=instances_buf.front();

            Warnv("----------Time : {} ----------", feature_frame.time);

            tt.Tic();
            vector<pair<double, Vec3d>> acc_vec, gyr_vec;
            buf_mutex.lock();
            if(cfg::is_use_imu)
                GetIMUInterval(prev_time, cur_time, acc_vec, gyr_vec);

            feature_buf.pop();
            if(cfg::slam == SlamType::kDynamic)
                instances_buf.pop();

            buf_mutex.unlock();

            if(cfg::is_use_imu)
            {
                if(!is_init_first_pose)
                    InitFirstIMUPose(acc_vec);
                for(size_t i = 0; i < acc_vec.size(); i++){
                    double dt;
                    if(i == 0)
                        dt = acc_vec[i].first - prev_time;
                    else if (i == acc_vec.size() - 1)
                        dt = cur_time - acc_vec[i - 1].first;
                    else
                        dt = acc_vec[i].first - acc_vec[i - 1].first;
                    ProcessIMU(acc_vec[i].first, dt, acc_vec[i].second, gyr_vec[i].second);
                }
            }
            process_mutex.lock();

            Infov("get input time: {} ms", tt.TocThenTic());
            Debugv("solver_flag:{}", solver_flag == SolverFlag::kInitial ? "INITIAL" : "NO-LINEAR");

            if(cfg::slam == SlamType::kDynamic)
                insts_manager.PushBack(frame, curr_insts);

            ProcessImage(feature_frame.features, feature_frame.time);
            prev_time = cur_time;

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
            process_mutex.unlock();

            static unsigned int estimator_cnt=0;
            estimator_cnt++;
            auto output_msg=fmt::format("cnt:{} ___________process time: {} ms_____________\n",estimator_cnt,
                                        tt.TocThenTic());
            Infov(output_msg);
            cerr<<output_msg<<endl;
        }
        std::this_thread::sleep_for(2ms);
    }

    Warnv("ProcessMeasurements 线程退出");

}


}

