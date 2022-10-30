/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai Univebody.Rsity
 *
 * This file is part of dynamic_vins.
 * Github:httbody.Ps://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong Univebody.Rsity of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/



#include <dirent.h>
#include <cstdio>

#include "estimator.h"
#include "utils/io/visualization.h"
#include "utils/def.h"
#include "vio_parameters.h"

#include "utils/utility.h"
#include "estimator/factor/pose_local_parameterization.h"
#include "estimator/factor/projection_two_frame_one_cam_factor.h"
#include "estimator/factor/projection_two_frame_two_cam_factor.h"
#include "estimator/factor/projection_one_frame_two_cam_factor.h"
#include "estimator/factor/line_parameterization.h"
#include "estimator/factor/line_projection_factor.h"

#include "utils/io/output.h"

namespace dynamic_vins{\


Estimator::Estimator(const string& config_path): feat_manager{body.Rs}
{
    para::SetParameters(config_path);

    ClearState();

    /*std::string path_test="/home/chen/slam/expriments/DynamicVINS/FlowTrack/1_vel";
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


/**
 * 添加残差块
 * @param problem
 */
void Estimator::AddInstanceParameterBlock(ceres::Problem &problem) {
    ///添加位姿顶点和bias顶点
    for (int i = 0; i < frame + 1; i++){
        ceres::LocalParameterization *lp = new PoseLocalParameterization();
        problem.AddParameterBlock(body.para_pose[i], kSizePose, lp);
        if(cfg::is_use_imu)
            problem.AddParameterBlock(body.para_speed_bias[i], kSizeSpeedBias);
    }

    if(!cfg::is_use_imu)
        problem.SetParameterBlockConstant(body.para_pose[0]);

    ///添加外参顶点
    for (int i = 0; i < cfg::kCamNum; i++){
        ceres::LocalParameterization *lp = new PoseLocalParameterization();
        problem.AddParameterBlock(body.para_ex_pose[i], kSizePose, lp);
        if ((cfg::is_estimate_ex && frame == kWinSize && body.Vs[0].norm() > 0.2) || openExEstimation)
            openExEstimation = true;
        else
            problem.SetParameterBlockConstant(body.para_ex_pose[i]);
    }

    ///Td变量
    problem.AddParameterBlock(body.para_td[0], 1);
    if (!cfg::is_estimate_td || body.Vs[0].norm() < 0.2)
        problem.SetParameterBlockConstant(body.para_td[0]);
}



/**
 * 添加重投影残差
 * @param problem
 * @param loss_function
 */
int Estimator::AddResidualBlock(ceres::Problem &problem, ceres::LossFunction *loss_function) {

    ///首先添加先验信息
    if (last_marg_info && last_marg_info->valid){
        auto *marg_factor = new MarginalizationFactor(last_marg_info);
        problem.AddResidualBlock(marg_factor,nullptr,last_marg_para_blocks);
    }

    ///IMU因子
    if(cfg::is_use_imu){
        for (int i = 0; i < frame; i++){
            int j = i + 1;
            if (pre_integrations[j]->sum_dt > 10.0)
                continue;
            auto* imu_factor = new IMUFactor(pre_integrations[j]);
            problem.AddResidualBlock(imu_factor, nullptr,
                                     body.para_pose[i], body.para_speed_bias[i], body.para_pose[j], body.para_speed_bias[j]);
        }
    }


    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &lm : feat_manager.point_landmarks){
        lm.used_num = lm.feats.size();
        if (lm.used_num < 4)
            continue;
        ++feature_index;
        int imu_i = lm.start_frame, imu_j = imu_i - 1;
        Vec3d pts_i = lm.feats[0].point;

        for (auto &feat : lm.feats){
            imu_j++;
            if (imu_i != imu_j){
                Vec3d pts_j = feat.point;
                auto *f_td = new ProjectionTwoFrameOneCamFactor(
                        pts_i, pts_j, lm.feats[0].velocity, feat.velocity,
                        lm.feats[0].cur_td, feat.cur_td);
                problem.AddResidualBlock(f_td, loss_function, body.para_pose[imu_i], body.para_pose[imu_j],
                                         body.para_ex_pose[0],
                                         body.para_point_features[feature_index], body.para_td[0]);
            }

            if(cfg::is_stereo && feat.is_stereo){
                Vec3d pts_j_right = feat.point_right;
                if(imu_i != imu_j){
                    auto *f = new ProjectionTwoFrameTwoCamFactor(pts_i, pts_j_right,
                                                                 lm.feats[0].velocity, feat.velocity_right,
                                                                 lm.feats[0].cur_td, feat.cur_td);
                    problem.AddResidualBlock(f, loss_function,
                                             body.para_pose[imu_i], body.para_pose[imu_j],
                                             body.para_ex_pose[0], body.para_ex_pose[1],
                                             body.para_point_features[feature_index], body.para_td[0]);
                }
                else{
                    auto *f = new ProjectionOneFrameTwoCamFactor(pts_i, pts_j_right,
                                                                 lm.feats[0].velocity, feat.velocity_right,
                                                                 lm.feats[0].cur_td, feat.cur_td);
                    problem.AddResidualBlock(f, loss_function,
                                             body.para_ex_pose[0], body.para_ex_pose[1],
                                             body.para_point_features[feature_index], body.para_td[0]);
                }

            }
            f_m_cnt++;
        }
    }
    return f_m_cnt;
}



/**
 * 执行非线性优化
 */
void Estimator::Optimization()
{
    TicToc tt;

    Vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    //loss_function = new ceres::CauchyLoss(1.0 / kFocalLength);

    ///添加残差块
    AddInstanceParameterBlock(problem);
    ///添加动态物体的相关顶点
    if(cfg::slam == SLAM::kDynamic){
        //Debugv(insts_manager.PrintInstancePoseInfo(false));
        im.AddInstanceParameterBlock(problem);
    }

    ///添加重投影误差
    int f_m_cnt = AddResidualBlock(problem,loss_function);

    ///添加动态物体的残差项
    if(cfg::slam == SLAM::kDynamic){
        im.AddResidualBlockForJointOpt(problem, loss_function);
    }

    Debugv("optimization | prepare:{} ms",tt.TocThenTic());
    Debugv("optimization 开始优化 visual measurement count: {}", f_m_cnt);

    //设置ceres选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = para::KNumIter;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_stebody.Ps = true;

    if (margin_flag == MarginFlag::kMarginOld)
        options.max_solver_time_in_seconds = para::kMaxSolverTime * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = para::kMaxSolverTime;

    ///求解
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Debugv("optimization 优化完成 Iterations: {}", summary.iterations.size());
    Debugv("optimization | Solve:{} ms",tt.TocThenTic());

    //string msg="相机位姿 优化前：\n" + LogCurrentPose();

    if(cfg::slam == SLAM::kDynamic){
        im.GetOptimizationParameters();
       // Debugv(insts_manager.PrintInstancePoseInfo(false));
    }

    Double2vector();

    //Debugv("相机位姿 优化后：\n" + LogCurrentPose());
    Debugv("optimization | postprocess:{} ms",tt.TocThenTic());

    if(frame < kWinSize)
        return;

    ///执行边缘化,设置先验残差
    SetMarginalizationInfo();

    Debugv("optimization | 边缘化:{} ms",tt.TocThenTic());

}


/**
 * 仅仅优化线特征
 */
void Estimator::OptimizationWithLine()
{
    TicToc tt;

    Vector2double();

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);

    ///位姿参数化，但是设置为固定
    for (int i = 0; i < kWinSize; i++){    // 将窗口内的 p,q 加入优化变量
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(body.para_pose[i], kSizePose, local_parameterization);  // p,q
        problem.SetParameterBlockConstant(body.para_pose[i]);// 固定 pose
    }
    for (int i = 0; i < cfg::kCamNum; i++) {        // 外参数
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
        problem.AddParameterBlock(body.para_ex_pose[i], kSizePose, local_parameterization);
        problem.SetParameterBlockConstant(body.para_ex_pose[i]);// 固定 外参数
    }


    /// 所有特征
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &landmark : feat_manager.line_landmarks){
        landmark.used_num = landmark.feats.size();// 已经被多少帧观测到， 这个已经在三角化那个函数里说了
        // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
        if (!(landmark.used_num >= kLineMinObs && landmark.start_frame < kWinSize - 2 && landmark.is_triangulation))
            continue;
        ++feature_index;

        ///线参数化
        ceres::LocalParameterization *line_para = new LineOrthParameterization();
        problem.AddParameterBlock(body.para_line_features[feature_index], kSizeLine, line_para);  // p,q
        int imu_i = landmark.start_frame,imu_j = imu_i - 1;

        ///设置线重投影因子
        for (auto &feat : landmark.feats){
            imu_j++;
            auto *f = new lineProjectionFactor(feat.line_obs);     // 特征重投影误差
            problem.AddResidualBlock(f, loss_function,
                                     body.para_pose[imu_j],
                                     body.para_ex_pose[0],
                                     body.para_line_features[feature_index]);
            f_m_cnt++;
        }
    }

    if(feature_index < 3){
        return;
    }

    //设置ceres选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = para::KNumIter;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_stebody.Ps = true;
    if (margin_flag == MarginFlag::kMarginOld)
        options.max_solver_time_in_seconds = para::kMaxSolverTime * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = para::kMaxSolverTime;

    ///求解
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Double2vector();

    feat_manager.RemoveLineOutlier();
}




/**
 * 设置边缘化先验信息,用于下次的优化
 */
void Estimator::SetMarginalizationInfo()
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

    ///边缘化最老帧
    if (margin_flag == MarginFlag::kMarginOld){
        Vector2double();

        auto *marg_info = new MarginalizationInfo();

        ///之前的边缘化信息
        if (last_marg_info && last_marg_info->valid){
            vector<int> drop_set;
            for (int i = 0; i < last_marg_para_blocks.size(); i++){
                if (last_marg_para_blocks[i] == body.para_pose[0] || last_marg_para_blocks[i] == body.para_speed_bias[0])
                    drop_set.push_back(i);
            }
            //根据上一时刻的边缘信息构建一个边缘化因子
            auto *marg_factor = new MarginalizationFactor(last_marg_info);
            auto *residual_block_info = new ResidualBlockInfo(
                    marg_factor, nullptr,last_marg_para_blocks,drop_set);
            marg_info->addResidualBlockInfo(residual_block_info);
        }
        ///IMU的边缘化信息
        if(cfg::is_use_imu){
            if (pre_integrations[1]->sum_dt < 10.0) {
                auto* imu_factor = new IMUFactor(pre_integrations[1]);
                auto *residual_block_info = new ResidualBlockInfo(
                        imu_factor, nullptr,
                        vector<double *>{body.para_pose[0], body.para_speed_bias[0], body.para_pose[1], body.para_speed_bias[1]},
                        vector<int>{0, 1});
                marg_info->addResidualBlockInfo(residual_block_info);
            }
        }

        ///将最老帧上的特征点的边缘化信息保留下来
        int feature_index = -1;
        for (auto &lm : feat_manager.point_landmarks) {
            lm.used_num = lm.feats.size();
            if (lm.used_num < 4)
                continue;
            ++feature_index;
            int imu_i = lm.start_frame, imu_j = imu_i - 1;
            if (imu_i != 0)//只处理最老帧上特征点的信息
                continue;

            Vec3d pts_i = lm.feats[0].point;

            for (auto &feat : lm.feats){
                imu_j++;
                if(imu_i != imu_j){
                    Vec3d pts_j = feat.point;
                    auto *f_td = new ProjectionTwoFrameOneCamFactor(
                            pts_i, pts_j, lm.feats[0].velocity, feat.velocity,
                            lm.feats[0].cur_td, feat.cur_td);
                    auto *residual_block_info = new ResidualBlockInfo(
                            f_td, loss_function,
                            vector<double *>{body.para_pose[imu_i], body.para_pose[imu_j],
                                             body.para_ex_pose[0], body.para_point_features[feature_index], body.para_td[0]},
                                             vector<int>{0, 3});
                    marg_info->addResidualBlockInfo(residual_block_info);
                }
                if(cfg::is_stereo && feat.is_stereo){
                    Vec3d pts_j_right = feat.point_right;
                    if(imu_i != imu_j){
                        auto *f = new ProjectionTwoFrameTwoCamFactor(
                                pts_i, pts_j_right, lm.feats[0].velocity, feat.velocity_right,
                                lm.feats[0].cur_td, feat.cur_td);
                        auto *residual_block_info = new ResidualBlockInfo(
                                f, loss_function,
                                vector<double *>{body.para_pose[imu_i], body.para_pose[imu_j], body.para_ex_pose[0],
                                                 body.para_ex_pose[1], body.para_point_features[feature_index], body.para_td[0]},
                                                 vector<int>{0, 4});
                        marg_info->addResidualBlockInfo(residual_block_info);
                    }
                    else{
                        auto *f = new ProjectionOneFrameTwoCamFactor(
                                pts_i, pts_j_right, lm.feats[0].velocity, feat.velocity_right,
                                lm.feats[0].cur_td, feat.cur_td);
                        auto *residual_block_info = new ResidualBlockInfo(
                                f, loss_function,
                                vector<double *>{body.para_ex_pose[0], body.para_ex_pose[1], body.para_point_features[feature_index],body.para_td[0]},
                                vector<int>{2});
                        marg_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        /// Line特征
          /*int linefeature_index = -1;
        for (auto &landmark : f_manager.line_landmarks){
            landmark.used_num = landmark.feats.size(); // 已经被多少帧观测到， 这个已经在三角化那个函数里说了
            // 如果这个特征才被观测到，那就跳过。实际上这里为啥不直接用如果特征没有三角化这个条件。
            if (!(landmark.used_num >= kLineMinObs && landmark.start_frame < kWinSize - 2 && landmark.is_triangulation))
                continue;
            ++linefeature_index;            // 这个变量会记录feature在 para_Feature 里的位置， 将深度存入para_Feature时索引的记录也是用的这种方式

            int imu_i = landmark.start_frame,imu_j = imu_i - 1;
            if (imu_i != 0)   // 如果这个特征的初始帧 不对应 要marg掉的最老帧0, 那就不用marg这个特征。即marg掉帧的时候，我们marg掉这帧上三角化的那些点
                continue;

            for (auto &feat : landmark.feats){
                imu_j++;
                std::vector<int> drop_set;
                if(imu_i == imu_j){
                    continue;
                }else{
                    drop_set = vector<int>{2};      // marg feature
                }

                Vector4d obs = feat.lineobs; // 在第j帧图像上的观测
                auto *f = new lineProjectionFactor(obs);     // 特征重投影误差
                auto *residual_block_info = new ResidualBlockInfo(
                        f, loss_function,
                        vector<double *>{body.para_pose[imu_j], body.para_ex_pose[0], body.para_line_features[linefeature_index]},drop_set);
                        marg_info->addResidualBlockInfo(residual_block_info);
            }
        }*/



        TicToc t_pre_margin;
        marg_info->preMarginalize();
        Debugv("pre marginalization {} ms", t_pre_margin.Toc());

        ///执行边缘化
        TicToc t_margin;
        marg_info->marginalize();
        Debugv("marginalization {} ms", t_margin.Toc());

        ///设置地址偏移
        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= kWinSize; i++){
            addr_shift[reinterpret_cast<long>(body.para_pose[i])] = body.para_pose[i - 1];
            if(cfg::is_use_imu)
                addr_shift[reinterpret_cast<long>(body.para_speed_bias[i])] = body.para_speed_bias[i - 1];
        }
        for (int i = 0; i < cfg::kCamNum; i++)
            addr_shift[reinterpret_cast<long>(body.para_ex_pose[i])] = body.para_ex_pose[i];
        addr_shift[reinterpret_cast<long>(body.para_td[0])] = body.para_td[0];

        ///设置keep_block_data
        vector<double *> parameter_blocks = marg_info->getParameterBlocks(addr_shift);

        delete last_marg_info;
        last_marg_info = marg_info;
        last_marg_para_blocks = parameter_blocks;
    }
    ///边缘化次新帧
    else
    {
        if (last_marg_info &&
        std::count(std::begin(last_marg_para_blocks), std::end(last_marg_para_blocks),body.para_pose[kWinSize - 1])){
            auto *marg_info = new MarginalizationInfo();
            Vector2double();
            ///仅添加先验因子
            if (last_marg_info && last_marg_info->valid){
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marg_para_blocks.size()); i++){
                    assert(last_marg_para_blocks[i] != body.para_speed_bias[kWinSize - 1]);
                    if (last_marg_para_blocks[i] == body.para_pose[kWinSize - 1])
                        drop_set.push_back(i);
                }
                // construct new marginlization_factor
                auto *marginalization_factor = new MarginalizationFactor(last_marg_info);
                auto *residual_block_info = new ResidualBlockInfo(
                        marginalization_factor, nullptr,
                        last_marg_para_blocks,drop_set);
                marg_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            Infov("begin marginalization");
            marg_info->preMarginalize();

            Infov("end pre marginalization, {} ms", t_pre_margin.Toc());

            TicToc t_margin;
            Infov("begin marginalization");
            marg_info->marginalize();

            Infov("end marginalization, {} ms", t_margin.Toc());

            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= kWinSize; i++){
                if (i == kWinSize - 1){
                    continue;
                }
                else if (i == kWinSize){
                    addr_shift[reinterpret_cast<long>(body.para_pose[i])] = body.para_pose[i - 1];
                    if(cfg::is_use_imu)
                        addr_shift[reinterpret_cast<long>(body.para_speed_bias[i])] = body.para_speed_bias[i - 1];
                }
                else{
                    addr_shift[reinterpret_cast<long>(body.para_pose[i])] = body.para_pose[i];
                    if(cfg::is_use_imu)
                        addr_shift[reinterpret_cast<long>(body.para_speed_bias[i])] = body.para_speed_bias[i];
                }
            }

            for (int i = 0; i < cfg::kCamNum; i++){
                addr_shift[reinterpret_cast<long>(body.para_ex_pose[i])] = body.para_ex_pose[i];
            }
            addr_shift[reinterpret_cast<long>(body.para_td[0])] = body.para_td[0];
            vector<double *> parameter_blocks = marg_info->getParameterBlocks(addr_shift);
            delete last_marg_info;
            last_marg_info = marg_info;
            last_marg_para_blocks = parameter_blocks;
        }

    }
}


void Estimator::ClearState()
{
    process_mutex.lock();
    while(!acc_buf.empty())acc_buf.pop();
    while(!gyr_buf.empty())gyr_buf.pop();

    feature_queue.clear();

    prev_time = -1;
    cur_time = 0;
    openExEstimation = false;
    initP = Vec3d(0, 0, 0);
    initR = Mat3d::Identity();
    input_image_cnt = 0;
    is_init_first_pose = false;

    for (int i = 0; i < kWinSize + 1; i++){
        body.Rs[i].setIdentity();
        body.Ps[i].setZero();
        body.Vs[i].setZero();
        body.Bas[i].setZero();
        body.Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();
        delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < cfg::kCamNum; i++){
        body.tic[i] = Vec3d::Zero();
        body.ric[i] = Mat3d::Identity();
    }

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame = 0;
    solver_flag = SolverFlag::kInitial;
    initial_timestamp = 0;
    all_image_frame.clear();

    delete tmp_pre_integration;
    delete last_marg_info;

    tmp_pre_integration = nullptr;
    last_marg_info = nullptr;
    last_marg_para_blocks.clear();
    feat_manager.ClearState();
    failure_occur = false;

    process_mutex.unlock();
}


void Estimator::SetParameter()
{
    process_mutex.lock();
    for (int i = 0; i < cfg::kCamNum; i++){
        body.tic[i] = para::TIC[i];
        body.ric[i] = para::RIC[i];
        Infov("SetParameter Extrinsic Cam {}:\n{} \n{}",i,EigenToStr(body.ric[i]),VecToStr(body.tic[i]));
    }
    feat_manager.SetRic(body.ric);
    ProjectionTwoFrameOneCamFactor::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();
    ProjectionTwoFrameTwoCamFactor::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();
    ProjectionOneFrameTwoCamFactor::sqrt_info = kFocalLength / 1.5 * Matrix2d::Identity();

    body.td = para::TD;
    body.g = para::G;

    Infov("SetParameter Set g:{}", VecToStr(body.g));
    process_mutex.unlock();
}


void Estimator::ChangeSensorType(int use_imu, int use_stereo)
{
    bool restart = false;
    process_mutex.lock();
    if(!use_imu && !use_stereo){
        printf("at least use two sensobody.Rs! \n");
    }
    else{
        if(cfg::is_use_imu != use_imu){
            cfg::is_use_imu = use_imu;
            if(cfg::is_use_imu){
                restart = true;
            }
            else{
                delete last_marg_info;
                tmp_pre_integration = nullptr;
                last_marg_info = nullptr;
                last_marg_para_blocks.clear();
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
        Publisher::PubLatestOdometry(latest_P, latest_Q, latest_V, t);
        propogate_mutex.unlock();
    }
}

/**
 * 返回t0到t1时间段内的IMU测量值, 其中t0为上一时刻, t1为当前时刻
 * @param t0
 * @param t1
 * @param acc_vec
 * @param gyr_vec
 * @return
 */
bool Estimator::GetIMUInterval(double t0, double t1,
                               vector<pair<double, Vec3d>> &acc_vec,
                               vector<pair<double, Vec3d>> &gyr_vec){
    if(acc_buf.empty()){
        Warnv("GetIMUInterval | not receive imu!");
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
        Warnv("GetIMUInterval | wait for imu!");
        return false;
    }
    return true;
}

/**
 * 初始化IMU的位姿
 * @param acc_vec
 */
void Estimator::InitFirstIMUPose(vector<pair<double, Vec3d>> &acc_vec)
{
    Infov("InitfirstIMUPose | init first imu pose");
    is_init_first_pose = true;
    //计算平均加速度
    Vec3d aver_acc(0, 0, 0);
    int n = (int)acc_vec.size();
    for(auto & acc_pair : acc_vec)
        aver_acc = aver_acc + acc_pair.second;
    aver_acc = aver_acc / n;
    Debugv("InitfirstIMUPose | average acc {} {} {}", aver_acc.x(), aver_acc.y(), aver_acc.z());

    Mat3d R0 = Utility::g2R(aver_acc);
    double yaw = Utility::R2ypr(R0).x();
    R0 = Utility::ypr2R(Vec3d{-yaw, 0, 0}) * R0;
    body.Rs[0] = R0;
    cout << "init R0 " << endl << body.Rs[0] << endl;
    //body.Vs[0] = Vec3d(5, 0, 0);
}

/**
 * 处理IMU的测量值, 包括状态递推 和 预积分
 * @param t 输入的测量值的时刻
 * @param dt 输入的测量值的时刻到上一时刻的时间差
 * @param linear_acceleration 输入的加速度测量值
 * @param angular_velocity 输入的陀螺仪测量值
 */
void Estimator::ProcessIMU(double t, double dt, const Vec3d &linear_acceleration, const Vec3d &angular_velocity)
{
    //第一次进入的时候执行
    if (!first_imu){
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }
    //只有前10帧才会执行
    if (!pre_integrations[frame]){
        pre_integrations[frame] = new IntegrationBase{acc_0, gyr_0, body.Bas[frame], body.Bgs[frame]};
    }

    if (frame != 0){
        ///预积分
        pre_integrations[frame]->push_back(dt, linear_acceleration, angular_velocity);
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame].push_back(dt);
        linear_acceleration_buf[frame].push_back(linear_acceleration);
        angular_velocity_buf[frame].push_back(angular_velocity);

        ///状态递推 body.Rs,body.Ps,body.Vs
        int j = frame;
        Vec3d un_acc_0 = body.Rs[j] * (acc_0 - body.Bas[j]) - body.g;
        Vec3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - body.Bgs[j];
        body.Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vec3d un_acc_1 = body.Rs[j] * (linear_acceleration - body.Bas[j]) - body.g;
        Vec3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        body.Ps[j] += dt * body.Vs[j] + 0.5 * dt * dt * un_acc;
        body.Vs[j] += dt * un_acc;
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
    for (auto &lm : feat_manager.point_landmarks)
    {
        int imu_j = lm.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = lm.feature_id;
        for (auto &feat : lm.feats){
            imu_j++;
            Vec3d pts_j = feat.point;
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
    auto frame_it = all_image_frame.begin();
    map<int, Vec3d>::iterator it;
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++)
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == body.headers[i]){
            frame_it->second.is_key_frame = true;
            frame_it->second.R = Q[i].toRotationMatrix() * para::RIC[0].transpose();
            frame_it->second.T = T[i];
            i++;
            continue;
        }
        if((frame_it->first) > body.headers[i]){
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
        for (auto &id_pts : frame_it->second.points){
            int feature_id = id_pts.first;
            for (auto &i_p : id_pts.second){
                it = sfm_tracked_points.find(feature_id);
                if(it != sfm_tracked_points.end()){
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
        if(pts_3_vector.size() < 6){
            Debugv("pts_3_vector size {}",pts_3_vector.size());
            Debugv("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec,
                           t, 1)){
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
        frame_it->second.R = R_pnp * para::RIC[0].transpose();
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
    bool result = VisualIMUAlignment(all_image_frame, body.Bgs, body.g, x);
    if(!result){
        Debugv("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame; i++)
    {
        Mat3d Ri = all_image_frame[body.headers[i]].R;
        Vec3d Pi = all_image_frame[body.headers[i]].T;
        body.Ps[i] = Pi;
        body.Rs[i] = Ri;
        all_image_frame[body.headers[i]].is_key_frame = true;
    }

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= kWinSize; i++)
    {
        pre_integrations[i]->repropagate(Vec3d::Zero(), body.Bgs[i]);
    }
    for (int i = frame; i >= 0; i--)
        body.Ps[i] = s * body.Ps[i] - body.Rs[i] * para::TIC[0] - (s * body.Ps[0] - body.Rs[0] * para::TIC[0]);
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++)
    {
        if(frame_i->second.is_key_frame)
        {
            kv++;
            body.Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3);
        }
    }

    Mat3d R0 = Utility::g2R(body.g);
    double yaw = Utility::R2ypr(R0 * body.Rs[0]).x();
    R0 = Utility::ypr2R(Vec3d{-yaw, 0, 0}) * R0;
    body.g = R0 * body.g;
    //Mat3d rot_diff = R0 * body.Rs[0].transpose();
    Mat3d rot_diff = R0;
    for (int i = 0; i <= frame; i++)
    {
        body.Ps[i] = rot_diff * body.Ps[i];
        body.Rs[i] = rot_diff * body.Rs[i];
        body.Vs[i] = rot_diff * body.Vs[i];
    }
    Debugv("g0:{}",VecToStr(body.g));
    Debugv("my R0 :{}", VecToStr(Utility::R2ypr(body.Rs[0])));

    feat_manager.ClearDepth();
    feat_manager.TriangulatePoints();

    return true;
}

bool Estimator::RelativePose(Mat3d &relative_R, Vec3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < kWinSize; i++)
    {
        vector<pair<Vec3d, Vec3d>> corres;
        corres = feat_manager.GetCorresponding(i, kWinSize);
        if (corres.size() > 20)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (auto & corre : corres){
                Vector2d pts_0(corre.first(0), corre.first(1));
                Vector2d pts_1(corre.second(0), corre.second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T)){
                l = i;
                Debugv("Average_parallax {} choose l {} and newest frame to TriangulatePoint the whole structure",
                       average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}


/**
 * 将待优化变量写入double数组,以便ceres进行优化
 */
void Estimator::Vector2double()
{
    body.SetOptimizeParameters();

    VectorXd dep = feat_manager.GetDepthVector();
    int point_size = feat_manager.GetFeatureCount();
    for (int i = 0; i < point_size; i++){
        body.para_point_features[i][0] = dep(i);
    }

    if(cfg::slam == SLAM::kLine){
        MatrixXd line_orth = feat_manager.GetLineOrthVector(body.Ps, body.tic, body.ric);
        int line_size = feat_manager.GetLineFeatureCount();
        for (int i = 0; i < line_size; ++i) {
            body.para_line_features[i][0] = line_orth.row(i)[0];
            body.para_line_features[i][1] = line_orth.row(i)[1];
            body.para_line_features[i][2] = line_orth.row(i)[2];
            body.para_line_features[i][3] = line_orth.row(i)[3];
            if(i > kNumFeat)
                std::cerr << " 1000  1000 1000 1000 1000 \n\n";
        }
    }
}


void Estimator::Double2vector()
{
    Vec3d origin_R0 = Utility::R2ypr(body.Rs[0]);
    Vec3d origin_P0 = body.Ps[0];

    if (failure_occur){
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = false;
    }

    body.GetOptimizationParameters(origin_R0,origin_P0);

    VectorXd dep = feat_manager.GetDepthVector();
    int point_size = feat_manager.GetFeatureCount();
    for (int i = 0; i < point_size; i++)
        dep(i) = body.para_point_features[i][0];
    feat_manager.SetDepth(dep);

    if(cfg::slam == SLAM::kLine){
        int line_size = feat_manager.GetLineFeatureCount();
        MatrixXd orth_vec(line_size, 4);
        for (int i = 0; i < line_size; ++i) {
            orth_vec.row(i) = Vector4d(body.para_line_features[i][0],
                                       body.para_line_features[i][1],
                                       body.para_line_features[i][2],
                                       body.para_line_features[i][3]);
        }
        feat_manager.SetLineOrth(orth_vec, body.Ps, body.Rs, body.tic, body.ric);
    }


}


bool Estimator::FailureDetection()
{
    return false;
    if (feat_manager.last_track_num < 2){
        Infov(" little feature %d", feat_manager.last_track_num);
    }
    if (body.Bas[kWinSize].norm() > 2.5){
        Infov(" big IMU acc bias estimation %f", body.Bas[kWinSize].norm());
        return true;
    }
    if (body.Bgs[kWinSize].norm() > 1.0){
        Infov(" big IMU gyr bias estimation %f", body.Bgs[kWinSize].norm());
        return true;
    }
    /*
    if (tic(0) > 1)
    {
        Infov(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vec3d tmp_P = body.Ps[kWinSize];
    if ((tmp_P - last_P).norm() > 5){
        //Infov(" big translation");
        //return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1){
        //Infov(" big z translation");
        //return true;
    }
    Mat3d tmp_R = body.Rs[kWinSize];
    Mat3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50){
        Infov(" big delta_angle ");
    }
    return false;
}

/**
 * 执行滑动窗口
 */
void Estimator::SlideWindow()
{
    TicToc t_margin;
    if (margin_flag == MarginFlag::kMarginOld){
        double t_0 = body.headers[0];
        back_R0 = body.Rs[0];
        back_P0 = body.Ps[0];
        if (frame == kWinSize){
            ///将 1-10的变量移动到0-9
            for (int i = 0; i < kWinSize; i++){
                body.headers[i] = body.headers[i + 1];
                body.Rs[i].swap(body.Rs[i + 1]);
                body.Ps[i].swap(body.Ps[i + 1]);
                if(cfg::is_use_imu){
                    std::swap(pre_integrations[i], pre_integrations[i + 1]);
                    dt_buf[i].swap(dt_buf[i + 1]);
                    linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                    angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                    body.Vs[i].swap(body.Vs[i + 1]);
                    body.Bas[i].swap(body.Bas[i + 1]);
                    body.Bgs[i].swap(body.Bgs[i + 1]);
                }
            }
            body.headers[kWinSize] = body.headers[kWinSize - 1];
            body.Ps[kWinSize] = body.Ps[kWinSize - 1];
            body.Rs[kWinSize] = body.Rs[kWinSize - 1];

            if(cfg::is_use_imu){
                body.Vs[kWinSize] = body.Vs[kWinSize - 1];
                body.Bas[kWinSize] = body.Bas[kWinSize - 1];
                body.Bgs[kWinSize] = body.Bgs[kWinSize - 1];

                delete pre_integrations[kWinSize];
                pre_integrations[kWinSize] = new IntegrationBase{acc_0, gyr_0, body.Bas[kWinSize], body.Bgs[kWinSize]};
                dt_buf[kWinSize].clear();
                linear_acceleration_buf[kWinSize].clear();
                angular_velocity_buf[kWinSize].clear();
            }

            ///从地图中删除 t_0
            if (true || solver_flag == SolverFlag::kInitial){
                auto it_0 = all_image_frame.find(t_0);
                delete it_0->second.pre_integration;
                all_image_frame.erase(all_image_frame.begin(), it_0);
            }

            SlideWindowOld();

        }
    }
    else
    {
        if (frame == kWinSize){
            body.headers[frame - 1] = body.headers[frame];
            body.Ps[frame - 1] = body.Ps[frame];
            body.Rs[frame - 1] = body.Rs[frame];

            if(cfg::is_use_imu){
                for (unsigned int i = 0; i < dt_buf[frame].size(); i++){
                    double tmp_dt = dt_buf[frame][i];
                    Vec3d tmp_linear_acceleration = linear_acceleration_buf[frame][i];
                    Vec3d tmp_angular_velocity = angular_velocity_buf[frame][i];
                    pre_integrations[frame - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);
                    dt_buf[frame - 1].push_back(tmp_dt);
                    linear_acceleration_buf[frame - 1].push_back(tmp_linear_acceleration);
                    angular_velocity_buf[frame - 1].push_back(tmp_angular_velocity);
                }

                body.Vs[frame - 1] = body.Vs[frame];
                body.Bas[frame - 1] = body.Bas[frame];
                body.Bgs[frame - 1] = body.Bgs[frame];

                delete pre_integrations[kWinSize];
                pre_integrations[kWinSize] = new IntegrationBase{acc_0, gyr_0, body.Bas[kWinSize], body.Bgs[kWinSize]};

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
    feat_manager.RemoveFront(frame);
}

void Estimator::SlideWindowOld()
{
    sum_of_back++;

    bool shift_depth = solver_flag == SolverFlag::kNonLinear ? true : false;
    if (shift_depth){
        Mat3d R0, R1;
        Vec3d P0, P1;
        R0 = back_R0 * body.ric[0];
        R1 = body.Rs[0] * body.ric[0];
        P0 = back_P0 + back_R0 * body.tic[0];
        P1 = body.Ps[0] + body.Rs[0] * body.tic[0];
        feat_manager.RemoveBackShiftDepth(R0, P0, R1, P1);
    }
    else{
        feat_manager.RemoveBack();
    }
}




void Estimator::PredictPtsInNextFrame()
{
    //printf("predict pts in next frame\n");
    if(frame < 2)
        return;
    // predict next pose. Assume constant velocity motion
    auto curT = body.GetPoseInWorldFrame();
    auto prevT=body.GetPoseInWorldFrame(frame - 1);
    Eigen::Matrix4d  nextT = curT * (prevT.inverse() * curT);
    map<int, Vec3d> predictPts;

    for (auto &lm : feat_manager.point_landmarks){
        if(lm.depth > 0){
            int firstIndex = lm.start_frame;
            int lastIndex = lm.start_frame + lm.feats.size() - 1;
            //printf("cur frame index  %d last frame index %d\n", frame_count, lastIndex);
            if((int)lm.feats.size() >= 2 && lastIndex == frame){
                Vec3d pts_w = body.CamToWorld(lm.depth * lm.feats[0].point, firstIndex);
                Vec3d pts_local = nextT.block<3, 3>(0, 0).transpose() * (pts_w - nextT.block<3, 1>(0, 3));
                Vec3d pts_cam = body.ric[0].transpose() * (pts_local - body.tic[0]);
                int ptsIndex = lm.feature_id;
                predictPts[ptsIndex] = pts_cam;
            }
        }
    }
    //featureTracker->setPrediction(predictPts);
    //printf("e output %d predict pts\n",(int)predictPts.size());
}




/**
 * 根据输入最新的IMU测量值递推得到最新的PVQ
 * @param t
 * @param linear_acceleration
 * @param angular_velocity
 */
void Estimator::FastPredictIMU(double t, const Vec3d& linear_acceleration,const Vec3d& angular_velocity)
{
    double dt = t - latest_time;
    latest_time = t;
    Vec3d un_acc_0 = latest_Q * (latest_acc_0 - latest_Ba) - body.g;
    Vec3d un_gyr = 0.5 * (latest_gyr_0 + angular_velocity) - latest_Bg;
    latest_Q = latest_Q * Utility::deltaQ(un_gyr * dt);

    Vec3d un_acc_1 = latest_Q * (linear_acceleration - latest_Ba) - body.g;
    Vec3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    latest_P = latest_P + dt * latest_V + 0.5 * dt * dt * un_acc;
    latest_V = latest_V + dt * un_acc;

    latest_acc_0 = linear_acceleration;
    latest_gyr_0 = angular_velocity;
}


void Estimator::UpdateLatestStates(){
    propogate_mutex.lock();
    latest_time = body.headers[frame] + body.td;
    latest_P = body.Ps[frame];
    latest_Q = body.Rs[frame];
    latest_V = body.Vs[frame];
    latest_Ba = body.Bas[frame];
    latest_Bg = body.Bgs[frame];
    latest_acc_0 = acc_0;
    latest_gyr_0 = gyr_0;
    buf_mutex.lock();
    queue<pair<double, Vec3d>> tmp_accBuf = acc_buf;
    queue<pair<double, Vec3d>> tmp_gyrBuf = gyr_buf;
    buf_mutex.unlock();
    while(!tmp_accBuf.empty()){
        double t = tmp_accBuf.front().first;
        Vec3d acc = tmp_accBuf.front().second;
        Vec3d gyr = tmp_gyrBuf.front().second;
        FastPredictIMU(t, acc, gyr);
        tmp_accBuf.pop();
        tmp_gyrBuf.pop();
    }
    propogate_mutex.unlock();
}


void Estimator::InitEstimator(double header){

    ///初始化外参数
    if(cfg::is_estimate_ex == 2){
        if (frame != 0){
            Infov("calibrating extrinsic param, rotation movement is needed");
            auto cor = feat_manager.GetCorresponding(frame - 1, frame);
            Mat3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(cor, pre_integrations[frame]->delta_q, calib_ric)){
                Debugv("initial extrinsic rotation calib success");
                Debugv("initial extrinsic rotation:\n{}", EigenToStr(calib_ric));
                body.ric[0] = calib_ric;
                para::RIC[0] = calib_ric;
                cfg::is_estimate_ex = 1;
            }
        }
    }

    if (!cfg::is_stereo && cfg::is_use_imu){ // monocular + IMU initilization
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
    else if(cfg::is_stereo && cfg::is_use_imu){
        feat_manager.InitFramePoseByPnP(frame, body.Ps, body.Rs, body.tic, body.ric);
        feat_manager.TriangulatePoints();
        if (frame == kWinSize){
            map<double, ImageFrame>::iterator frame_it;
            int i = 0;
            for (frame_it = all_image_frame.begin(); frame_it != all_image_frame.end(); frame_it++){
                frame_it->second.R = body.Rs[i];
                frame_it->second.T = body.Ps[i];
                i++;
            }
            SolveGyroscopeBias(all_image_frame, body.Bgs);
            for (int j = 0; j <= kWinSize; j++)
                pre_integrations[j]->repropagate(Vec3d::Zero(), body.Bgs[j]);
            Optimization();
            UpdateLatestStates();
            solver_flag = SolverFlag::kNonLinear;
            SlideWindow();
            Infov("Initialization finish!");
        }
    }
    // stereo only initilization
    else if(cfg::is_stereo && !cfg::is_use_imu){
        feat_manager.InitFramePoseByPnP(frame, body.Ps, body.Rs, body.tic, body.ric);
        feat_manager.TriangulatePoints();
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
        body.frame=frame;
        int prev_frame = frame - 1;
        body.Ps[frame] = body.Ps[prev_frame];
        body.Vs[frame] = body.Vs[prev_frame];
        body.Rs[frame] = body.Rs[prev_frame];
        body.Bas[frame] = body.Bas[prev_frame];
        body.Bgs[frame] = body.Bgs[prev_frame];
    }
}


/**
 * VIO估计器的主函数
 * @param image
 * @param header
 */
void Estimator::ProcessImage(FrontendFeature &image, const double header){
    TicToc tt;

    Infov("processImage adding feature: points_size:{},lines_size:{}", image.features.points.size(), image.features.lines.size());

    ///添加背景特征点到管理器,并判断视差
    bool margin_flag_bool;
    if(cfg::slam == SLAM::kLine)
         margin_flag_bool = feat_manager.AddFeatureCheckParallax(frame, image.features, body.td);
    else
         margin_flag_bool = feat_manager.AddFeatureCheckParallax(frame, image.features.points, body.td);

    if (margin_flag_bool)
        margin_flag = MarginFlag::kMarginOld;
    else
        margin_flag = MarginFlag::kMarginSecondNew;

    Infov("processImage all feature: points_size:{},lines_size:{}",
          feat_manager.point_landmarks.size(), feat_manager.line_landmarks.size());
    Debugv("processImage margin_flag:{}", margin_flag == MarginFlag::kMarginSecondNew ? "kMarginSecondNew" : "kMarginOld");
    Debugv("processImage 地图中被观测4次以上的地点的数量: {}", feat_manager.GetFeatureCount());

    body.headers[frame] = header;

    ///创建帧,并设置该帧的预积分
    ImageFrame img_frame(image.features.points, header);
    img_frame.pre_integration = tmp_pre_integration;
    all_image_frame.insert({header, img_frame});
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, body.Bas[frame], body.Bgs[frame]};

    Infov("processImage AddFeatureCheckParallax && create ImageFrame:{} ms",tt.TocThenTic());

    ///VINS的初始化
    if (solver_flag == SolverFlag::kInitial){
        InitEstimator(header);
        Infov("processImage InitEstimator:{} ms",tt.TocThenTic());
        return ;
    }


    ///若没有IMU,则需要根据PnP得到当前帧的位姿
    if(!cfg::is_use_imu)
        feat_manager.InitFramePoseByPnP(frame, body.Ps, body.Rs, body.tic, body.ric);

    ///三角化背景特征点
    feat_manager.TriangulatePoints();

    ///三角化线特征
    if(cfg::slam == SLAM::kLine){
        //if(cfg::is_stereo){
        //    feat_manager.TriangulateLineStereo(cam1->baseline);
        //}
        //else{
            feat_manager.TriangulateLineMono();
        //}
    }

    if(para::is_print_detail && cfg::slam == SLAM::kLine){
        PrintLineInfo(feat_manager);
    }
    Infov("processImage background Triangulate:{} ms",tt.TocThenTic());
    Debugv("--开始处理动态物体--");

    if(cfg::slam == SLAM::kDynamic){
        ///添加动态特征点,并创建物体
        im.PushBack(frame, image.instances);

        im.SetOutputInstInfo();//将输出实例的速度信息
        ///动态物体的位姿递推
        im.PropagatePose();
        ///动态特征点的三角化
        im.Triangulate();
        ///若动态物体未初始化, 则进行初始化
        im.InitialInstance();
        ///初始化速度
        im.InitialInstanceVelocity();
        ///根据重投影误差和对极几何判断物体是运动的还是静态的
        im.SetDynamicOrStatic();

        if(para::is_print_detail){
            PrintFeaturesInfo(im, true, true);
        }

        ///单独优化动态物体
        if(para::is_print_detail){
            PrintInstancePoseInfo(im, true);
        }
        im.Optimization();

        if(para::is_print_detail){
            PrintInstancePoseInfo(im, true);
        }

        im.OutliersRejection();

        Infov("processImage dynamic Optimization:{} ms",tt.TocThenTic());

        Debugv("--完成处理动态物体--");
    }

    ///优化直线
    if(cfg::slam == SLAM::kLine){
        OptimizationWithLine();
    }

    ///VIO窗口的非线性优化
    Optimization();

    Infov("processImage Optimization:{} ms", tt.TocThenTic());

    ///外点剔除
    set<int> removeIndex;
    OutliersRejection(removeIndex, feat_manager.point_landmarks);

    feat_manager.RemoveOutlier(removeIndex);
    feat_manager.RemoveLineOutlier();


    if (FailureDetection()){
        Warnv("failure detection!");
        failure_occur = true;
        ClearState();
        SetParameter();
        Warnv("system reboot!");
        return;
    }

    Infov("processImage Outliebody.RsRejection:{} ms", tt.TocThenTic());

    ///动态物体的滑动窗口
    if(cfg::slam == SLAM::kDynamic){
        im.ManageTriangulatePoint();
        Debugv("finish ManageTriangulatePoint()");
        //insts_manager.SetWindowPose();
        im.SlideWindow(margin_flag);
    }

    /// 滑动窗口
    SlideWindow();

    ///动态物体的外点剔除
    if(cfg::slam == SLAM::kDynamic){
        im.OutliersRejection();

         im.DeleteBadLandmarks();
    }

    Infov("processImage SlideWindow:{} ms", tt.TocThenTic());

    feat_manager.RemoveFailures();

    /// prepare output of VINS
    key_poses.clear();
    for (int i = 0; i <= kWinSize; i++)
        key_poses.push_back(body.Ps[i]);

    last_R = body.Rs[kWinSize];
    last_P = body.Ps[kWinSize];
    last_R0 = body.Rs[0];
    last_P0 = body.Ps[0];
    UpdateLatestStates();

    if(cfg::slam == SLAM::kDynamic){
        im.SetInstanceCurrentPoint3d();
    }

}


/**
 * 滑动窗口状态估计的入口函数
 */
void Estimator::ProcessMeasurements(){
    int cnt=0;
    double time_sum=0;
    TicToc tt,t_all;

    while (cfg::ok.load(std::memory_order_seq_cst)){
        if(feature_queue.empty()){
            std::this_thread::sleep_for(2ms);
            continue;
        }
        ///获取VIO的处理数据
        auto front_time=feature_queue.front_time();
        if(!front_time)
            continue;
        cur_time = *front_time + body.td;
        if(cfg::is_use_imu && !IMUAvailable(cur_time)){
            std::cerr<<"wait for imu ..."<<endl;
            std::this_thread::sleep_for(5ms);
            continue;
        }

        tt.Tic();
        t_all.Tic();

        ///获取上一帧时刻到当前时刻的IMU测量值
        vector<pair<double, Vec3d>> acc_vec, gyr_vec;
        if(cfg::is_use_imu){
            std::unique_lock<std::mutex> lock(buf_mutex);
            GetIMUInterval(prev_time, cur_time, acc_vec, gyr_vec);
        }
        //获取前端得到的特征
        feature_frame = *(feature_queue.request());

        Infov("\n \n \n \n");
        Warnv("----------Time:{} seq:{} ----------", std::to_string(*front_time),feature_frame.seq_id);
        Infov("ProcessMeasurements::GetIMUInterval:{} ms",tt.TocThenTic());

        body.seq_id = feature_frame.seq_id;
        body.frame_time = feature_frame.time;

        ///IMU预积分 和 状态递推
        if(cfg::is_use_imu){
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

        Infov("ProcessMeasurements::ProcessIMU:{} ms",tt.TocThenTic());
        Debugv("solver_flag:{}", solver_flag == SolverFlag::kInitial ? "INITIAL" : "NO-LINEAR");

        process_mutex.lock();

        ///进入主函数
        ProcessImage(feature_frame ,feature_frame.time);

        prev_time = cur_time;

        double time_cost=t_all.Toc();
        time_sum+=time_cost;
        cnt++;

        Infov("ProcessMeasurements::ProcessImage:{} ms",tt.TocThenTic());

        ///输出
        SetOutputEgoInfo(body.Rs[kWinSize], body.Ps[kWinSize], body.ric[0], body.tic[0]);

        string log_time="headers: ";
        for(int i=0;i<=kWinSize;++i){
            log_time += fmt::format("{}:{} ",i,body.headers[i]);
        }
        Debugv(log_time);

        Publisher::PrintStatistics(0);

        std_msgs::Header header;
        header.frame_id = "world";
        header.stamp = ros::Time(feature_frame.time);

        if(cfg::slam == SLAM::kDynamic){
            //printInstanceData(*this);
            Publisher::PubInstances(header);
            cv::Mat img_topview = DrawTopView(im);
            ImagePublisher::Pub(img_topview,"top_view");
        }

        Publisher::PubOdometry(header);
        Publisher::PubKeyPoses(header);
        Publisher::PubCameraPose(header);
        Publisher::PubPointCloud(header);
        Publisher::PubKeyframe();
        Publisher::PubTF(header);

        if(cfg::slam == SLAM::kLine){
            Publisher::PubLines(header);
        }

        //PubPredictBox3D(*this,feature_frame.boxes);

        process_mutex.unlock();

        ///轨迹保存
        SaveBodyTrajectory(header);

        //保存所有物体在当前帧的位姿
        if(cfg::slam == SLAM::kDynamic){
            SaveInstancesTrajectory(im);
        }

        Infov("ProcessMeasurements::Output:{} ms",tt.TocThenTic());

        static unsigned int estimator_cnt=0;
        auto output_msg=fmt::format("cnt:{} ___________estimator process time: {} ms_____________\n",
                                    cnt,t_all.TocThenTic());
        Infov(output_msg);
        cout<<output_msg<<endl;
    }

    Infov("VIO Avg cost:{} ms",time_sum/cnt);
    Warnv("ProcessMeasurements 线程退出");

}




}

