/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "marginalization_factor.h"
#include "parameters.h"

namespace dynamic_vins{\

/**
 * 计算残差和雅可比
 */
void ResidualBlockInfo::Evaluate()
{
    residuals.resize(cost_function->num_residuals());//设置残差的维度
    std::vector<int> block_sizes = cost_function->parameter_block_sizes();//cost_function连接的各个参数块的大小
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++){
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

/*    std::vector<int> tmp_idx(block_sizes.size());
    Eigen::MatrixXd tmp(dim, dim);
    for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    {
        int size_i = localSize(block_sizes[i]);
        Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
        for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
        {
            int size_j = localSize(block_sizes[j]);
            Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
            tmp_idx[j] = sub_idx;
            tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    std::cout << saes.eigenvalues() << std::endl;
    ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);*/

    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;
        double sq_norm, rho[3];
        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0)){
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else{
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++){
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}



MarginalizationInfo::~MarginalizationInfo()
{
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
        delete it->second;

    for (int i = 0; i < (int)factors.size(); i++)
    {
        delete[] factors[i]->raw_jacobians;
        delete factors[i]->cost_function;
        delete factors[i];
    }
}

/**
 * 添加因子
 * @param residual_block_info
 */
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    factors.emplace_back(residual_block_info);

    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    for (int i = 0; i < residual_block_info->parameter_blocks.size(); i++){
        double *addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }
    ///添加待优化变量
    for (int drop_id : residual_block_info->drop_set){
        double *addr = parameter_blocks[drop_id];
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

/**
 * 计算所有残差项的雅可比和残差,并关联的参数复制到 parameter_block_data
 */
void MarginalizationInfo::preMarginalize()
{
    for (auto factor : factors){
        ///计算雅可比和残差
        factor->Evaluate();

        std::vector<int> block_sizes = factor->cost_function->parameter_block_sizes();
        ///将该残差关联的参数复制到 parameter_block_data
        for (int i = 0; i < block_sizes.size(); i++){
            long addr = reinterpret_cast<long>(factor->parameter_blocks[i]);
            int size = block_sizes[i];
            if (parameter_block_data.find(addr) == parameter_block_data.end()){
                auto *data = new double[size];
                memcpy(data, factor->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;
            }
        }
    }
}

/**
 * 用于多线程计算雅可比
 */
struct ThreadsStruct{
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

/**
 * 构建 A 矩阵
 * @param threads_struct
 * @return
 */
void* ThreadsConstructA(void* threads_struct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threads_struct);
    ///构造Hessian矩阵 A 和 b
    for (auto factor : p->sub_factors){
        for (int i = 0; i < factor->parameter_blocks.size(); i++){
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(factor->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(factor->parameter_blocks[i])];
            if (size_i == 7) size_i = 6;
            Eigen::MatrixXd jacobian_i = factor->jacobians[i].leftCols(size_i);
            for (int j = i; j < factor->parameter_blocks.size(); j++){
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(factor->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(factor->parameter_blocks[j])];
                if (size_j == 7) size_j = 6;
                Eigen::MatrixXd jacobian_j = factor->jacobians[j].leftCols(size_j);
                if (i == j){
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                }
                else{
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * factor->residuals;
        }
    }
    return threads_struct;
}


/**
 * 执行边缘化
 */
void MarginalizationInfo::marginalize()
{
    ///设置每个参数在矩阵中的索引
    //首先设置待marg变量的索引
    int pos = 0;
    for (auto &pair : parameter_block_idx){
        pair.second = pos;
        pos += localSize(parameter_block_size[pair.first]);
    }
    m = pos;
    //设置保留变量的索引
    for (const auto &pair : parameter_block_size){
        if (parameter_block_idx.find(pair.first) == parameter_block_idx.end()){
            parameter_block_idx[pair.first] = pos;
            pos += localSize(pair.second);
        }
    }
    n = pos - m;

    if(m == 0){
        valid = false;
        printf("unstable tracking...\n");
        Errorv("marginalize | unstable tracking...");
        return;
    }


    ///多线程计算A和b

    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */

    //multi thread
    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threads_struct[NUM_THREADS];

    int i = 0;
    for (auto factor : factors){
        threads_struct[i].sub_factors.push_back(factor);
        i++;
        i = i % NUM_THREADS;
    }

    for (int i = 0; i < NUM_THREADS; i++){
        TicToc zero_matrix;
        threads_struct[i].A = Eigen::MatrixXd::Zero(pos, pos);
        threads_struct[i].b = Eigen::VectorXd::Zero(pos);
        threads_struct[i].parameter_block_size = parameter_block_size;
        threads_struct[i].parameter_block_idx = parameter_block_idx;
        int ret = pthread_create( &tids[i], nullptr, ThreadsConstructA ,(void*)&(threads_struct[i]));
        if (ret != 0){
            ROS_WARN("pthread_create error");
            ROS_BREAK();
        }
    }
    //合并Hessian矩阵
    for( int i = NUM_THREADS - 1; i >= 0; i--){
        pthread_join( tids[i], nullptr );
        A += threads_struct[i].A;
        b += threads_struct[i].b;
    }

    ///执行边缘化
    Eigen::MatrixXd Amm = 0.5 * (  A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose() );
    //求Amm的逆矩阵
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() *
            Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
            saes.eigenvectors().transpose();

    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    ///从边缘化后的Hessian矩阵中反解出雅可比和残差项
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
}




std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx){
        if (it.second >= m){
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            keep_block_data.push_back(parameter_block_data[it.first]);
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info)
: marg_info(_marginalization_info)
{
    int cnt = 0;
    for (auto it : marg_info->keep_block_size){
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    set_num_residuals(marg_info->n);
}

bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    int n = marg_info->n;
    int m = marg_info->m;
    Eigen::VectorXd dx(n);
    for (int i = 0; i < static_cast<int>(marg_info->keep_block_size.size()); i++){
        int size = marg_info->keep_block_size[i];
        int idx = marg_info->keep_block_idx[i] - m;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marg_info->keep_block_data[i], size);
        if (size != 7){
            dx.segment(idx, size) = x - x0;
        }
        else{
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(
                    Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() *
                    Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() *
                Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(
                        Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() *
                        Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }

    Eigen::Map<Eigen::VectorXd>(residuals, n) = marg_info->linearized_residuals + marg_info->linearized_jacobians * dx;
    if (jacobians){
        for (int i = 0; i < static_cast<int>(marg_info->keep_block_size.size()); i++){
            if (jacobians[i]){
                int size = marg_info->keep_block_size[i], local_size = marg_info->localSize(size);
                int idx = marg_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marg_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}

}
