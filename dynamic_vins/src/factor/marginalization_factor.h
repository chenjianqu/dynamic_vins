/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "utility/utility.h"
#include "utils.h"

namespace dynamic_vins{\


const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function,
                      std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
                      : cost_function(_cost_function), loss_function(_loss_function),
                      parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    static int localSize(int size){
        return size == 7 ? 6 : size;
    }

    ceres::CostFunction *cost_function;//残差连接
    ceres::LossFunction *loss_function;//核函数
    std::vector<double *> parameter_blocks;//优化变量
    std::vector<int> drop_set;//待marg变量的id(在parameter_blocks中的索引)

    double **raw_jacobians;//雅可比
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;//残差
};



class MarginalizationInfo
{
  public:
    MarginalizationInfo(){
        valid = true;
    };

    ~MarginalizationInfo();

    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    static int localSize(int size) {
        return size == 7 ? 6 : size;
    }

    static int globalSize(int size) {
        return size == 6 ? 7 : size;
    }

    std::vector<ResidualBlockInfo *> factors;//保存要边缘化的相关残差信息

    int m; //要marg掉变量的维数
    int n; //保留下来的变量维数
    int sum_block_size;

    std::unordered_map<long, int> parameter_block_size; //global size,优化变量的内存地址以及变量的global size
    std::unordered_map<long, int> parameter_block_idx; //所有优化变量在矩阵中的索引,矩阵的前面m个维度是待优化变量,后面的n维度变量是保留的变量
    std::unordered_map<long, double *> parameter_block_data;//优化变量的地址和数据

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
    bool valid;

};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marg_info;
};

}
