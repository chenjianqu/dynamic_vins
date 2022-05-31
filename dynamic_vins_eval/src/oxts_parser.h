//
// Created by chen on 2022/5/20.
//

#ifndef DYNAMIC_VINS_EVAL_OXTS_PARSER_H
#define DYNAMIC_VINS_EVAL_OXTS_PARSER_H

#include <string>
#include <vector>
#include <Eigen/Dense>


void ParseOxts(std::vector<Eigen::Matrix4d> &pose,const std::string &data_path);


#endif //DYNAMIC_VINS_EVAL_OXTS_PARSER_H
