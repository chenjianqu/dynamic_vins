/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <set>
#include <algorithm>

#include "tracker_manager.h"
#include "hungarian.h"

namespace dynamic_vins{\


void AssociateDetectionsToTrackersIdx(const DistanceMetricFunc &metric,
                                      vector<int> &unmatched_trks,
                                      vector<int> &unmatched_dets,
                                      vector<tuple<int, int>> &matched) {
    ///获取外观代价矩阵
    auto dist = metric(unmatched_trks, unmatched_dets);

    ///将tensor转移到vector上
    auto dist_a = dist.accessor<float, 2>();
    auto dist_v = vector<vector<double>>(dist.size(0), vector<double>(dist.size(1)));
    for (size_t i = 0; i < dist.size(0); ++i) {
        for (size_t j = 0; j < dist.size(1); ++j) {
            dist_v[i][j] = dist_a[i][j];
        }
    }

    ///匈牙利算法分配
    vector<int> assignment;//assignment[i]表示第i个轨迹对应的检测框的id
    HungarianAlgorithm().Solve(dist_v, assignment);

    ///根据dist_v去掉一些匹配
    for (size_t i = 0; i < assignment.size(); ++i) {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (dist_v[i][assignment[i]] > INVALID_DIST / 10) {
            assignment[i] = -1;
        }
        else {
            matched.emplace_back(unmatched_trks[i], unmatched_dets[assignment[i]]);
        }
    }

    ///将unmatched_trks已经匹配的轨迹设置为-1，并在unmatched_trks删除这些已经匹配的轨迹，即更新unmatched_trks
    for (size_t i = 0; i < assignment.size(); ++i) {
        if (assignment[i] != -1) {
            unmatched_trks[i] = -1;
        }
    }
    unmatched_trks.erase(remove_if(unmatched_trks.begin(), unmatched_trks.end(),[](int i) { return i == -1; }),unmatched_trks.end());


    ///求unmatched_dets和assignment的差集，得到未匹配边界框ID集合，更新unmatched_dets
    vector<int> unmatched_dets_new;
    sort(assignment.begin(), assignment.end());
    std::set_difference(unmatched_dets.begin(), unmatched_dets.end(),
                   assignment.begin(), assignment.end(),
                   inserter(unmatched_dets_new, unmatched_dets_new.begin()));
    //更新unmatched_dets
    unmatched_dets = std::move(unmatched_dets_new);
}












}
