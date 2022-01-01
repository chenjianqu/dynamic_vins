//
// Created by chen on 2021/11/30.
//

#include <vector>
#include <tuple>

#include "TrackerManager.h"
#include "Hungarian.h"

/**
 *
 * @param metric
 * @param unmatched_trks
 * @param unmatched_dets
 * @param matched
 */
void associate_detections_to_trackers_idx(const DistanceMetricFunc &metric,
                                          std::vector<int> &unmatched_trks,
                                          std::vector<int> &unmatched_dets,
                                          std::vector<std::tuple<int, int>> &matched
                                          ) {

    //获得两两box之间的距离
    auto dist = metric(unmatched_trks, unmatched_dets);
    auto dist_a = dist.accessor<float, 2>();
    //将计算结果放到二维数组中
    auto dist_v = std::vector<std::vector<double>>(dist.size(0), std::vector<double>(dist.size(1)));
    for (int64_t i = 0; i < dist.size(0); ++i) {
        for (int64_t j = 0; j < dist.size(1); ++j) {
            dist_v[i][j] = dist_a[i][j];
        }
    }

    ///调用匈牙利算法进行分配
    std::vector<int> assignment;
    HungarianAlgorithm().Solve(dist_v, assignment);

    // filter out matched with low IOU
    for (size_t i = 0; i < assignment.size(); ++i) {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (dist_v[i][assignment[i]] > INVALID_DIST / 10) {
            assignment[i] = -1;
        }
        else {
            matched.emplace_back(std::make_tuple(unmatched_trks[i], unmatched_dets[assignment[i]]));
        }
    }


    for (size_t i = 0; i < assignment.size(); ++i) {
        if (assignment[i] != -1) {
            unmatched_trks[i] = -1;
        }
    }
    unmatched_trks.erase(
            remove_if(unmatched_trks.begin(), unmatched_trks.end(),[](int i) { return i == -1; }),
            unmatched_trks.end());


    std::sort(assignment.begin(), assignment.end());

    //计算assignment和unmatched_dets的交集，表示剩余未匹配的新检测
    std::vector<int> unmatched_dets_new;
    std::set_difference(unmatched_dets.begin(), unmatched_dets.end(),
                        assignment.begin(), assignment.end(),
                        std::inserter(unmatched_dets_new, unmatched_dets_new.begin()));

    unmatched_dets = move(unmatched_dets_new);
}


