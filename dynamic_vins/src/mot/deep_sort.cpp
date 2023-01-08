/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include <algorithm>

#include "deep_sort.h"
#include "extractor.h"
#include "tracker_manager.h"
#include "mot_parameter.h"

namespace dynamic_vins{\

using namespace std;
using namespace cv;


/**
 * 两个巨型之间的IoU
 * @param bb_test
 * @param bb_gt
 * @return
 */
float RectIou(const Rect2f &bb_test, const Rect2f &bb_gt) {
    auto in = (bb_test & bb_gt).area();
    auto un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return in / un;
}


/**
 * 计算滤波器轨迹边界框和检测框之间的IoU分数 （1-iou）
 * @param dets
 * @param trks
 * @return 返回代价矩阵A，其中A[i,j]表示第i个轨迹和第j个检测框之间的iou分数，分数越小重合度越高
 */
torch::Tensor CalIouDist(const vector<Rect2f> &dets, const vector<Rect2f> &trks) {
    auto trk_num = trks.size();
    auto det_num = dets.size();

    auto dist = torch::empty({int64_t(trk_num), int64_t(det_num)});

    for (size_t i = 0; i < trk_num; i++){ // compute iou matrix as a distance matrix
        for (size_t j = 0; j < det_num; j++) {
            dist[i][j] = 1 - RectIou(trks[i], dets[j]);
        }
    }

    return dist;
}



DeepSORT::DeepSORT(const string& config_path,const array<int64_t, 2> &dim)
: extractor(make_unique<Extractor>()),
manager(make_unique<TrackerManager<TrackData>>(data, dim)),
feat_metric(make_unique<FeatureMetric<TrackData>>(data))
{
    mot_para::SetParameters(config_path);
}



vector<Box2D::Ptr> DeepSORT::update(const std::vector<Box2D::Ptr> &detections, cv::Mat ori_img) {
    ///卡尔曼预测
    manager->predict();
    manager->remove_nan();

    ///用于计算外观距离的函数
    DistanceMetricFunc confirmed_metric = [this, &detections, &ori_img](
            const std::vector<int> &trk_ids,//卡尔曼滤波的轨迹
            const std::vector<int> &det_ids) { //检测框的ID
        //卡尔曼滤波器中的边界框
        vector<cv::Rect2f> trks;
        for (auto t : trk_ids) {
            trks.push_back(data[t].kalman.rect());
        }
        //检测得到的边界框和ROI
        vector<cv::Mat> boxes;
        vector<cv::Rect2f> dets;
        for (auto d:det_ids) {
            dets.push_back(detections[d]->rect);
            boxes.push_back(ori_img(detections[d]->rect));
        }
        ///获得滤波器边界框和检测边界框之间两两的IoU代价
        auto iou_mat = CalIouDist(dets, trks);

        ///获得滤波器边界框和检测边界框之间两两的外观代价
        auto feat_mat = feat_metric->distance(extractor->extract(boxes), trk_ids);
        ///根据IoU和外观距离过滤掉某些值
        feat_mat.masked_fill_((iou_mat > 0.8f).__ior__(feat_mat > 0.2f), INVALID_DIST);

        return feat_mat;
    };

    ///计算IoU代价的函数
    DistanceMetricFunc unconfirmed_metric = [this, &detections](const std::vector<int> &trk_ids, const std::vector<int> &det_ids) {
        vector<cv::Rect2f> trks;
        for (auto t : trk_ids) {
            trks.push_back(data[t].kalman.rect());
        }
        vector<cv::Rect2f> dets;
        for (auto &d:det_ids) {
            dets.push_back(detections[d]->rect);
        }
        ///计算IoU代价
        auto iou_mat = CalIouDist(dets, trks);
        iou_mat.masked_fill_(iou_mat > 0.7f, INVALID_DIST);

        return iou_mat;
    };


    ///执行更新
    auto matched = manager->update(detections,confirmed_metric,unconfirmed_metric);

    ///更新视觉特征
    vector<cv::Mat> boxes;//新的ROI
    vector<int> targets;//轨迹的ID
    for (auto[x, y]:matched) {
        targets.emplace_back(x);
        boxes.emplace_back(ori_img(detections[y]->rect));
    }
    feat_metric->update(extractor->extract(boxes), targets);

    manager->remove_deleted();

    ///输出结果
    //return manager->visible_tracks();
    return manager->visible_tracks_info();
}


}