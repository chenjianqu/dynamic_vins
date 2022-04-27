//
// Created by chen on 2021/11/30.
//

#include "DeepSORT.h"
#include "track.h"
#include "utils/parameters.h"
#include "utils/def.h"



/**
 * 计算当前帧检测到box与跟踪的box的IOU
 * @param dets 当前检测的box
 * @param trks 正在跟踪的box
 * @return
 */
torch::Tensor iou_dist(const std::vector<cv::Rect2f> &dets, const std::vector<cv::Rect2f> &trks) {
    auto trk_num = trks.size();
    auto det_num = dets.size();
    auto dist = torch::empty({int64_t(trk_num), int64_t(det_num)});
    for (int64_t i = 0; i < trk_num; i++){ // compute iou matrix as a distance matrix
        for (int64_t j = 0; j < det_num; j++){
            dist[i][j] = 1 - getBoxIoU(trks[i], dets[j]);
        }
    }
    return dist;
}


DeepSORT::DeepSORT(const std::array<int64_t, 2> &dim){
    vioLogger->info("start init DeepSORT");
    extractor = std::make_unique<Extractor>(Config::EXTRACTOR_MODEL_PATH);
    featureMetric = std::make_unique<FeatureMetric<InstanceTrackData>>(data);
    trackerManager = std::make_unique<TrackerManager<InstanceTrackData>>(data, dim);
    vioLogger->info("init DeepSORT finished");

}



std::vector<InstInfo> DeepSORT::update( std::vector<InstInfo> &detections, cv::Mat &ori_img)
{
    ///卡尔曼滤波预测跟踪物体的当前位置
    trackerManager->predict();
    trackerManager->remove_nan();

    /**
     * 计算comfirmed状态的box与检测的box之间的相似度，包括IOU和特征相似度
     * 返回两两box之间的特征距离
     */
    auto confirmed_metric=[this, &detections, &ori_img](const std::vector<int> &trk_ids, const std::vector<int> &det_ids) {
        vector<cv::Rect2f> trks;
        for (auto t : trk_ids) {
            trks.push_back(data[t].kalman.rect());
        }
        vector<cv::Mat> boxes;
        vector<cv::Rect2f> dets;
        for (auto d:det_ids) {
            dets.push_back(detections[d].rect);
            boxes.push_back(ori_img(detections[d].rect));
        }
        //计算两两box之间的IOU距离
        auto iou_mat = iou_dist(dets, trks);
        //计算两两box之间的特征距离
        auto feats=extractor->extract(boxes);
        auto feat_mat = featureMetric->distance(feats, trk_ids);

        //根据IOU距离和特征距离过滤掉不匹配的特征距离
        auto mask=(iou_mat > 0.8f).__ior__(feat_mat > 0.2f);
        feat_mat.masked_fill_(mask, INVALID_DIST);
        return feat_mat;
    };

    /**
     * 计算两个box之间的IOU距离
     */
    auto unconfirmed_metric=[this, &detections](const std::vector<int> &trk_ids, const std::vector<int> &det_ids) {
        vector<cv::Rect2f> trks;
        for (auto t : trk_ids) {
            trks.push_back(data[t].kalman.rect());
        }
        vector<cv::Rect2f> dets;
        for (auto &d:det_ids) {
            dets.push_back(detections[d].rect);
        }
        auto iou_mat = iou_dist(dets, trks);
        iou_mat.masked_fill_(iou_mat > 0.7f, INVALID_DIST);
        return iou_mat;
    };

    auto matched = trackerManager->update(detections,confirmed_metric,unconfirmed_metric);

    //将新检测的特征添加到特征管理器中
    vector<cv::Mat> boxes;
    vector<int> targets;
    for (auto[x, y] : matched) {
        targets.emplace_back(x);
        boxes.emplace_back(ori_img(detections[y].rect));
    }
    featureMetric->update(extractor->extract(boxes), targets);

    trackerManager->remove_deleted();

    //return trackerManager->visible_tracks();
    return trackerManager->visible_tracks_info();

}


