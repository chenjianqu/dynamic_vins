/*******************************************************
 * Copyright (C) 2022, Chen Jianqu, Shanghai University
 *
 * This file is part of dynamic_vins.
 * Github:https://github.com/chenjianqu/dynamic_vins
 *
 * Licensed under the MIT License;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "box2d.h"

#include "utils/dataset/coco_utils.h"
#include "utils/dataset/kitti_utils.h"

namespace dynamic_vins{\


/**
 * 根据实例分割的mask构建box2d
 * @param seg_label 实例分割mask,大小:[num_mask, cols, rows]
 * @param cate_label 类别标签:[num_mask]
 * @param cate_score 类别分数
 * @return
 */
vector<Box2D::Ptr> Box2D::BuildBoxes2D(torch::Tensor &seg_label,torch::Tensor &cate_label,torch::Tensor &cate_score)
{
    vector<Box2D::Ptr> insts;

    ///根据mask计算包围框
    for(int i=0;i<seg_label.sizes()[0];++i){
        auto nz=seg_label[i].nonzero();
        auto max_xy =std::get<0>( torch::max(nz,0) );
        auto min_xy =std::get<0>( torch::min(nz,0) );

        Box2D::Ptr inst = std::make_shared<Box2D>();
        inst->id = i;

        int coco_id = cate_label[i].item().toInt();
        string coco_name = coco::CocoLabel[coco_id];
        if(auto it=coco::CocoToKitti.find(coco_name);it!=coco::CocoToKitti.end()){
            string kitti_name = *(it->second.begin());
            int kitti_id = kitti::GetKittiLabelIndex(kitti_name);
            inst->class_id =kitti_id;
            inst->class_name = kitti_name;
        }
        else{
            inst->class_id =coco_id;
            inst->class_name = coco_name;
        }

        inst->max_pt.x = max_xy[1].item().toInt();
        inst->max_pt.y = max_xy[0].item().toInt();
        inst->min_pt.x = min_xy[1].item().toInt();
        inst->min_pt.y = min_xy[0].item().toInt();
        inst->rect = cv::Rect2f(inst->min_pt,inst->max_pt);

        inst->score = cate_score[i].item().toFloat();

        insts.push_back(inst);
    }

    return insts;
}


/**
 * 计算两个box之间的IOU
 * @param bb_test
 * @param bb_gt
 * @return
 */
float Box2D::IoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt){
    auto in = (bb_test & bb_gt).area();
    auto un = bb_test.area() + bb_gt.area() - in;
    if (un <  DBL_EPSILON)
        return 0;
    return in / un;
}



}
