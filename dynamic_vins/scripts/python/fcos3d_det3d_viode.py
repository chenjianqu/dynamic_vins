# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)
import re
from copy import deepcopy
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                          DepthInstance3DBoxes, LiDARInstance3DBoxes,
                          show_multi_modality_result, show_result,
                          show_seg_result)
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger



def detect_kitti_folder():
    parser = ArgumentParser()
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()


    config = "configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py"
    checkpoint = "checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"

    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint, device=args.device)
    # test a single image
    #result, data = inference_mono_3d_detector(model, image, ann)

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)

    #ann = "demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525_mono3d.coco.json"
    # get data info containing calib
    #data_infos = mmcv.load(ann)

    sequence_name="day_03_high"
    sequence_path = "/home/chen/datasets/VIODE/cam0/"+sequence_name
    result_path = "/home/chen/datasets/VIODE/det3d_cam0/"+sequence_name
    cam_instrinsic = [[376.0,0,376.0],[0,376.0,240.0],[0,0,1]]

    names = os.listdir(sequence_path)
    for name in names:
        image = os.path.join(sequence_path,name)
        print(image)

        img_info = dict()
        img_info["file_name"]=name
        img_info["cam_intrinsic"]=cam_instrinsic

        data = dict(
            img_prefix=osp.dirname(image),
            img_info=dict(filename=osp.basename(image)),
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
            img_fields=[],
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[])

        # camera points to image conversion
        if box_mode_3d == Box3DMode.CAM:
            data['img_info'].update(dict(cam_intrinsic=img_info['cam_intrinsic']))

        data = test_pipeline(data)

        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device.index])[0]
        else:
            # this is a workaround to avoid the bug of MMDataParallel
            data['img_metas'] = data['img_metas'][0].data
            data['img'] = data['img'][0].data

        # forward the model
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        box_3d = result[0]['img_bbox']['boxes_3d'].tensor.numpy() #200x9
        scores_3d = result[0]['img_bbox']['scores_3d'].numpy() #200x1
        labels_3d = result[0]['img_bbox']['labels_3d'].numpy() #200x1
        attrs_3d = result[0]['img_bbox']['attrs_3d'].numpy() #200x1

        number = box_3d.shape[0]

        stem, suffix = os.path.splitext(name)
        write_path = os.path.join(result_path,stem+".txt")
        with open(write_path, 'w') as f:
            for i in range(number):
                s=str()
                s = str(labels_3d[i])+" "+str(attrs_3d[i])+" "+str(scores_3d[i])
                box_s = ""
                for j in range(box_3d[i].shape[0]):
                    box_s += " " + str(box_3d[i,j])
                f.write(s+box_s+"\n")



    # show the results
    # show_result_meshlab(
    #     data,
    #     result,
    #     args.out_dir,
    #     args.score_thr,
    #     show=args.show,
    #     snapshot=args.snapshot,
    #     task='mono-det')


def detect_kitti_single():
    parser = ArgumentParser()
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    config = "configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py"
    checkpoint = "checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth"

    # build the model from a config file and a checkpoint file
    model = init_model(config, checkpoint, device=args.device)
    # test a single image
    # result, data = inference_mono_3d_detector(model, image, ann)

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)

    # ann = "demo/data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525_mono3d.coco.json"
    # get data info containing calib
    # data_infos = mmcv.load(ann)

    sequence_path = "/home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/0010/000000.png"
    result_path = "/home/chen/PycharmProjects/mmdetection3d/output"
    cam_instrinsic = [[7.215377000000e+02, 0, 6.095593000000e+02], [0, 7.215377000000e+02, 1.728540000000e+02],
                      [0, 0, 1]]

    image = sequence_path
    print(image)

    img_info = dict()
    img_info["file_name"] = os.path.basename(image)
    img_info["cam_intrinsic"] = cam_instrinsic

    data = dict(
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])

    # camera points to image conversion
    if box_mode_3d == Box3DMode.CAM:
        data['img_info'].update(dict(cam_intrinsic=img_info['cam_intrinsic']))

    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    box_3d = result[0]['img_bbox']['boxes_3d'].tensor.numpy()  # 200x9
    scores_3d = result[0]['img_bbox']['scores_3d'].numpy()  # 200x1
    labels_3d = result[0]['img_bbox']['labels_3d'].numpy()  # 200x1
    attrs_3d = result[0]['img_bbox']['attrs_3d'].numpy()  # 200x1

    number = box_3d.shape[0]

    stem, suffix = os.path.splitext(os.path.basename(image))
    write_path = os.path.join(result_path, stem + ".txt")
    with open(write_path, 'w') as f:
        for i in range(number):
            s = str()
            s = str(labels_3d[i]) + " " + str(attrs_3d[i]) + " " + str(scores_3d[i])
            box_s = ""
            for j in range(box_3d[i].shape[0]):
                box_s += " " + str(box_3d[i, j])
            f.write(s + box_s + "\n")

    #show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='mono-det')


if __name__ == '__main__':
    detect_kitti_folder()
    #detect_kitti_single()
