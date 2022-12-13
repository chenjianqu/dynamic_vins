from mmdet.apis import init_detector, inference_detector, show_result_pyplot, show_result_ins
import mmcv
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

import cv2
from scipy import ndimage
import io
import os

class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def main():
    config_file = 'configs/solov2/solov2_x101_dcn_fpn_8gpu_3x.py'
    checkpoint_file = '/home/chen/Backup/models/SOLO/SOLOv2_X101_DCN_3x.pth'

    root_img_path = '/home/chen/datasets/MyData/ZED_data/room_dynamic_3/cam0/'
    save_img_path = '/home/chen/datasets/MyData/ZED_data/room_dynamic_3/mask/'


    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # model = init_detector(config_file, checkpoint_file, device='cpu')

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)

    files = os.listdir(root_img_path)
    files.sort()

    for name in files:
        img_path = root_img_path + name
        print(img_path)

        # prepare data
        data = dict(img=img_path)
        data = test_pipeline(data)
        data = scatter(collate([data], samples_per_gpu=1), [device])[0]
        # forward the model
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        # result是一个list，表示每张图片的分割结果,result[i]是一个tuple，分别是(seg_masks, cate_labels, cate_scores)

        if not result or result == [None]:
            continue

        score_thr = 0.25

        cur_result = result[0]
        seg_label = cur_result[0]
        seg_label = seg_label.cpu()

        cate_label = cur_result[1]
        cate_label = cate_label.cpu()

        score = cur_result[2].cpu()

        #过滤掉低分数的
        vis_inds = score > score_thr

        seg_label = seg_label[vis_inds]
        cate_label = cate_label[vis_inds]
        cate_score = score[vis_inds]

        #只保留人的类别
        vis_inds = cate_label == 0
        seg_label = seg_label[vis_inds]
        cate_label = cate_label[vis_inds]
        cate_score = cate_score[vis_inds]

        img_seq = name.split('.')[0]

        f = io.BytesIO()
        torch.save(seg_label, f, _use_new_zipfile_serialization=True)
        save_tensor_path = save_img_path + "seg_label_"+ img_seq + '.pt'
        with open(save_tensor_path, "wb") as outfile:
            outfile.write(f.getbuffer())

        f = io.BytesIO()
        torch.save(cate_label, f, _use_new_zipfile_serialization=True)
        save_tensor_path = save_img_path + "cate_label_" + img_seq + '.pt'
        with open(save_tensor_path, "wb") as outfile:
            outfile.write(f.getbuffer())

        f = io.BytesIO()
        torch.save(cate_score, f, _use_new_zipfile_serialization=True)
        save_tensor_path = save_img_path + "cate_score_" + img_seq + '.pt'
        with open(save_tensor_path, "wb") as outfile:
            outfile.write(f.getbuffer())


if __name__ == '__main__':
    main()
