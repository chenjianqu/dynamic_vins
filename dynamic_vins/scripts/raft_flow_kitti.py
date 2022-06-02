import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder
import tqdm


DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()


    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    if cv2.waitKey(1) == ord('q'):
        sys.exit(0)



def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    base_dir = "/home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/0003"
    flow_dir = "/home/chen/datasets/kitti/tracking/flow_02/training/image_02/0003"

    with torch.no_grad():
        images =[ os.path.join(base_dir,n) for n in os.listdir(base_dir)]
        images = sorted(images)
        for imfile1, imfile2 in tqdm.tqdm( zip(images[:-1], images[1:]) ):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            #flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flow_up = model(image1, image2, iters=20, test_mode=True)[-1]

            flo = flow_up[0].permute(1, 2, 0).cpu().numpy()
            save_name = os.path.join(flow_dir,os.path.basename(imfile2))
            cv2.writeOpticalFlow(save_name,flo)

            viz(image1, flow_up)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    args = parser.parse_args()

    args.models = "models/raft-kitti.pth"

    demo(args)
