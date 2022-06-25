from __future__ import print_function
import argparse
import skimage
import skimage.io
import skimage.transform
from PIL import Image
from math import log10

import sys
import shutil
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from retrain.LEAStereo import LEAStereo

from config_utils.predict_args import obtain_predict_args
from utils.colorize import get_color_map
from utils.multadds_count import count_parameters_in_MB, comp_multadds
from time import time
from struct import unpack
import matplotlib.pyplot as plt
import re
import numpy as np
import pdb
from path import Path

opt = obtain_predict_args()
opt.kitti2015=1
opt.maxdisp = 192
opt.crop_height = 384
opt.crop_width = 1248
opt.data_path = './dataset/kitti2015/testing/'
opt.test_list = './dataloaders/lists/kitti2015_test.list'
opt.save_path = './predict/kitti2015/images/'
opt.fea_num_layer=6
opt.mat_num_layers=12
opt.fea_filter_multiplier=8
opt.fea_block_multiplier=4
opt.fea_step=3
opt.mat_filter_multiplier=8
opt.mat_block_multiplier=4
opt.mat_step=3
opt.net_arch_fea = 'run/sceneflow/best/architecture/feature_network_path.npy'
opt.cell_arch_fea = 'run/sceneflow/best/architecture/feature_genotype.npy'
opt.net_arch_mat = 'run/sceneflow/best/architecture/matching_network_path.npy'
opt.cell_arch_mat = 'run/sceneflow/best/architecture/matching_genotype.npy'
opt.resume='./run/Kitti15/best/best.pth'


print(opt)

torch.backends.cudnn.benchmark = True

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Building LEAStereo model')
model = LEAStereo(opt)

print('Total Params = %.2fMB' % count_parameters_in_MB(model))
print('Feature Net Params = %.2fMB' % count_parameters_in_MB(model.feature))
print('Matching Net Params = %.2fMB' % count_parameters_in_MB(model.matching))

mult_adds = comp_multadds(model, input_size=(3, opt.crop_height, opt.crop_width))  # (3,192, 192))
print("compute_average_flops_cost = %.2fMB" % mult_adds)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

turbo_colormap_data = get_color_map()


def RGBToPyCmap(rgbdata):
    nsteps = rgbdata.shape[0]
    stepaxis = np.linspace(0, 1, nsteps)

    rdata = [];
    gdata = [];
    bdata = []
    for istep in range(nsteps):
        r = rgbdata[istep, 0]
        g = rgbdata[istep, 1]
        b = rgbdata[istep, 2]
        rdata.append((stepaxis[istep], r, r))
        gdata.append((stepaxis[istep], g, g))
        bdata.append((stepaxis[istep], b, b))

    mpl_data = {'red': rdata,
                'green': gdata,
                'blue': bdata}

    return mpl_data


mpl_data = RGBToPyCmap(turbo_colormap_data)
#plt.register_cmap(name='turbo', data=mpl_data, lut=turbo_colormap_data.shape[0])


def readPFM(file):
    with open(file, "rb") as f:
        # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

        # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)

    return img, height, width


def save_pfm(filename, image, scale=1):
    '''
    Save a Numpy array to a PFM file.
    '''
    color = None
    file = open(filename, "w")
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image.tofile(file)


def test_transform(temp_data, crop_height, crop_width):
    _, h, w = np.shape(temp_data)

    if h <= crop_height and w <= crop_width:
        # padding zero
        temp = temp_data
        temp_data = np.zeros([6, crop_height, crop_width], 'float32')
        temp_data[:, crop_height - h: crop_height, crop_width - w: crop_width] = temp
    else:
        start_x = int((w - crop_width) / 2)
        start_y = int((h - crop_height) / 2)
        temp_data = temp_data[:, start_y: start_y + crop_height, start_x: start_x + crop_width]
    left = np.ones([1, 3, crop_height, crop_width], 'float32')
    left[0, :, :, :] = temp_data[0: 3, :, :]
    right = np.ones([1, 3, crop_height, crop_width], 'float32')
    right[0, :, :, :] = temp_data[3: 6, :, :]
    return torch.from_numpy(left).float(), torch.from_numpy(right).float(), h, w


def load_data(leftname, rightname):
    left = Image.open(leftname)
    right = Image.open(rightname)
    size = np.shape(left)
    height = size[0]
    width = size[1]
    temp_data = np.zeros([6, height, width], 'float32')
    left = np.asarray(left)
    right = np.asarray(right)
    r = left[:, :, 0]
    g = left[:, :, 1]
    b = left[:, :, 2]
    temp_data[0, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[1, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[2, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    r = right[:, :, 0]
    g = right[:, :, 1]
    b = right[:, :, 2]
    # r,g,b,_ = right.split()
    temp_data[3, :, :] = (r - np.mean(r[:])) / np.std(r[:])
    temp_data[4, :, :] = (g - np.mean(g[:])) / np.std(g[:])
    temp_data[5, :, :] = (b - np.mean(b[:])) / np.std(b[:])
    return temp_data



def test_kitti(leftname, rightname, savename):
    input1, input2, height, width = test_transform(load_data(leftname, rightname), opt.crop_height, opt.crop_width)

    input1 = Variable(input1, requires_grad=False)
    input2 = Variable(input2, requires_grad=False)

    model.eval()
    if cuda:
        input1 = input1.cuda()
        input2 = input2.cuda()

    with torch.no_grad():
        prediction = model(input1, input2)

    temp = prediction.cpu()
    temp = temp.detach().numpy()
    if height <= opt.crop_height and width <= opt.crop_width:
        temp = temp[0, opt.crop_height - height: opt.crop_height, opt.crop_width - width: opt.crop_width]
    else:
        temp = temp[0, :, :]
    skimage.io.imsave(savename, (temp * 256).astype('uint16'))




#def plot_disparity(savename, data, max_disp):
#    plt.imsave(savename, data, vmin=0, vmax=max_disp, cmap='turbo')


if __name__ == "__main__":
    #left_path="/home/chen/PycharmProjects/LEAStereo-master/kitti_tracking/image_2/000000.png"
    #right_path="/home/chen/PycharmProjects/LEAStereo-master/kitti_tracking/image_3/000000.png"
    #save_path="/home/chen/PycharmProjects/LEAStereo-master/kitti_tracking/stereo/000000.png"
    #test_kitti(left_path, right_path, save_path)


    sequence_name="0004"
    sequence_path_left = "/home/chen/datasets/kitti/tracking/data_tracking_image_2/training/image_02/"+sequence_name
    sequence_path_right = "/home/chen/datasets/kitti/tracking/data_tracking_image_3/training/image_03/"+sequence_name
    result_path = "/home/chen/datasets/kitti/tracking/stereo/training/"+sequence_name

    names = os.listdir(sequence_path_left)
    names.sort()
    for name in names:
        left_image_path = os.path.join(sequence_path_left,name)
        right_image_path = os.path.join(sequence_path_right,name)
        print(left_image_path)
        print(right_image_path)
        save_image_path = os.path.join(result_path,name)
        test_kitti(left_image_path, right_image_path, save_image_path)



