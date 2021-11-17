import argparse
import os
import sys

import cv2
import numpy as np
import torch

import RRDBNet_arch as arch

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
args = parser.parse_args()

device = torch.device('cpu')
model_path = sys.path[0] + '/models/RRDB_ESRGAN_x4.pth'
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)
print('Model path: {:s}. \nTesting...'.format(model_path))

if os.path.isfile(args.input):
    _, filename = os.path.split(args.input)
    assert filename != None and filename != ""
    parent_output_folder = os.path.dirname(args.output)
    if not os.path.exists(parent_output_folder):
        os.makedirs(parent_output_folder)
    input_img = args.input
    output_img = args.output
    print("Single Image, reading:", input_img)

    img = cv2.imread(input_img, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite(output_img, output)
    print("Single Image, output:", output_img)

if os.path.isdir(args.input):
    _, filename = os.path.split(args.input)
    assert filename == None or filename == ""
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    input_img_folder = args.input
    output_img_folder = args.output
    idx = 0
    filelist = os.listdir(input_img_folder)
    resultlist = []
    for file in filelist:
        if file.endswith('.png') or file.endswith('jpg'):
            resultlist.append(file)
    print("Reading Image Folder:", input_img_folder)
    for filename in resultlist:
        idx += 1
        path = input_img_folder + '/' + filename
        print("No.", idx, filename)

        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        output_file = output_img_folder + '/' + filename
        cv2.imwrite(output_file, output)
        print("No.", idx, "Output:", output_file)
