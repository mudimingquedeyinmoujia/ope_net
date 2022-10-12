import os

import torch
import numpy as np
import myutils
from PIL import Image
import PIL.Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def resize_img(img, size):
    img_norm = (img+1)/2
    ans = transforms.ToTensor()(
        transforms.Resize(size, InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img_norm.squeeze(0)))).unsqueeze(0)
    return (ans-0.5)/0.5


def bicubic_scale(img_path, scale, save_path):
    img = myutils.load_imageToten(img_path)
    img_name = img_path.split('/')[-1].split('.')[0]
    h, w = img.shape[-2:]
    print(f'img: {img_name}, h: {h}, w: {w}')

    target_h = int(h // scale)
    target_w = int(w // scale)
    img_target = myutils.load_imageToten(img_path, resize=(target_h, target_w))
    print(f'img: {img_name}, target_h: {target_h}, target_w: {target_w}')

    save_name = f'{img_name}_{target_h}x{target_w}.png'
    os.makedirs(save_path, exist_ok=True)
    myutils.save_tenimage(img_target, save_path, save_name)


def cut_img(loadpath, h_1=0, h_2=1, w_1=0, w_2=1):
    save_path = 'test_imgs/pre_cut'
    os.makedirs(save_path, exist_ok=True)
    img = myutils.load_imageToten(loadpath)
    img_name = loadpath.split('/')[-1].split('.')[0]
    h, w = img.shape[-2:]
    h_up = int(h * h_1)
    h_down = int(h * h_2)
    w_left = int(w * w_1)
    w_right = int(w * w_2)
    img = img[:, :, h_up:h_down, w_left:w_right]
    cut_h, cut_w = img.shape[-2:]
    save_name = f'{img_name}_cut_{cut_h}x{cut_w}.png'
    myutils.save_tenimage(img, save_path, save_name)


def cut_img_dir(img, h_1=0, h_2=1, w_1=0, w_2=1):
    h, w = img.shape[-2:]
    h_up = int(h * h_1)
    h_down = int(h * h_2)
    w_left = int(w * w_1)
    w_right = int(w * w_2)
    img = img[:, :, h_up:h_down, w_left:w_right]
    cut_h, cut_w = img.shape[-2:]
    return img, [cut_h, cut_w]


def near_scale(loadpath, h, w):
    img = Image.open(loadpath)
    transform_near = transforms.Compose([
        transforms.Resize((h, w), interpolation=PIL.Image.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img_res = transform_near(img).unsqueeze(0)
    return img_res


if __name__ == '__main__':
    # cut_img
    # cut_path = 'datasets/div2k/DIV2K_valid_HR/0810.png'
    # cut_img(cut_path, h_1=0.1,h_2=0.8, w_1=0.1, w_2=0.7)

    # scale img
    img_path = 'test_imgs/pre_cut/0810_cut_1075x1224.png'
    scale = 20
    save_path = 'test_imgs/scale_test'
    bicubic_scale(img_path=img_path, scale=scale, save_path=save_path)
