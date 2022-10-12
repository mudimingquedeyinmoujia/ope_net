import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import myutils
import datasets
import models
import utils
from test import single_img_sr
import random
import numpy as np
from cut_img_bicubic import near_scale
import cv2


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def main(config_, save_path, args):
    global config, log, writer
    config = config_
    log, _ = utils.set_save_path(save_path, remove=False, writer=False)

    model, _, _, _ = prepare_training()
    img_lr = myutils.load_imageToten(args.lr_path).cuda()
    init_h, init_w = img_lr.shape[-2:]
    scale_list = np.arange(1 + args.step, args.scale + args.step, args.step)
    id = 0
    p_bar = tqdm(scale_list, leave=True)
    p_bar.set_description(f'saving to {save_path} ...')
    max_h = init_h
    max_w = init_w
    for tmp_scale in p_bar:
        tmp_h = int(init_h * tmp_scale)
        tmp_w = int(init_w * tmp_scale)
        tmp_img_bicubic = myutils.load_imageToten(args.lr_path, resize=(tmp_h, tmp_w)).cuda()
        tmp_img_sr, _, _, _ = single_img_sr(img_lr, model, h=tmp_h, w=tmp_w, gt=None)
        myutils.save_tenimage(tmp_img_sr, save_path, f'sr_{str(id).zfill(4)}.png')
        myutils.save_tenimage(tmp_img_bicubic, save_path, f'bicubic_{str(id).zfill(4)}.png')
        id = id + 1
        if tmp_h > max_h:
            max_h = tmp_h
        if tmp_w > max_w:
            max_w = tmp_w

    bic_list = []
    sr_list = []
    p_bar2 = tqdm(sorted(os.listdir(save_path)), leave=True)
    p_bar2.set_description(f'preparing video  ...')
    for img_name in p_bar2:
        if not img_name.endswith('.png'):
            continue
        full_path = os.path.join(save_path, img_name)
        tmp_img = near_scale(full_path, h=max_h, w=max_w)
        img_name_ = img_name.split('.')[0]
        bic_save_path = os.path.join(save_path, 'bic_video')
        sr_save_path = os.path.join(save_path, 'sr_video')
        os.makedirs(bic_save_path,exist_ok=True)
        os.makedirs(sr_save_path,exist_ok=True)
        if img_name.startswith('bicubic_'):
            bic_save_name = f'{img_name_}_near_{max_h}x{max_w}.png'
            myutils.save_tenimage(tmp_img, svpath=bic_save_path, svname=bic_save_name)
            bic_list.append(os.path.join(bic_save_path, bic_save_name))
        elif img_name.startswith('sr_'):
            sr_save_name = f'{img_name_}_near_{max_h}x{max_w}.png'
            myutils.save_tenimage(tmp_img, svpath=sr_save_path, svname=sr_save_name)
            sr_list.append(os.path.join(sr_save_path, sr_save_name))
        else:
            raise RuntimeError('other image not bicubic/sr')

    # process two folder to video
    bic_list.sort()
    sr_list.sort()
    video_bic = cv2.VideoWriter(os.path.join(bic_save_path, 'bic_video.mp4'),
                                cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), args.frame, (max_w, max_h))
    video_sr = cv2.VideoWriter(os.path.join(sr_save_path, 'sr_video.mp4'),
                               cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), args.frame, (max_w, max_h))
    for i in tqdm(bic_list):
        img = cv2.imread(i)
        video_bic.write(img)
    video_bic.release()

    for i in tqdm(sr_list):
        img = cv2.imread(i)
        video_sr.write(img)
    video_sr.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='save/_train_rdn-OPE-001_exp01')
    parser.add_argument('--ckpt_name', default='epoch-500.pth')
    parser.add_argument('--lr_path', default='test_imgs/scale_test/0810_cut_1075x1224_53x61.png')
    parser.add_argument('--tag', default='1')
    parser.add_argument('--scale', default=20)
    parser.add_argument('--step', default=0.2)
    parser.add_argument('--frame', default=10)
    parser.add_argument('--gpu', default='2')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config_path = os.path.join(args.exp_folder, 'config.yaml')
    if args.ckpt_name is None:
        resume_path = os.path.join(args.exp_folder, 'epoch-best-psnr.pth')
    else:
        resume_path = os.path.join(args.exp_folder, args.ckpt_name)
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    config['resume'] = resume_path

    ckpt_name_ = resume_path.split('/')[-1].split('.')[-2]  # epoch-xxx
    lr_name = args.lr_path.split('/')[-1].split('.')[0]  # 0xxx.png ---> 0xxx
    sub_save_folder = 'SISR_folder_video/' + ckpt_name_ + '/' + lr_name + '_video-'+args.tag
    save_path = os.path.join(args.exp_folder, sub_save_folder)

    main(config, save_path, args)
