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
import time
from cut_img_bicubic import cut_img_dir, resize_img, resize_img_near


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

    img_hr = myutils.load_imageToten(args.hr_path).cuda()

    scale_list = args.scale_list
    cut_ratio = args.cut_ratio
    h, w = img_hr.shape[-2:]
    p_bar = tqdm(scale_list, leave=False)
    for scale in p_bar:
        p_bar.set_description(f'processing x{scale} ...')
        tmp_h = int(h // scale)
        tmp_w = int(w // scale)
        img_lr = myutils.load_imageToten(loadpath=args.hr_path, resize=(tmp_h, tmp_w)).cuda()
        sr_img, psnr_model, ssim_model, run_time = single_img_sr(img_lr, model, h=h, w=w, gt=img_hr)
        bic_sr_img = resize_img(img_lr, size=(h, w)).cuda()
        near_sr_img = resize_img_near(img_lr, size=(h, w)).cuda()
        metric_fn_psnr = utils.calc_psnr
        metric_fn_ssim = utils.calc_ssim
        psnr_bic = metric_fn_psnr((bic_sr_img + 1) / 2, (img_hr + 1) / 2)
        ssim_bic = metric_fn_ssim((bic_sr_img + 1) / 2, (img_hr + 1) / 2, norm=False)
        log(f'scale: {scale}, psnr: {psnr_model}/{psnr_bic}, ssim: {ssim_model}/{ssim_bic}')

        sr_img_cut, cut_size1 = cut_img_dir(sr_img, cut_ratio[0], cut_ratio[1], cut_ratio[2], cut_ratio[3])
        bic_sr_img_cut, cut_size2 = cut_img_dir(bic_sr_img, cut_ratio[0], cut_ratio[1], cut_ratio[2], cut_ratio[3])
        near_sr_img_cut, cut_size3 = cut_img_dir(near_sr_img, cut_ratio[0], cut_ratio[1], cut_ratio[2], cut_ratio[3])
        myutils.save_tenimage(imgTensor=sr_img_cut, svpath=save_path,
                              svname=f'x{scale}_{cut_size1[0]}x{cut_size1[1]}_sr.png')
        myutils.save_tenimage(imgTensor=bic_sr_img_cut, svpath=save_path,
                              svname=f'x{scale}_{cut_size2[0]}x{cut_size2[1]}_bic.png')
        myutils.save_tenimage(imgTensor=near_sr_img_cut, svpath=save_path,
                              svname=f'x{scale}_{cut_size3[0]}x{cut_size3[1]}_near.png')
        myutils.save_tenimage(imgTensor=img_lr, svpath=save_path,
                              svname=f'x{scale}_{tmp_h}x{tmp_w}_input.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='save/_train_rdn-OPE-004_exp01')
    parser.add_argument('--ckpt_name', default='epoch-480.pth')
    parser.add_argument('--hr_path', default='test_imgs/div2k_test/0830.png')
    parser.add_argument('--cut_ratio', default=[0, 1, 0, 1])
    parser.add_argument('--scale_list', default=[4, 6, 8, 12, 16, 20, 24, 30])
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

    ckpt_name_ = resume_path.split('/')[-1].split('.')[-2]
    img_hr_name = args.hr_path.split('/')[-1].split('.')[-2]
    sub_save_folder = 'SISR-zoom_folder/' + ckpt_name_ + '/' + img_hr_name
    save_path = os.path.join(args.exp_folder, sub_save_folder)

    main(config, save_path, args)
