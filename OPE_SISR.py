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
from gpu_mem_track import MemTracker

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
    gpu_tracker = MemTracker(path=save_path+'/')
    gpu_tracker.track()
    model, _, _, _ = prepare_training()
    gpu_tracker.track()
    img_lr = myutils.load_imageToten(args.lr_path).cuda()
    gpu_tracker.track()
    sr_img_bicubic = myutils.load_imageToten(args.lr_path, resize=(args.h, args.w)).cuda()
    bic_psnr = None
    bic_ssim = None
    if args.gt_path is not None:
        img_gt = myutils.load_imageToten(args.gt_path).cuda()
        bic_psnr = utils.calc_psnr((sr_img_bicubic + 1) / 2, (img_gt + 1) / 2)
        bic_ssim = utils.calc_ssim((sr_img_bicubic + 1) / 2, (img_gt + 1) / 2, norm=False)
    else:
        img_gt = None
    gpu_tracker.track()
    sr_img, psnr, ssim, run_time = single_img_sr(img_lr, model, h=args.h, w=args.w, gt=img_gt)
    gpu_tracker.track()
    lr_name = args.lr_path.split('/')[-1].split('.')[0]
    myutils.save_tenimage(sr_img, save_path, f'{lr_name}_SISR_{args.h}x{args.w}.png')
    myutils.save_tenimage(sr_img_bicubic, save_path, f'{lr_name}_bicubic_{args.h}x{args.w}.png')
    log_info = f'lr_path: {args.lr_path}, gt_path: {args.gt_path}, psnr: {psnr}/{bic_psnr}, ssim: {ssim}/{bic_ssim}\n'
    log_info += f'encoder_time: {run_time[0]}, decoder_time: {run_time[1]}, all_time: {run_time[2]}\n'
    log(log_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='save/_train_rdn-OPE-001_exp01')
    parser.add_argument('--ckpt_name', default='epoch-60.pth')
    parser.add_argument('--lr_path', default='test_imgs/div2k_test/0803x4.png')
    parser.add_argument('--gt_path', default='test_imgs/div2k_test/0803.png')
    parser.add_argument('--h', default=1536)
    parser.add_argument('--w', default=2040)
    parser.add_argument('--gpu', default='0')
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
    sub_save_folder = 'SISR_folder/'+ckpt_name_
    save_path = os.path.join(args.exp_folder, sub_save_folder)

    main(config, save_path, args)
