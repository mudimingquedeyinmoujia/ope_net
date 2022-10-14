import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import make_grid
import datasets
import models
import utils
import time
from cut_img_bicubic import resize_img


def eval_both_ope(loader, model):
    model.eval()

    metric_fn_psnr = utils.calc_psnr
    metric_fn_ssim = utils.calc_ssim

    val_res_psnr = utils.Averager()
    val_res_ssim = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val_both')
    with torch.no_grad():
        for batch in pbar:
            batch_lr = batch['lr'].cuda()
            gt = batch['hr'].cuda()
            gt_size = gt.shape[2]
            pred, _ = model.inference(batch_lr, h=gt_size, w=gt_size)
            pred.clamp_(-1, 1)

            res_psnr = metric_fn_psnr((pred + 1) / 2, (gt + 1) / 2)
            res_ssim = metric_fn_ssim((pred + 1) / 2, (gt + 1) / 2, norm=False)
            val_res_psnr.add(res_psnr.item(), gt.shape[0])
            val_res_ssim.add(res_ssim.item(), gt.shape[0])

            pbar.set_description('val {:.4f}'.format(val_res_psnr.item()))

    return val_res_psnr.item(), val_res_ssim.item()


def test_both_ope(loader, model, log_fn, log_name, eval_type=None, up_down=None):
    model.eval()
    if up_down is None:
        up_down = 1
    metric_fn_ssim = utils.calc_ssim
    if eval_type is None:
        metric_fn_psnr = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn_psnr = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn_psnr = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res_psnr = utils.Averager()
    val_res_ssim = utils.Averager()
    avg_time_encoder = utils.Averager()
    avg_time_render = utils.Averager()
    avg_time_all = utils.Averager()
    pbar = tqdm(loader, leave=False, desc='test_both')
    id = 0
    with torch.no_grad():
        for batch in pbar:
            batch_lr = batch['lr'].cuda()
            gt = batch['gt'].cuda()
            gt_size = gt.shape[-2:]
            if up_down != 1:
                pred, run_time = model.inference(batch_lr, h=gt_size[0] * up_down, w=gt_size[1] * up_down)
                pred.clamp_(-1, 1)
                pred = resize_img(pred, (gt_size[0], gt_size[1])).cuda()

            else:
                pred, run_time = model.inference(batch_lr, h=gt_size[0], w=gt_size[1])
                pred.clamp_(-1, 1)

            res_psnr = metric_fn_psnr((pred + 1) / 2, (gt + 1) / 2)
            res_ssim = metric_fn_ssim((pred + 1) / 2, (gt + 1) / 2, norm=False)
            log_fn(
                f'test_img: {id}, psnr: {res_psnr.item()}, ssim: {res_ssim.item()}, time: {run_time[0]}s/{run_time[1]}s/{run_time[2]}s',
                filename=log_name)
            val_res_psnr.add(res_psnr.item(), gt.shape[0])
            val_res_ssim.add(res_ssim.item(), gt.shape[0])
            avg_time_encoder.add(run_time[0], gt.shape[0])
            avg_time_render.add(run_time[1], gt.shape[0])
            avg_time_all.add(run_time[2], gt.shape[0])

            id += 1

            pbar.set_description('img:{}, psnr: {:.4f}, ssim: {:.4f}'.format(id - 1, res_psnr.item(), res_ssim.item()))

    return val_res_psnr.item(), val_res_ssim.item(), [avg_time_encoder.item(), avg_time_render.item(),
                                                      avg_time_all.item()]


def single_img_sr(lr_img, model, h, w, gt=None, up_down=None, flip=None):
    model.eval()
    if up_down is None:
        up_down = 1
    with torch.no_grad():
        # pred, run_time = model.inference(lr_img, h=h, w=w)
        # pred.clamp_(-1, 1)
        if flip is not None:
            pred, run_time = model.inference(lr_img, h=h, w=w, flip_conf=flip)
            pred.clamp_(-1, 1)
        else:
            if up_down != 1:
                pred, run_time = model.inference(lr_img, h=h * up_down, w=w * up_down)
                pred.clamp_(-1, 1)
                pred = resize_img(pred, (h, w)).cuda()
            else:
                pred, run_time = model.inference(lr_img, h=h, w=w)
                pred.clamp_(-1, 1)

        if gt is not None:
            metric_fn_psnr = utils.calc_psnr
            metric_fn_ssim = utils.calc_ssim
            res_psnr = metric_fn_psnr((pred + 1) / 2, (gt + 1) / 2)
            res_ssim = metric_fn_ssim((pred + 1) / 2, (gt + 1) / 2, norm=False)
            return pred, res_psnr, res_ssim, run_time
        else:
            return pred, None, None, run_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='save/_train_rdn-OPE-001_exp01')
    parser.add_argument('--ckpt_name', default='epoch-1000.pth')
    parser.add_argument('--test_config', default='configs/test-CIRnet/test_CIR-SR-div2k-x4.yaml')
    parser.add_argument('--gpu', default='2')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    exp_config_path = os.path.join(args.exp_folder, 'config.yaml')
    resume_path = os.path.join(args.exp_folder, args.ckpt_name)
    log_name = args.ckpt_name.split('.')[-2] + '_log.txt'
    with open(exp_config_path, 'r') as f:
        exp_config = yaml.load(f, Loader=yaml.FullLoader)
        print('exp_config loaded.')
    exp_config['resume'] = resume_path

    with open(args.test_config, 'r') as f:
        test_config = yaml.load(f, Loader=yaml.FullLoader)

    test_spec = test_config['test_dataset']
    dataset = datasets.make(test_spec['dataset'])
    dataset = datasets.make(test_spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=test_spec['batch_size'],
                        num_workers=8, pin_memory=True)
    test_name = args.test_config.split('/')[-1].split('.')[-2]
    save_path = os.path.join(args.exp_folder, 'TEST_folder/' + test_name)
    log, _ = utils.set_save_path(save_path, remove=False, writer=False)

    sv_file = torch.load(exp_config['resume'], map_location=lambda storage, loc: storage)
    model = models.make(sv_file['model'], load_sd=True).cuda()

    test_psnr, test_ssim, test_run_time = test_both_ope(loader, model, log, log_name,
                                                        eval_type=test_config.get('eval_type'),
                                                        up_down=test_config.get('up_down'))

    log('test avg: psnr={:.4f}'.format(test_psnr), filename=log_name)
    log('test avg: ssim={:.4f}'.format(test_ssim), filename=log_name)
    log(f'test avg encoder time: {test_run_time[0]}s', filename=log_name)
    log(f'test avg render time: {test_run_time[1]}s', filename=log_name)
    log(f'test avg all time: {test_run_time[2]}s', filename=log_name)
