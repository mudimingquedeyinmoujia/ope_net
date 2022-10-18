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
import myutils
from tensorboardX import SummaryWriter
from test import eval_both_ope, test_both_ope, single_img_sr

# def eval_both_ope(loader, model):
#     model.eval()
#
#     metric_fn_psnr = utils.calc_psnr
#     metric_fn_ssim = utils.calc_ssim
#
#     val_res_psnr = utils.Averager()
#     val_res_ssim = utils.Averager()
#
#     pbar = tqdm(loader, leave=False, desc='val_both')
#     with torch.no_grad():
#         for batch in pbar:
#             batch_lr = batch['lr'].cuda()
#             gt = batch['hr'].cuda()
#             gt_size = gt.shape[2]
#             pred, _ = model.inference(batch_lr, h=gt_size, w=gt_size)
#             pred.clamp_(-1, 1)
#
#             res_psnr = metric_fn_psnr((pred + 1) / 2, (gt + 1) / 2)
#             res_ssim = metric_fn_ssim((pred + 1) / 2, (gt + 1) / 2, norm=False)
#             val_res_psnr.add(res_psnr.item(), gt.shape[0])
#             val_res_ssim.add(res_ssim.item(), gt.shape[0])
#
#             pbar.set_description('val {:.4f}'.format(val_res_psnr.item()))
#
#     return val_res_psnr.item(), val_res_ssim.item()
#
#
# def test_both_ope(loader, model, log_fn, log_name, eval_type=None):
#     model.eval()
#     metric_fn_ssim = utils.calc_ssim
#
#     if eval_type is None:
#         metric_fn_psnr = utils.calc_psnr
#     elif eval_type.startswith('div2k'):
#         scale = int(eval_type.split('-')[1])
#         metric_fn_psnr = partial(utils.calc_psnr, dataset='div2k', scale=scale)
#     elif eval_type.startswith('benchmark'):
#         scale = int(eval_type.split('-')[1])
#         metric_fn_psnr = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
#     else:
#         raise NotImplementedError
#
#     val_res_psnr = utils.Averager()
#     val_res_ssim = utils.Averager()
#     avg_time_encoder = utils.Averager()
#     avg_time_render = utils.Averager()
#     avg_time_all = utils.Averager()
#     pbar = tqdm(loader, leave=False, desc='test_both')
#     id = 0
#     with torch.no_grad():
#         for batch in pbar:
#             batch_lr = batch['lr'].cuda()
#             gt = batch['gt'].cuda()
#             gt_size = gt.shape[-2:]
#             pred, run_time = model.inference(batch_lr, h=gt_size[0], w=gt_size[1])
#             pred.clamp_(-1, 1)
#
#             res_psnr = metric_fn_psnr((pred + 1) / 2, (gt + 1) / 2)
#             res_ssim = metric_fn_ssim((pred + 1) / 2, (gt + 1) / 2, norm=False)
#             log_fn(
#                 f'test_img: {id}, psnr: {res_psnr.item()}, ssim: {res_ssim.item()}, time: {run_time[0]}s/{run_time[1]}s/{run_time[2]}s',
#                 filename=log_name)
#             val_res_psnr.add(res_psnr.item(), gt.shape[0])
#             val_res_ssim.add(res_ssim.item(), gt.shape[0])
#             avg_time_encoder.add(run_time[0], gt.shape[0])
#             avg_time_render.add(run_time[1], gt.shape[0])
#             avg_time_all.add(run_time[2], gt.shape[0])
#
#             id += 1
#
#             pbar.set_description('img:{}, psnr: {:.4f}, ssim: {:.4f}'.format(id - 1, res_psnr.item(), res_ssim.item()))
#
#     return val_res_psnr.item(), val_res_ssim.item(), [avg_time_encoder.item(), avg_time_render.item(),
#                                                       avg_time_all.item()]
#
#
# def single_img_sr(lr_img, model, h, w, gt=None):
#     model.eval()
#     with torch.no_grad():
#         pred, run_time = model.inference(lr_img, h=h, w=w)
#         pred.clamp_(-1, 1)
#         if gt is not None:
#             metric_fn_psnr = utils.calc_psnr
#             metric_fn_ssim = utils.calc_ssim
#             res_psnr = metric_fn_psnr((pred + 1) / 2, (gt + 1) / 2)
#             res_ssim = metric_fn_ssim((pred + 1) / 2, (gt + 1) / 2, norm=False)
#             return pred, res_psnr, res_ssim, run_time
#         else:
#             return pred, None, None, run_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='save/_train_rdn-OPE-007_exp01')
    parser.add_argument('--test_config', default='configs/test-CIRnet/test_CIR-SR-set14-x4.yaml')
    parser.add_argument('--gpu', default='2')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    exp_config_path = os.path.join(args.exp_folder, 'config.yaml')
    with open(exp_config_path, 'r') as f:
        exp_config = yaml.load(f, Loader=yaml.FullLoader)
        print('exp_config loaded.')

    test_name = args.test_config.split('/')[-1].split('.')[-2]
    save_dir = os.path.join(args.exp_folder, 'TEST_folder/' + test_name)
    os.makedirs(save_dir, exist_ok=True)
    log, _ = utils.set_save_path(save_dir, remove=False, writer=False)

    ckpt_list = [int(ckpt_name.split('.')[0].split('-')[-1]) for ckpt_name in os.listdir(args.exp_folder) if
                 ckpt_name.endswith('0.pth') or ckpt_name.endswith('5.pth')]
    ckpt_list = sorted(ckpt_list)

    test_dic = {}
    if 'test_info.json' in os.listdir(save_dir):
        test_dic = myutils.load_json(path=os.path.join(save_dir, 'test_info.json'))
        tested_ckpt_list = [int(key) for key in test_dic.keys()]
        tested_ckpt_list = sorted(tested_ckpt_list)
    else:
        tested_ckpt_list = []

    for i in range(1, 1001):
        if i in ckpt_list and i not in tested_ckpt_list:
            # perform test
            ckpt_num = i
            print(f'testing ckpt: {ckpt_num}')
            ckpt_name = f'epoch-{ckpt_num}.pth'
            resume_path = os.path.join(args.exp_folder, ckpt_name)

            exp_config['resume'] = resume_path

            log_name = ckpt_name.split('.')[-2] + '_log.txt'

            with open(args.test_config, 'r') as f:
                test_config = yaml.load(f, Loader=yaml.FullLoader)

            test_spec = test_config['test_dataset']
            dataset = datasets.make(test_spec['dataset'])
            dataset = datasets.make(test_spec['wrapper'], args={'dataset': dataset})
            loader = DataLoader(dataset, batch_size=test_spec['batch_size'],
                                num_workers=8, pin_memory=True)

            sv_file = torch.load(exp_config['resume'], map_location=lambda storage, loc: storage)
            model = models.make(sv_file['model'], load_sd=True).cuda()

            test_psnr, test_ssim, test_run_time = test_both_ope(loader, model, log, log_name,
                                                                eval_type=test_config.get('eval_type'), up_down=test_config.get('up_down'))

            log('test avg: psnr={:.4f}'.format(test_psnr), filename=log_name)
            log('test avg: ssim={:.4f}'.format(test_ssim), filename=log_name)
            log(f'test avg encoder time: {test_run_time[0]}s', filename=log_name)
            log(f'test avg render time: {test_run_time[1]}s', filename=log_name)
            log(f'test avg all time: {test_run_time[2]}s', filename=log_name)

            test_dic.update({str(i): [test_psnr, test_ssim]})
            myutils.save_json(path=os.path.join(save_dir, 'test_info.json'), save_dic=test_dic)

    writer = SummaryWriter(os.path.join(save_dir, 'runs'))
    all_keys = sorted([int(key) for key in test_dic.keys()])
    for key in all_keys:
        writer.add_scalar('scalar/test_psnr', test_dic[str(key)][0], key)
        writer.add_scalar('scalar/test_ssim', test_dic[str(key)][1], key)
