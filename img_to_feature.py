import random

from models.CIR_net import Fourier_render_patch_int, calc_ind
from models.CIR_net import get_embedder, make_coord
import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import myutils
import utils
from torch import optim
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


class Img2FeatureProcessor_full():
    """
    process img: [3,H,W] to feature: [Sn,h,w], where h=H/down_sample,w=W/down_sample
    """

    def __init__(self, img_folder, save_folder, down_scale=4, all_freq=2, sample_factor=1):
        self.render = Fourier_render_patch_int(all_freq=all_freq)
        self.embed_fn, self.all_C = get_embedder(all_freq=all_freq, omega=math.pi)
        self.save_folder = save_folder
        self.img_folder = img_folder
        self.down_scale = down_scale
        self.all_freq = all_freq
        self.sample_factor = sample_factor

        self.feature_folder = os.path.join(self.save_folder, 'features')
        self.re_rend_folder = os.path.join(self.save_folder, 're_rend')
        self.log_folder = os.path.join(self.save_folder, 'logs')
        os.makedirs(self.feature_folder, exist_ok=True)
        os.makedirs(self.re_rend_folder, exist_ok=True)
        os.makedirs(self.log_folder, exist_ok=True)

    def process_one(self, img_path, re_rend=True):
        print(f'processing {img_path}')
        img_name = img_path.split('/')[-1].split('.')[-2]
        img_hr = myutils.load_imageToten(img_path).cuda()
        hr_h, hr_w = img_hr.shape[-2:]
        f_h, f_w = hr_h // self.down_scale, hr_w // self.down_scale
        crop_hr_h, crop_hr_w = f_h * self.down_scale, f_w * self.down_scale
        img_hr = img_hr[:, :, :crop_hr_h, :crop_hr_w]

        log_info = [f'# process {img_path}, H: {crop_hr_h}/{hr_h}, W: {crop_hr_w}/{hr_w}']
        log_info.append(
            f'# all_freq: {self.all_freq}, down_scale: {self.down_scale}, sample_factor: {self.sample_factor}')
        log_info.append(f'feature size: {f_h}x{f_w}')

        hr_center_coords = make_coord((f_h * self.down_scale * self.sample_factor,
                                       f_w * self.down_scale * self.sample_factor)).unsqueeze(0).cuda()  # [1,H*W,2]
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6
        rx = 2 / f_h / 2  # half pixel of feature map
        ry = 2 / f_w / 2
        feature_center_coords = make_coord((f_h, f_w), flatten=False) \
            .permute(2, 0, 1).unsqueeze(0).expand(img_hr.shape[0], 2, f_h, f_w).cuda()  # N,2,f_h,f_w

        feature_sets = {}
        j = 0
        for vx in vx_lst:
            for vy in vy_lst:
                j = j + 1
                hr_center_coords_ = hr_center_coords.clone()
                hr_center_coords_[:, :, 0] += vx * rx + eps_shift
                hr_center_coords_[:, :, 1] += vy * ry + eps_shift
                # hr_center_coords_.clamp_(-1 + 1e-6, 1 - 1e-6)
                hr_center_coords_clamp = torch.clamp(hr_center_coords_, -1 + 1e-6, 1 - 1e-6)
                q_coord = F.grid_sample(  # get pixel coord of feature map
                    feature_center_coords, hr_center_coords.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  # [N,H*W,2]
                rel_coord = hr_center_coords_ - q_coord  # N,H*W,2
                rel_coord[:, :, 0] *= feature_center_coords.shape[-2] * 0.5
                rel_coord[:, :, 1] *= feature_center_coords.shape[-1] * 0.5
                emb_value, C = self.embed_fn(rel_coord, sf=0)  # [N,H*W,3C]

                hr_rgb = query_gt(img_hr, hr_center_coords_clamp)  # [N,H*W,3]
                hr_R = hr_rgb[:, :, 0].unsqueeze(-1).repeat(1, 1, self.all_C)
                hr_G = hr_rgb[:, :, 1].unsqueeze(-1).repeat(1, 1, self.all_C)
                hr_B = hr_rgb[:, :, 2].unsqueeze(-1).repeat(1, 1, self.all_C)
                hr_CCC = torch.cat([hr_R, hr_G, hr_B], dim=-1)  # [N,H*W,3C]

                projs = hr_CCC * emb_value  # [N,H*W,3C]
                tmp_bar = tqdm(range(q_coord.shape[1]), leave=False)
                tmp_bar.set_description(f'adding product {j}/4 ... {img_name}.png')
                for i in tmp_bar:
                    tmp_key = (q_coord[0, i, :][0].item(), q_coord[0, i, :][1].item())
                    if feature_sets.get(tmp_key) is None:
                        feature_sets.update({tmp_key: []})
                    feature_sets[tmp_key].append(projs[0, i, :].unsqueeze(0))  # (,2) ---> [(1,3C),...]

        ans_feature = torch.zeros(1, C * 3, f_h, f_w).cuda()
        p_bar = tqdm(range(f_h), leave=False)
        p_bar.set_description(f'pixel integral ... {img_name}.png')
        for x in p_bar:
            for y in range(f_w):
                x_center = feature_center_coords[0, 0, x, y]
                y_center = feature_center_coords[0, 1, x, y]
                search_key = (x_center.item(), y_center.item())
                # print(f'{search_key}: num:{len(feature_sets[search_key])}')

                tmp_projs_list = feature_sets[search_key]
                tmp_projs_ten = torch.cat(tmp_projs_list, dim=0)  # nums,3C
                final_projs = torch.sum(tmp_projs_ten, dim=0, keepdim=True)  # 1,3C
                final_projs = final_projs / ((self.down_scale * self.sample_factor) ** 2)
                final_projs = final_projs / 4
                feature_sets[search_key] = final_projs
                # (,2) ---> (1,3C)
                ans_feature[0, :, x, y] = final_projs.squeeze(0)

        feature_name = f'{img_name}_x{self.down_scale}_f{self.all_freq}_full.pt'
        log_name = f'{img_name}_x{self.down_scale}_f{self.all_freq}_full_log.txt'
        f_base = myutils.channelExtract(ans_feature, sf=0, ef=0)  # N,3C,f_h,f_w
        f_left = myutils.channelExtract(ans_feature, sf=1, ef=self.all_freq)  # N,3C,f_h,f_w
        torch.save({'feature_base': f_base, 'feature_left': f_left},
                   os.path.join(self.feature_folder, feature_name))
        if re_rend:
            print('re_rending...')
            rend_base = self.render(f_base, h=crop_hr_h, w=crop_hr_w, sf=0, ef=0)
            rend_left = self.render(f_left, h=crop_hr_h, w=crop_hr_w, sf=1)
            re_rend_img = rend_base + rend_left
            re_rend_img.clamp_(-1, 1)
            # re_rend_img = self.render(ans_feature, h=crop_hr_h, w=crop_hr_w, sf=0, ef=self.all_freq)
            myutils.save_tenimage(re_rend_img, self.re_rend_folder,
                                  f'{img_name}_x{self.down_scale}_f{self.all_freq}_full_R.png')
            myutils.save_tenimage(rend_left, self.re_rend_folder,
                                  f'{img_name}_x{self.down_scale}_f{self.all_freq}_full_Rl.png')
            myutils.save_tenimage(f_base, self.re_rend_folder,
                                  f'{img_name}_x{self.down_scale}_f{self.all_freq}_full_F.png')
            myutils.save_tenimage(rend_base, self.re_rend_folder,
                                  f'{img_name}_x{self.down_scale}_f{self.all_freq}_full_Rb.png')
            psnr_metric = utils.calc_psnr
            ssim_metric = utils.calc_ssim
            psnr_val = psnr_metric((re_rend_img + 1) / 2, (img_hr + 1) / 2)
            ssim_val = ssim_metric((re_rend_img + 1) / 2, (img_hr + 1) / 2, norm=False)
            log_info.append(f're_rend metric: psnr: {psnr_val}, ssim: {ssim_val}')
        with open(os.path.join(self.log_folder, log_name), 'a') as f:
            print('\n'.join(log_info), file=f)

        return psnr_val, ssim_val

    def process(self, ele_list=None):

        log_path = os.path.join(self.save_folder, f'down_x{self.down_scale}_freq{self.all_freq}_log.txt')
        self.files_path = [os.path.join(self.img_folder, file) for file in sorted(os.listdir(self.img_folder))]
        if ele_list is None:
            ele_list = [i for i in range(len(self.files_path))]

        avg_psnr = utils.Averager()
        avg_ssim = utils.Averager()
        for index in ele_list:
            tmp_psnr, tmp_ssim = self.process_one(self.files_path[index])
            avg_psnr.add(tmp_psnr)
            avg_ssim.add(tmp_ssim)
            tmp_info = f'img_path: {self.files_path[index]}, psnr: {tmp_psnr}, ssim:{tmp_ssim}'
            with open(log_path, 'a') as f:
                print(tmp_info, file=f)

        final_info = f'avg_psnr: {avg_psnr.item()}, avg_ssim: {avg_ssim.item()}'
        with open(log_path, 'a') as f:
            print(final_info, file=f)



def query_gt(gt_img, coords):
    """
    gt_img: [N,3,H,W]
    coord: [N,qbatch,2]
    return: [N,qbatch,3]
    """
    gt_query = F.grid_sample(
        gt_img, coords.flip(-1).unsqueeze(1),
        mode='bicubic', align_corners=False)[:, :, 0, :] \
        .permute(0, 2, 1)  # N,qbatch,3
    return gt_query



def get_sample(qsize):
    return (torch.rand((1, qsize, 2)) - 0.5) * 2  # 1,qsize,2



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    data_folder = 'datasets/benchmark/Set14/HR'
    down_scale = 5
    all_freq = 6
    save_folder = f'datasets/process_benchmark/process_set14/down_x{down_scale}_freq{all_freq}_fix'
    os.makedirs(save_folder, exist_ok=False)
    feature_processor = Img2FeatureProcessor_full(data_folder, save_folder, down_scale=down_scale,
                                                  all_freq=all_freq)
    feature_processor.process()


    # process one
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # data_folder = 'datasets/div2k/DIV2K_valid_HR'
    # save_folder = 'datasets/process_div2k_validHR_freq3'
    # target_path = 'datasets/div2k/DIV2K_valid_HR/0845.png'
    # feature_processor = Img2FeatureProcessor_full(data_folder, save_folder, all_freq=3)
    # feature_processor.process_one(target_path)

    # check
    # files = sorted(os.listdir('datasets/process_div2k_trainHR_freq3/features'))
    # n = 1
    # print(len(files))
