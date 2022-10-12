import torch
from torchvision import utils
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import PIL.Image
from torchvision import transforms
import random
import json


def load_json(path):
    with open(path, encoding='utf8') as f:
        target_dic = json.load(f)
    return target_dic

def save_json(path, save_dic):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(save_dic, f, ensure_ascii=False, indent=2)


def loss_curve(epochs, loss_list):
    epochs_list = np.arange(epochs) + 1

    plt.plot(epochs_list, loss_list, label="loss")
    plt.xlabel('freq')
    plt.ylabel('value')
    plt.legend(loc=0, ncol=1)  # 参数：loc设置显示的位置，0是自适应；ncol设置显示的列数

    plt.show()


def save_tenimage(imgTensor, svpath, svname, norm=True):
    utils.save_image(
        imgTensor,
        os.path.join(svpath, svname),
        nrow=1,
        normalize=norm,
        value_range=(-1, 1),
    )


def load_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    return ckpt


def calc_ind_(sf, ef):
    """
    get start frequency and end frequency
    return: the start index, end index in orthogonal position encoding and tmp elements number : P_sf + ... + P_ef
    """
    if sf == 0:
        start_ind = 0
    else:
        start_ind = 4 * (sf - 1) * (sf - 1) + 4 * (sf - 1) + 1
    end_ind = 4 * ef * ef + 4 * ef + 1
    return start_ind, end_ind, end_ind - start_ind


def channelExtract(implicit, sf, ef):
    """
    suppose the implicit is all_freq
    return: [N,(P_sf+...+P_ef) * 3,H,W]
    """
    n, ccc, h, w = implicit.shape
    c = int(ccc / 3)
    s_ind, e_ind, num = calc_ind_(sf, ef)
    channel_list = [implicit[:, s_ind:e_ind, :, :], implicit[:, c + s_ind:c + e_ind, :, :],
                    implicit[:, 2 * c + s_ind:2 * c + e_ind, :, :]]
    res = torch.cat(channel_list, dim=1)
    return res


def mergeFeature(feature_base, feature_left):
    N, ccc, h, w = feature_left.shape
    c = int(ccc / 3)
    f_Rb = feature_base[:, 0, :, :].unsqueeze(1)
    f_Rl = feature_left[:, :c, :, :]
    f_Gb = feature_base[:, 1, :, :].unsqueeze(1)
    f_Gl = feature_left[:, c:c * 2, :, :]
    f_Bb = feature_base[:, 2, :, :].unsqueeze(1)
    f_Bl = feature_left[:, c * 2:, :, :]
    return torch.cat([f_Rb, f_Rl, f_Gb, f_Gl, f_Bb, f_Bl], dim=1)


def cutMix(imageLR, imageHR):
    """
    cut mix operation for img super-resolution
    :param imageLR:
    :param imageHR:
    :return: image with size 256(LR) which have a patch of img512(HR)
    """
    N, C, H, W = imageLR.shape
    cN, cC, cH, cW = imageHR.shape
    ratio = random.random() * 0.5 + 0.5  # patch range in 0.3-0.6
    rH, rW = random.randint(0, H - int(ratio * H)), random.randint(0, W - int(ratio * W))
    ds_imageHR = torch.nn.functional.interpolate(imageHR, (H, W))
    ret = imageLR.clone()
    mix0 = random.random()
    mix1 = random.random()
    mix2 = random.random()
    if mix0 < 0.5:
        ret[:, 0, rH:rH + int(ratio * H), rW: rW + int(ratio * W)] = ds_imageHR[:, 0, rH: rH + int(ratio * H),
                                                                     rW: rW + int(ratio * W)]
    if mix1 < 0.5:
        ret[:, 1, rH:rH + int(ratio * H), rW: rW + int(ratio * W)] = ds_imageHR[:, 1, rH: rH + int(ratio * H),
                                                                     rW: rW + int(ratio * W)]
    if mix2 < 0.5:
        ret[:, 2, rH:rH + int(ratio * H), rW: rW + int(ratio * W)] = ds_imageHR[:, 2, rH: rH + int(ratio * H),
                                                                     rW: rW + int(ratio * W)]
    return ret


def load_imageToten(loadpath, resize=None):
    """
    from load path to load image to tensor
    return: [1,3,H,W]
    """
    img = Image.open(loadpath).convert('RGB')

    if resize is not None:
        if isinstance(resize, tuple):
            transform_bicub = transforms.Compose([
                transforms.Resize(resize, interpolation=PIL.Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        if isinstance(resize, int):
            transform_bicub = transforms.Compose([
                transforms.Resize((resize, resize), interpolation=PIL.Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    else:
        transform_bicub = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    img_res = transform_bicub(img).unsqueeze(0)
    return img_res


if __name__ == "__main__":
    img1 = load_imageToten('datasets/benchmark/Set14/HR/bridge.png')
    print(img1.shape)
