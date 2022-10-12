import os
import json
from PIL import Image
import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                                        '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('paired-image-feature-folder')
class FeatureFolder(Dataset):

    def __init__(self, feature_path, hr_path, repeat=20, cache='none', use_feature=True):
        self.repeat = repeat
        self.use_feature = use_feature
        self.cache = cache
        hr_names = sorted(os.listdir(hr_path))
        self.i_files = []
        # prepare img
        for hrname in hr_names:
            if self.cache == 'none':
                file = os.path.join(hr_path, hrname)
                self.i_files.append(file)
            elif self.cache == 'in_memory':
                file_path = os.path.join(hr_path, hrname)
                file = transforms.ToTensor()(Image.open(file_path).convert('RGB'))
                self.i_files.append(file)
            else:
                raise RuntimeError('please use none/in_memory')
        if self.use_feature:
            self.f_files = []
            feature_names = sorted(os.listdir(feature_path))
            for featurename in feature_names:
                file = os.path.join(feature_path, featurename)
                self.f_files.append(file)
            assert len(self.f_files) == len(self.i_files)


    def __len__(self):
        return len(self.i_files) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.i_files)
        if self.cache == 'none':
            hr_img = transforms.ToTensor()(
                Image.open(self.i_files[idx]).convert('RGB'))
        if self.cache == 'in_memory':
            hr_img = self.i_files[idx]

        if not self.use_feature:
            return None, None, hr_img
        feature_dic = torch.load(self.f_files[idx], map_location=lambda storage, loc: storage)
        return feature_dic['feature_base'].squeeze(0), feature_dic['feature_left'].squeeze(0), hr_img

@register('image-folder-ope')
class ImageFoloerOPE(Dataset):

    def __init__(self, hr_path, repeat=20, cache='none'):
        self.repeat = repeat
        self.cache = cache
        hr_names = sorted(os.listdir(hr_path))
        self.i_files = []
        # prepare img
        for hrname in hr_names:
            if self.cache == 'none':
                file = os.path.join(hr_path, hrname)
                self.i_files.append(file)
            elif self.cache == 'in_memory':
                file_path = os.path.join(hr_path, hrname)
                file = transforms.ToTensor()(Image.open(file_path).convert('RGB'))
                self.i_files.append(file)
            else:
                raise RuntimeError('please use none/in_memory')

    def __len__(self):
        return len(self.i_files) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.i_files)
        if self.cache == 'none':
            hr_img = transforms.ToTensor()(
                Image.open(self.i_files[idx]).convert('RGB'))
        if self.cache == 'in_memory':
            hr_img = self.i_files[idx]

        return hr_img


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
