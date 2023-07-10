import copy
import math
import os
import os.path
import random

import numpy as np
import torch
import scipy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
#from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from . import augmentation as psp_trsform
from .base import BaseDataset


class cryoem(BaseDataset):
    def __init__(
        self, data_root, data_list, trs_form, seed=0, n_sup=24, split="val"      #TODO: edit n_sup with total number of images
    ):
        super(cryoem, self).__init__(data_list)
        self.data_root = data_root
        self.transform = trs_form
        random.seed(seed)
        if len(self.list_sample) >= n_sup and split == "train":
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        elif len(self.list_sample) < n_sup and split == "train":
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat

            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])       # gets name of images for training
        label_path = os.path.join(self.data_root, self.list_sample_new[index][1])       # get name of images for validation
        image = self.cryoem_loader(image_path)
        label = self.cryoem_loader(label_path)
        image, label = tile(image, label)
        image, label = self.transform(image, label)
        return image[0], label[0, 0].long()     # TODO: check why this is indexed in pascal_voc

    def __len__(self):          # TODO: artificially lengthen epoch
        return len(self.list_sample_new)

# Using PDF from generate_dist, cut random tile from image and return the tile and its label
def tile(image, label):
    # TODO: using gaussian filter, cut random tile and return the tile and its label
    # use Ashira's cut_tile for this
    # 1. sample random "center" (future center for tile) within the densest region of image according to PDF
    # 2. using the center, cut the tile => apply augmentation such as rotation, flipping, etc (use U2PL's RandRotate, etc in augmentation.py)

    return image, label

# generates gaussian filter. Run filter through image 10 times to get the probability density function
# From Ashira's code
def generate_dist(mask, count=10, sigma=40, plotting=False):
    dist = mask #.astype(float)
    count = 10
    for i in range(count):
        dist = scipy.ndimage.gaussian_filter(dist, sigma, mode='reflect')
    dist = dist/np.sum(dist)

    if plotting:
        fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(0, 4096)
        X, Y = np.meshgrid(x, y)
        Z = dist
        # ax.plot_surface(X, Y, Z)
        plt.imshow(Z,cmap='hot')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        plt.show()

    return dist

def build_transfrom(cfg):       # TODO: edit config file to input the right transforms
    trs_form = []
    #mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Z_score())
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
    return psp_trsform.Compose(trs_form)


def build_cryoloader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 10582)         # TODO: Change n_sup
    # build transform
    trs_form = build_transfrom(cfg)
    dset = cryoem(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup)

    # build sampler
#    sample = DistributedSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
 #       sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    return loader


def build_cryo_semi_loader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = 10582 - cfg.get("n_sup", 10582)

    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)
    dset = cryoem(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split)

    if split == "val":
        # build sampler
  #      sample = DistributedSampler(dset)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
   #         sampler=sample,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        # build sampler for unlabeled set
        data_list_unsup = cfg["data_list"].replace("labeled.txt", "unlabeled.txt")
        dset_unsup = cryoem(
            cfg["data_root"], data_list_unsup, trs_form_unsup, seed, n_sup, split
        )

    #    sample_sup = DistributedSampler(dset)
        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
     #       sampler=sample_sup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

     #   sample_unsup = DistributedSampler(dset_unsup)
        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
      #      sampler=sample_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup
