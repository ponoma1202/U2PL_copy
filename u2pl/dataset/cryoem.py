import copy
import math
import os
import os.path
import random

import numpy as np
import skimage
import torch
import scipy
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import logging

#from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import albumentations.pytorch

# local imports
from . import augmentation as augment
from .base import BaseDataset


class cryoem(BaseDataset):
    def __init__(
        self, data_root, data_list, trs_form, seed=0, n_sup=24, split="val", cfg=None):      #TODO: edit n_sup with total number of images
        super(cryoem, self).__init__(data_list)
        self.data_root = data_root
        self.transform = trs_form
        self.cfg = cfg          # added for tiling

        step = self.cfg.get("n_steps", 1)
        batch_size = self.cfg.get("batch_size")

        random.seed(seed)
        if len(self.list_sample) >= n_sup and split == "train":
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        elif len(self.list_sample) < n_sup and split == "train":
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat

            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample

        self.epoch_len = set_longer_epoch(step, batch_size, len(self.list_sample_new))

    def __getitem__(self, index):
        # load image and its label
        index = index % len(self.list_sample_new)
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])       # edited because of artificially lengthening the epoch
        label_path = os.path.join(self.data_root, self.list_sample_new[index][1])       # edited because of artificially lengthening the epoch
        image = self.cryoem_img_loader(image_path)
        label = self.cryoem_label_loader(label_path)        # dimensions are 4096 x 4096

        # getting random tile and preprocessing
        distribution = generate_dist(label)
        preprocess = build_preprocessing(self.cfg)
        result_dict = preprocess(image=image, mask=label)
        image, label = result_dict["image"].squeeze(), result_dict["mask"]              # TODO: More self.transform(image, label) after preprocessing

        image, label = tile(image, label, distribution, self.cfg)     # pass in transformed image, label pair

        # reshaping
        image = image.repeat(1, 3, 1, 1)            # duplicate the image 3 times to have 3 channels
        label = label.repeat(1, 1, 1, 1)                   # TODO: edited from (1, 3, 1, 1)

        return image[0], label[0, 0].long()

    def __len__(self):          # artificially lengthen epoch.
        return self.epoch_len


def set_longer_epoch(step, batch_size, length):
    if step != 1:
        epoch_len = batch_size * step
        logging.info("Artificially setting epoch to length {} with batch size {} and testing every {} steps".format(epoch_len, batch_size, step))
        return epoch_len
    return length


# Using PDF from generate_dist, cut random tile from image and return the tile and its label
# also handles transform
def tile(image, label, distribution, cfg):
    # TODO: possibly copy Ashira's Tile class to keep track of tile stats
    # 1. sample random "center" (future center for tile) within the densest region of image according to PDF
    # 2. using the center, cut the tile => apply augmentation such as rotation, flipping, etc (use U2PL's RandRotate, etc in augmentation.py)

    tile_size = cfg["tile_size"]        # TODO: Edit tile_size

    x,y = get_centers(label.squeeze(), 1, tile_size=tile_size, distribution=distribution)      # get center of one future tile
    x, y = x[0], y[0]
    half = int(tile_size / 2)
    image_tile = image[(x - half):(x + half), (y - half):(y + half)]
    label_tile = label[(x - half):(x + half), (y - half):(y + half)]

    # TODO: do horizontal/vertical flips here

    return image_tile, label_tile


# generates gaussian filter. Run filter through image 10 times to get the probability density function
# From Ashira's code
def generate_dist(mask, count=10, sigma=40, plotting=False):            # is giving negative values for centers
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


def get_centers(mask, n_tiles=1, tile_size=1024, distribution=None, plotting=False):
    if distribution is None:
        distribution = generate_dist(mask)

    distribution = distribution + np.max(distribution) / 10
    distribution = distribution / np.sum(distribution)

    fx = np.sum(distribution, axis=1)
    Fx = np.cumsum(fx)
    Fx = Fx / Fx[-1]
    fy = np.divide(distribution, np.expand_dims(fx, axis=1))
    Fy = np.cumsum(fy, axis=1)
    Fy = np.divide(Fy, Fy[:, -1])

    if plotting:
        fig, ax = plt.subplots(2, 2, sharex=True)
        ax[0][0].plot(fx)
        ax[0][1].plot(Fx)
        ax[0][0].set_title("pdf X")
        ax[0][1].set_title("cdf X")
        ax[1][0].plot(fy)
        ax[1][1].plot(Fy)
        ax[1][0].set_title("pdf Y")
        ax[1][1].set_title("cdf Y")
        plt.show()

    ids = np.random.uniform(size=(n_tiles, 2))
    idx = np.searchsorted(Fx, ids[:, 0])
    idy = np.zeros_like(idx)
    for i in range(n_tiles):
        idy[i] = np.searchsorted(Fy[idx[i], :], ids[i, 1])

    if plotting:
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
        # We can set the number of bins with the *bins* keyword argument.
        n_bins = 50
        axs[0].hist(idx, bins=n_bins, density=True)
        axs[1].hist(idy, bins=n_bins, density=True)
        plt.show()

        fig = plt.figure()
        plt.scatter(idx, idy)
        plt.show()

    idx = np.minimum(np.maximum(idx, tile_size / 2), mask.shape[0] - tile_size / 2).astype(int)
    idy = np.minimum(np.maximum(idy, tile_size / 2), mask.shape[1] - tile_size / 2).astype(int)
    # print(max(idx), max(idy))

    return (idx, idy)


# TODO: edit config file to input the right transforms
def build_transform_before(cfg):        # what will transform the image before it goes into the dataloader
    trs_form = []
    trs_form.append(augment.Z_score())
    # trs_form = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     #transforms.Normalize()                # how to add custom normalization method?
    #     #transforms.GaussianBlur()
    # ])
    return augment.Compose(trs_form) #trs_form

# start my preprocessing code
# Preprocessing normalization transforms: z-score norm, median filter, and gaussian filter
def build_preprocessing(cfg):
    chain = []
    cfg_pre = cfg["preprocess"]
    if cfg.get(cfg_pre["z_score"]):
        chain.append(augment.ZScoreNorm())
    if cfg.get(cfg_pre["median_filter"]):
        chain.append(albumentations.MedianBlur())
    if cfg.get(cfg_pre["gaussian_blur"]):
        chain.append(albumentations.GaussianBlur(sigma_limit=150))
    chain.append(albumentations.pytorch.ToTensorV2())
    return albumentations.Compose(chain)


def build_cryoloader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 1024)         # TODO: Change n_sup
    # build transform
    trs_form = build_transform_before(cfg)          # TODO: Add transform initializations
    dset = cryoem(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, cfg=cfg)

    # build sampler
#    sample = DistributedSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
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
    n_sup = 10582 - cfg.get("n_sup", 10582)         # TODO change n_sup

    # build transform
    trs_form = build_transform_before(cfg)                    # TODO: add more transform initializtions
    trs_form_unsup = build_transform_before(cfg)
    dset = cryoem(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split, cfg)

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
            cfg["data_root"], data_list_unsup, trs_form_unsup, seed, n_sup, split, cfg
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
