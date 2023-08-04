import logging
import os
import time
from argparse import ArgumentParser

import albumentations.pytorch
import numpy as np
import skimage.io
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import yaml
from PIL import Image
from tqdm import tqdm
import mrcfile

from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.utils import (
    AverageMeter,
    check_makedirs,
    colorize,
    convert_state_dict,
    intersectionAndUnion,
)
from u2pl.dataset.cryoem import generate_dist, tile
from u2pl.dataset.augmentation import ZScoreNorm


# Setup Parser
def get_parser():
    parser = ArgumentParser(description="PyTorch Evaluation")
    parser.add_argument("--config", type=str, default="experiments/pascal/1464/suponly/config.yaml")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/vsp1/U2PL/sup_stats_1/model-state-dict.pt",        #edited this
        help="evaluation model path",
    )
    parser.add_argument(
        "--save_folder", type=str, default="infer_experiment", help="results save folder"
    )
    return parser


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger, cfg
    args = get_parser().parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = get_logger()
    logger.info(args)

    num_classes = cfg["net"]["num_classes"]

    assert num_classes > 1

    os.makedirs(args.save_folder, exist_ok=True)
    gray_folder = os.path.join(args.save_folder, "gray")
    os.makedirs(gray_folder, exist_ok=True)
    color_folder = os.path.join(args.save_folder, "color")
    os.makedirs(color_folder, exist_ok=True)

    cfg_dset = cfg["dataset"]
    data_root, f_data_list = cfg_dset["val"]["data_root"], cfg_dset["val"]["data_list"]
    data_list = []

    for line in open(f_data_list, "r"):
        arr = [
            "images/{}".format(line.strip()),
            "gold_truth/{}".format(line.replace(".mrc", ".tif").strip())
        ]
        arr = [os.path.join(data_root, item) for item in arr]
        data_list.append(arr)

    # Create network.
    args.use_auxloss = True if cfg["net"].get("aux_loss", False) else False
    logger.info("=> creating model from '{}' ...".format(args.model_path))

    cfg["net"]["sync_bn"] = False
    model = ModelBuilder(cfg["net"])
    checkpoint = torch.load(args.model_path)
    key = "teacher_state" if "teacher_state" in checkpoint.keys() else "model_state"
    logger.info(f"=> load checkpoint[{key}]")

    saved_state_dict = torch.load(args.model_path)        # get Mike's model
    #saved_state_dict = checkpoint["model_state"]           #get U2PL's saved model version
    #saved_state_dict = convert_state_dict(checkpoint["model_state"])
    model.load_state_dict(saved_state_dict, strict=False)
    model.cuda()
    logger.info("Load Model Done!")

    input_scale = [769, 769] if "cityscapes" in data_root else [513, 513]
    colormap = create_pascal_label_colormap()

    # start my code
    if os.path.exists(os.path.join(args.save_folder, "iou_results")):
        print("IoU file already exists. Deleting...")
        os.remove(os.path.join(args.save_folder, "iou_results"))
    iou_folder = open(os.path.join(args.save_folder, "iou_results"), "x")
    # end my code

    # preload probability distribution functions
    image_pdfs = {}
    for label_path in data_list:
        label = skimage.io.imread(label_path[1], as_gray=True)  # as gray makes pixel values either 0 or 1 by default
        label = (label > 0).astype('float32')
        distribution = generate_dist(label)
        image_pdfs[label_path[1]] = distribution

    model.eval()
    for image_path, label_path in tqdm(data_list):
        image_name = image_path.split("/")[-1]

        # Get image
        with mrcfile.open(image_path) as mrc:
            image = mrc.data.astype('float32')
        h, w = image.shape

        # Get label
        label = skimage.io.imread(label_path, as_gray=True)  # as gray makes pixel values either 0 or 1 by default
        label = (label > 0).astype('float32')

        # TODO: do normalization and tiling (albumentations transform) and calculate pdfs
        _normalize = ZScoreNorm()
        _toTensor = albumentations.pytorch.ToTensorV2()
        norm_dict = _normalize(image=image)
        image = norm_dict["image"]
        #image_tile, label_tile = tile(image, label, image_pdfs.get(label_path), cfg=cfg["dataset"])
        tensor_dict = _toTensor(image=image, mask=label)
        image, label = tensor_dict["image"], tensor_dict["mask"]
        image = image.unsqueeze(dim=0)
        image = image.repeat(1, 3, 1, 1)

        #image = F.interpolate(image, input_scale, mode="bilinear", align_corners=True)

        input = image.cuda()
        output = model(input)["pred"]
        output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        color_mask = colorful(mask,colormap)
        skimage.io.imsave(os.path.join(color_folder, image_name), np.uint8(color_mask), check_contrast=False)

        skimage.io.imsave(os.path.join(gray_folder, image_name), np.uint8(mask), check_contrast=False)

        #start my code
        area_intersection, area_union, area_target = intersectionAndUnion(np.array(mask), np.array(label), 1)
        iou = area_intersection / area_union
        iou_folder.write("IoU for " + str(image_name) + " is: " + str(iou) + "\n")
        #end my code


def colorful(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]

    return np.uint8(color_mask)


def create_pascal_label_colormap():
    """Creates a label colormap used in Pascal segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = 255 * np.ones((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]
    colormap[1] = [128, 0, 0]
    colormap[2] = [0, 128, 0]
    colormap[3] = [128, 128, 0]
    colormap[4] = [0, 0, 128]
    colormap[5] = [128, 0, 128]
    colormap[6] = [0, 128, 128]
    colormap[7] = [128, 128, 128]
    colormap[8] = [64, 0, 0]
    colormap[9] = [192, 0, 0]
    colormap[10] = [64, 128, 0]
    colormap[11] = [192, 128, 0]
    colormap[12] = [64, 0, 128]
    colormap[13] = [192, 0, 128]
    colormap[14] = [64, 128, 128]
    colormap[15] = [192, 128, 128]
    colormap[16] = [0, 64, 0]
    colormap[17] = [128, 64, 0]
    colormap[18] = [0, 192, 0]
    colormap[19] = [128, 192, 0]
    colormap[20] = [0, 64, 128]

    return colormap


@torch.no_grad()
def net_process(model, image):
    input = image.cuda()
    output = model(input)["pred"]
    return output


if __name__ == "__main__":
    main()
