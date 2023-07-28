import argparse
import logging
import os
import os.path as osp
import pprint
import random
import time
from datetime import datetime
import shutil                                                   # For Mike's code

import skimage
from sklearn.metrics import adjusted_rand_score

from pytorch_utils import lr_scheduler as lr_scheduler_custom   # For Mike's code
from pytorch_utils import metadata                              # For Mike's code
import copy                                                     # For Mike's code
import psutil                                                     # For Mike's code
import subprocess                                                     # For Mike's code

import numpy as np
import torch
import torch.backends.cudnn as cudnn
#import torch.distributed as dist
import torch.nn.functional as F
import yaml

from u2pl.dataset.builder import get_loader
from u2pl.models.model_helper import ModelBuilder
#from u2pl.utils.dist_helper import setup_distributed
from u2pl.utils.loss_helper import get_criterion
from u2pl.utils.lr_helper import get_optimizer, get_scheduler
from u2pl.utils.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    load_state,
    set_random_seed,
)
from infer import create_pascal_label_colormap, colorful

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
# parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
# parser.add_argument("--port", default=None, type=int)
parser.add_argument("--output_dirpath", type=str, default="./stats")
parser.add_argument("--network", type=str, default="resnet", help="Type of backbone to use: 'unet' or 'resnet'")
parser.add_argument("--prediction_dir", type=str, default="./predictions_training")
#parser.add_argument("--lower_precision", type=bool, default=False, help="Enables gradient scaling for 16 bit floating point precision")

logger = init_log("global", logging.INFO)
logger.propagate = 0
logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",             # Mike's logger
                        handlers=[logging.StreamHandler()])

def main():
    global args, cfg
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    cfg["exp_path"] = os.path.dirname(args.config)
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    cudnn.enabled = True
    cudnn.benchmark = True

    #rank, word_size = setup_distributed(port=args.port)
    rank = 0
    # if rank == 0:
    #     logger.info("{}".format(pprint.pformat(cfg)))
    #     current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     tb_logger = SummaryWriter(
    #         osp.join(cfg["exp_path"], "log/events_seg/" + current_time)
    #     )
    # else:
    #     tb_logger = None

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]) and rank == 0:
        os.makedirs(cfg["saver"]["snapshot_dir"])

    # Create network.
    is_unet = True
    if args.network == 'resnet':
        is_unet = False
    model = ModelBuilder(cfg["net"], is_unet=is_unet)
    if is_unet:
        module = model.unet
    else:
        modules_back = [model.encoder]
        if cfg["net"].get("aux_loss", False):
            modules_head = [model.auxor, model.decoder]
        else:
            modules_head = [model.decoder]

        if cfg["net"].get("sync_bn", True):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    #local_rank = int(os.environ["LOCAL_RANK"])
    #model = torch.nn.parallel.DistributedDataParallel(
    #    model,
    #    device_ids=[local_rank],
    #    output_device=local_rank,
    #    find_unused_parameters=False,
    #)

    criterion = get_criterion(cfg)

    train_loader_sup, val_loader = get_loader(cfg, seed=seed)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []
    if is_unet:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"]))
    else:
        for module in modules_back:
            params_list.append(
                dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
            )
            model_parameters = filter(lambda p: p.requires_grad, module.parameters())
            #params_encoder = sum([np.prod(p.size()) for p in model_parameters])
        for module in modules_head:
            params_list.append(
                dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
            )
            model_parameters = filter(lambda p: p.requires_grad, module.parameters())
            #params_decoder = sum([np.prod(p.size()) for p in model_parameters])

    optimizer = get_optimizer(params_list, cfg_optim)

    best_prec = 0
    last_epoch = 0

    # auto_resume > pretrain
    if cfg["saver"].get("auto_resume", False):
        lastest_model = os.path.join(cfg["save_path"], "ckpt.pth")
        if not os.path.exists(lastest_model):
            "No checkpoint found in '{}'".format(lastest_model)
        else:
            print(f"Resume model from: '{lastest_model}'")
            best_prec, last_epoch = load_state(
                lastest_model, model, optimizer=optimizer, key="model_state"
            )

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, keys="model_state")

    optimizer_old = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_old, start_epoch=last_epoch
    )

    # start Mike's code
    if os.path.exists(args.output_dirpath):
        logging.info("Stats directory exists, deleting")
        shutil.rmtree(args.output_dirpath)
    os.makedirs(args.output_dirpath)

    plateau_scheduler = lr_scheduler_custom.EarlyStoppingReduceLROnPlateau(optimizer, mode="max")
    train_stats = metadata.TrainingStats()

    best_model = None
    epoch = -1

    # add the file based handler to the logger
    logging.getLogger().addHandler(logging.FileHandler(filename=os.path.join(args.output_dirpath, 'log.txt')))
    train_start_time = time.time()

    # end Mike's code

    # Start to train model
    #for epoch in range(last_epoch, cfg_trainer["epochs"]):

    #start Mike's code
    while not plateau_scheduler.is_done():
        epoch += 1
        logging.info("Epoch: {}".format(epoch))

        train_stats.export(args.output_dirpath)  # update metrics data on disk
        train_stats.plot_all_metrics(output_dirpath=args.output_dirpath)

        train_stats.add_global('batch size', cfg["dataset"]["batch_size"])

    #end Mike's code

        # Training

        #start for inference debugging
        # if os.path.exists(args.prediction_dir):
        #     logging.info("Prediction training directory exists, deleting")
        #     shutil.rmtree(args.prediction_dir)
        # os.makedirs(args.prediction_dir)

        data_list = iter(train_loader_sup.dataset.list_sample)
        tile_size = cfg["dataset"]["train"]["tile_size"]
        #scaler = torch.cuda.amp.GradScaler(enabled=args.lower_precision)  # enabled toggles this on or off
        # end for inference debugging

        # TODO: Replace lr_scheduler with Mike's plateu_scheduler to test with his code
        train(
            model,
            optimizer,
            plateau_scheduler,
            #lr_scheduler,
            criterion,
            train_loader_sup,
            epoch,
            #tb_logger,
            train_stats,             # for plateau scheduler
            is_unet,
            data_list,
            tile_size,
            #scaler                     # for 16 bit float precision
        )

        data_list = None

        # Validation and store checkpoint
        prec = validate(model,
                        val_loader,
                        epoch,
                        train_stats,        # for plateau scheduler
                        criterion,          # for plateau scheduler
                        is_unet)

        if rank == 0:
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_miou": best_prec,
            }

            if prec > best_prec:
                best_prec = prec
                state["best_miou"] = prec
                torch.save(
                    model.state_dict(), osp.join(cfg["saver"]["snapshot_dir"], "best_state_dict.pth")
                )

            torch.save(model.state_dict(), osp.join(cfg["saver"]["snapshot_dir"], "model-state-dict.pth"))

            # logger.info(
            #     "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
            #         best_prec * 100
            #     )
            # )
            # tb_logger.add_scalar("mIoU val", prec, epoch)

            # start Mike's code

            #val_loss = train_stats.get_epoch('val_loss', epoch=epoch) - commented out in Mike's code
            val_accuracy = train_stats.get_epoch('val_accuracy', epoch=epoch)
            plateau_scheduler.step(val_accuracy)

            # update global metadata stats
            train_stats.add_global('train_wall_time', train_stats.get('train_wall_time', aggregator='sum'))
            train_stats.add_global('val_wall_time', train_stats.get('val_wall_time', aggregator='sum'))
            train_stats.add_global('num_epochs_trained', epoch)

            # handle early stopping when loss converges
            if plateau_scheduler.is_equiv_to_best_epoch:
                logging.info('Updating best model with epoch: {} accuracy: {}'.format(epoch, val_accuracy))
                best_model = copy.deepcopy(model)
                # update the global metrics with the best epoch
                train_stats.update_global(epoch)
                # save a state dict (weights only) version of the model
                torch.save(best_model.state_dict(), os.path.join(args.output_dirpath, 'model-state-dict.pt'))

            # end Mike's code

    # start Mike's code
    wall_time = time.time() - train_start_time
    train_stats.add_global('wall_time', wall_time)
    logging.info("Total WallTime: {}seconds".format(train_stats.get_global('wall_time')))

    train_stats.export(args.output_dirpath)  # update metrics data on disk
    train_stats.plot_all_metrics(output_dirpath=args.output_dirpath)
    best_model.cpu()  # move to cpu before saving to simplify loading the model
    # save a python class embedded version of the model
    torch.save(best_model, os.path.join(args.output_dirpath, 'model.pt'))
    # save a state dict (weights only) version of the model
    torch.save(best_model.state_dict(), os.path.join(args.output_dirpath, 'model-state-dict.pt'))
    # end Mike's code

def train(
    model,
    optimizer,
    lr_scheduler,
    criterion,
    data_loader,
    epoch,
    #tb_logger,
    train_stats,                     # for plateau scheduler
    is_unet,
    data_list,
    tile_size
    #scaler                          # for 16 bit float precision
):

    model.train()                   #enables train mode => mean and variance get updated with every epoch

    #data_loader.sampler.set_epoch(epoch)
    data_loader_iter = iter(data_loader)

    #rank, world_size = dist.get_rank(), dist.get_world_size()
    rank = 0
    world_size = 1

    losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    # start Mike's code
    start_time = time.time()
    per_class_accuracy = []

    batch_end = time.time()
    for step, tensor_dict in enumerate(data_loader):
        #with torch.cuda.amp.autocast(enabled=args.lower_precision):  # enabled toggles this on or off
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(data_loader) + step        # step is the batch number
        # lr = lr_scheduler.get_lr()
        # learning_rates.update(lr[0])
        # lr_scheduler.step()

        image, label = tensor_dict                  # NOTE: Add third argument: img_path to debug image saving

        batch_size, h, w = label.size()

        # start checking if images from dataloader look good
        # if epoch >= 0:
        #     for img in range(batch_size):
        #         # color_mask = colorful(mask[img], colormap)
        #         image_path, _ = next(data_list)
        #         image_name = image_path.split("/")[-1]
        #         image_ = image[img][0].squeeze()
        #         skimage.io.imsave(os.path.join(args.prediction_dir, image_name), image_,
        #                           check_contrast=False)
        # end my code

        image, label = image.cuda().float(), label.cuda()
        outs = model(image)

        if is_unet:
            pred = outs
        else:
            pred = outs["pred"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

        if "aux_loss" in cfg["net"].keys():
            aux = outs["aux"]
            aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
            loss = criterion([pred, aux], label)
        else:
            loss = criterion(pred, label)

        max_pred = torch.max(pred)
        min_pred = torch.min(pred)
        max_label = torch.max(label)
        min_label = torch.max(label)

        optimizer.zero_grad()                   # zeros out gradients to start with a clean slate for the next forward pass (accumulating gradients is unnecessary and computationally heavy)
        loss.backward()                         # calculate loss during backprop
        optimizer.step()                        # step in direction calculated from loss

        # once you have the loss value computed
        # scale it to undo fp16 interpretation
        #scaler.scale(loss).backward()
        #scaler.unscale_(optimizer)

        # step the optimizer
        #scaler.step(optimizer)

        # update the scaler to it can keep your activations in the appropriate range
        #scaler.update()

        # gather all loss from different gpus
        reduced_loss = loss.clone().detach()
        #dist.all_reduce(reduced_loss)
        losses.update(reduced_loss.item())

        # start Mike's code
        train_stats.append_accumulate('train_loss', loss.item())
        pred = torch.argmax(pred, dim=1)

        # start my inference debug code
        # mask = pred.squeeze().cpu().numpy()
        # colormap = create_pascal_label_colormap()
        # if epoch >= 0:
        #     for img in range(batch_size):
        #         color_mask = colorful(mask[img], colormap)
        #         image_path, _ = next(data_list)
        #         image_name = image_path.split("/")[-1]
        #         skimage.io.imsave(os.path.join(args.prediction_dir, image_name), np.uint8(color_mask), check_contrast=False)          # to debug inferencing
        # end my inference debug code

        # TODO: start my adjusted random index (ARI) score and per class accuracy code
        #batch_class_accuracy, ARI = ARI_and_class_accuracy(pred.cpu().detach().numpy(), label.cpu().detach().numpy(), batch_size)
        #per_class_accuracy.append(batch_class_accuracy)
        # end my per class accuracy code

        accuracy = torch.mean((pred == label / label.numel()).type(torch.FloatTensor))
        #train_stats.append_accumulate("ARI", ARI)
        train_stats.append_accumulate('train_accuracy', accuracy.item())
        train_stats.append_accumulate('learning_rates', optimizer.param_groups[0]['lr'])

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 100 == 0 and rank == 0:
            # start Mike's code
            cpu_mem_percent_used = psutil.virtual_memory().percent
            gpu_mem_percent_used, memory_total_info = get_gpu_memory()
            gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
            # end Mike's code

            logging.info(
                "Iter [{}/{}]\t"
                #"Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "LR {lr:.5f} ({lr:.7f})\t"
                "cpu_mem: {cpu_mem:2.1f}%\t"
                "gpu_mem: {gpu_mem}% of {total_mem}MiB\t".format(
                    i_iter,
                    cfg["trainer"]["epochs"] * len(data_loader),
                    data_time=data_times,
                    batch_time=batch_times,
                    loss=losses,
                    lr=optimizer.param_groups[0]['lr'],
                    cpu_mem=cpu_mem_percent_used,
                    gpu_mem=gpu_mem_percent_used,
                    total_mem=memory_total_info
                )
            )
        #
        #     tb_logger.add_scalar("lr", learning_rates.avg, i_iter)
        #     tb_logger.add_scalar("Loss", losses.avg, i_iter)

    # start Mike's code
    #per_class_accuracy = np.mean(per_class_accuracy, axis=0)    # calculate per class accuracy for entire epoch
    num_classes = cfg["net"]["num_classes"]
    # for num in range(num_classes):
    #     train_stats.add(epoch, 'per_class_accuracy_class_{}'.format(num), per_class_accuracy[num])
    #train_stats.close_accumulate(epoch, "ARI", method='avg')
    train_stats.close_accumulate(epoch, 'train_loss', method='avg')  # this adds the avg loss to the train stats
    train_stats.close_accumulate(epoch, 'train_accuracy', method='avg')
    train_stats.close_accumulate(epoch, 'learning_rates', method='avg')
    train_stats.add(epoch, 'train_wall_time', time.time() - start_time)


# start my per class accuracy code
def ARI_and_class_accuracy(pred, label, batch_size):
    batch_class_accuracy = []
    ari_accuracy = []
    num_classes = cfg["net"]["num_classes"]

    for img in range(batch_size):
        pred_cpy = pred[img]
        label_cpy = label[img]
        img_accuracy = []

        # make pixel mask for all classes in ground truth and compare with the prediction
        for id in range(0, num_classes):
            pred_mask = pred_cpy == id
            label_mask = label_cpy == id
            per_pixel_accuracy = np.logical_and(pred_mask, label_mask)

            # Divide total number of correctly identified non background pixels, divide by total number of actual non background pixels there should be
            try:
                total_accuracy = np.count_nonzero(per_pixel_accuracy) / np.count_nonzero(label_mask)  # maybe edit denominator to just count_nonzero of label
            except:
                total_accuracy = 0
            img_accuracy.append(total_accuracy)
        batch_class_accuracy.append(img_accuracy)
        ari_accuracy.append(adjusted_rand_score(label_cpy.flatten(), pred_cpy.flatten()))

    return np.mean(batch_class_accuracy, axis=0), np.mean(ari_accuracy)
# end my per class accuracy code


def get_img_path(cfg):
    data_list = []
    for line in open(cfg["dataset"]["val"]["data_list"], "r"):
        arr = [
            "JPEGImages/{}.jpg".format(line.strip()),
            "SegmentationClassAug/{}.png".format(line.strip()),
        ]
        arr = [os.path.join(cfg["dataset"]["val"]["data_root"], item) for item in arr]
        data_list.append(arr)
    return data_list


# start Mike's code
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_info = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    memory_total_info = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
    memory_used_percent = np.asarray(memory_used_info) / np.asarray(memory_total_info)
    return memory_used_percent, memory_total_info
# end Mike's code


def validate(
    model,
    data_loader,
    epoch,
    train_stats,         # for plateau scheduler
    criterion,            # for plateau scheduler
    is_unet
):
    # start Mike's code
    logging.info('Evaluating model against validation data')
    start_time = time.time()
    # end Mike's code

    model.eval()            #turns on eval mode => stop updating mean and variance, etc
    #data_loader.sampler.set_epoch(epoch)

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )
    #rank, world_size = dist.get_rank(), dist.get_world_size()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    for step, batch in enumerate(data_loader):
        images, labels = batch
        images = images.cuda()
        labels = labels.long().cuda()
        batch_size, h, w = labels.shape

        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        if is_unet:
            output = outs
        else:
            output = outs["pred"]
            output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)

        # start Mike's code
        loss = criterion(output, labels)
        train_stats.append_accumulate('val_loss', loss.item())
        pred = torch.argmax(output, dim=1)
        accuracy = torch.mean((pred == labels / labels.numel()).type(torch.FloatTensor))
        train_stats.append_accumulate('val_accuracy', accuracy.item())
        # end Mike's code

        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        #dist.all_reduce(reduced_intersection)
        #dist.all_reduce(reduced_union)
        #dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)

    # start Mike's code

    # close out the accumulating stats with the specified method
    train_stats.append_accumulate('iou', [metric for metric in iou_class])
    train_stats.close_accumulate(epoch, 'iou', method='avg')
    train_stats.close_accumulate(epoch, 'val_loss', method='avg')
    # this adds the avg loss to the train stats
    train_stats.close_accumulate(epoch, 'val_accuracy', method='avg')
    train_stats.add(epoch, 'val_wall_time', time.time() - start_time)

    # end Mike's code

    mIoU = np.mean(iou_class)

    # if rank == 0:
    #     for i, iou in enumerate(iou_class):
    #         logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
    #     logger.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU


if __name__ == "__main__":
    main()
