import argparse
import logging
import os
import pprint
import time
import shutil

from sklearn.metrics import adjusted_rand_score
from pytorch_utils import lr_scheduler as lr_scheduler_custom
from pytorch_utils import metadata
import copy
import psutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from u2pl.dataset.builder import get_loader
from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.loss_helper import get_criterion
from u2pl.utils.lr_helper import get_optimizer
from u2pl.utils.utils import (
    AverageMeter,
    intersectionAndUnion,
    set_random_seed,
)

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="experiments/pascal/1464/suponly/config.yaml")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dirpath", type=str, default="./stats")

def main():
    global args, cfg
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

    cfg["exp_path"] = os.path.dirname(args.config)
    logging.info("{}".format(pprint.pformat(cfg)))

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    # Create network.
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    criterion = get_criterion(cfg)
    train_loader_sup, val_loader = get_loader(cfg, seed=seed)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]

    params_list = []
    for module in modules_back:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"]))
    for module in modules_head:
        params_list.append(dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"]))

    optimizer = get_optimizer(params_list, cfg_optim)

    if os.path.exists(args.output_dirpath):
        logging.info("Stats directory exists, deleting")
        shutil.rmtree(args.output_dirpath)
    os.makedirs(args.output_dirpath)

    plateau_scheduler = lr_scheduler_custom.EarlyStoppingReduceLROnPlateau(optimizer, mode="max")
    train_stats = metadata.TrainingStats()

    best_model = None
    epoch = -1

    logging.getLogger().addHandler(logging.FileHandler(filename=os.path.join(args.output_dirpath, 'log.txt')))
    train_start_time = time.time()

    while not plateau_scheduler.is_done():
        epoch += 1
        logging.info("Epoch: {}".format(epoch))

        train_stats.export(args.output_dirpath)  # update metrics data on disk
        train_stats.plot_all_metrics(output_dirpath=args.output_dirpath)
        train_stats.add_global('batch size', cfg["dataset"]["batch_size"])

        train(
            model,
            optimizer,
            criterion,
            train_loader_sup,
            epoch,
            train_stats,
        )

        validate(model,
                val_loader,
                epoch,
                train_stats,
                criterion,
                )

        val_iou = train_stats.get_epoch('val_iou', epoch=epoch)
        plateau_scheduler.step(val_iou)

        # update global metadata stats
        train_stats.add_global('train_wall_time', train_stats.get('train_wall_time', aggregator='sum'))
        train_stats.add_global('val_wall_time', train_stats.get('val_wall_time', aggregator='sum'))
        train_stats.add_global('num_epochs_trained', epoch)

        # handle early stopping when loss converges
        if plateau_scheduler.is_equiv_to_best_epoch:
            logging.info('Updating best model with epoch: {} IoU {}'.format(epoch, val_iou * 100))
            best_model = copy.deepcopy(model)
            # update the global metrics with the best epoch
            train_stats.update_global(epoch)
            # save a state dict (weights only) version of the model
            torch.save(best_model.state_dict(), os.path.join(args.output_dirpath, 'model-state-dict.pt'))

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

def train(
    model,
    optimizer,
    criterion,
    data_loader,
    epoch,
    train_stats,
):

    model.train()

    losses = AverageMeter(10)
    batch_times = AverageMeter(10)

    start_time = time.time()
    for step, tensor_dict in enumerate(data_loader):
        batch_start = time.time()

        i_iter = epoch * len(data_loader) + step
        image, label = tensor_dict
        batch_size, h, w = label.size()

        image, label = image.cuda().float(), label.cuda()
        outs = model(image)

        pred = outs["pred"]
        pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

        loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        reduced_loss = loss.clone().detach()
        losses.update(reduced_loss.item())

        train_stats.append_accumulate('train_loss', loss.item())
        pred = torch.argmax(pred, dim=1)

        ari = ARI(pred.cpu().detach().numpy(), label.cpu().detach().numpy(), batch_size)
        accuracy = torch.mean((pred == label / label.numel()).type(torch.FloatTensor))
        train_stats.append_accumulate('ARI', ari)
        train_stats.append_accumulate('train_accuracy', accuracy.item())
        train_stats.append_accumulate('learning_rates', optimizer.param_groups[0]['lr'])

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        if i_iter % 100 == 0:
            cpu_mem_percent_used = psutil.virtual_memory().percent
            gpu_mem_percent_used, memory_total_info = get_gpu_memory()
            gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]

            logging.info(
                "Iter {}\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "LR {lr:.5f} ({lr:.7f})\t"
                "cpu_mem: {cpu_mem:2.1f}%\t"
                "gpu_mem: {gpu_mem}% of {total_mem}MiB\t"
                .format(
                    i_iter,
                    batch_time=batch_times,
                    loss=losses,
                    lr=optimizer.param_groups[0]['lr'],
                    cpu_mem=cpu_mem_percent_used,
                    gpu_mem=gpu_mem_percent_used,
                    total_mem=memory_total_info
                )
            )

    train_stats.close_accumulate(epoch, "ARI", method='avg')
    train_stats.close_accumulate(epoch, 'train_loss', method='avg')
    train_stats.close_accumulate(epoch, 'train_accuracy', method='avg')
    train_stats.close_accumulate(epoch, 'learning_rates', method='avg')
    train_stats.add(epoch, 'train_wall_time', time.time() - start_time)


def ARI(pred, label, batch_size):
    ari_accuracy = []
    for img in range(batch_size):
        pred_cpy = pred[img]
        label_cpy = label[img]
        ari_accuracy.append(adjusted_rand_score(label_cpy.flatten(), pred_cpy.flatten()))
    return np.mean(ari_accuracy)

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_total_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_used_info = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    memory_total_info = [int(x.split()[0]) for i, x in enumerate(memory_total_info)]
    memory_used_percent = np.asarray(memory_used_info) / np.asarray(memory_total_info)
    return memory_used_percent, memory_total_info


def validate(
    model,
    data_loader,
    epoch,
    train_stats,
    criterion,
):
    logging.info('Evaluating model against validation data')
    start_time = time.time()

    model.eval()

    num_classes, ignore_label = (
        cfg["net"]["num_classes"],
        cfg["dataset"]["ignore_label"],
    )

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
        output = outs["pred"]
        output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
        loss = criterion(output, labels)
        train_stats.append_accumulate('val_loss', loss.item())
        pred = torch.argmax(output, dim=1)
        accuracy = torch.mean((pred == labels / labels.numel()).type(torch.FloatTensor))
        train_stats.append_accumulate('val_accuracy', accuracy.item())

        output = output.data.max(1)[1].cpu().numpy()
        target_origin = labels.cpu().numpy()

        # start to calculate miou
        intersection, union, target = intersectionAndUnion(
            output, target_origin, num_classes, ignore_label
        )

        # gather all validation information
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)

    # close out the accumulating stats with the specified method
    train_stats.append_accumulate('val_iou', [metric for metric in iou_class])
    train_stats.close_accumulate(epoch, 'val_iou', method='avg')
    train_stats.close_accumulate(epoch, 'val_loss', method='avg')
    # this adds the avg loss to the train stats
    train_stats.close_accumulate(epoch, 'val_accuracy', method='avg')
    train_stats.add(epoch, 'val_wall_time', time.time() - start_time)

    mIoU = train_stats.get_epoch('val_iou', epoch=epoch)
    for i, iou in enumerate(iou_class):
        logging.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
    logging.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))


if __name__ == "__main__":
    main()
