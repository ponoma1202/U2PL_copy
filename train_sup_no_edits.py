import argparse
import logging
import os
import os.path as osp
import pprint
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
#import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tensorboardX import SummaryWriter

from pytorch_utils import lr_scheduler as lr_scheduler_custom   # For Mike's code
from pytorch_utils import metadata                              # For Mike's code
import copy                                                     # For Mike's code
import psutil                                                     # For Mike's code
import subprocess                                                     # For Mike's code
import shutil                                                   # For Mike's code

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

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="config.yaml")
#parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dirpath", type=str, default="./stats")
#parser.add_argument("--port", default=None, type=int)

#logger = init_log("global", logging.INFO)
#logger.propagate = 0
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
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    # local_rank = int(os.environ["LOCAL_RANK"])
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model,
    #     device_ids=[local_rank],
    #     output_device=local_rank,
    #     find_unused_parameters=False,
    # )

    criterion = get_criterion(cfg)

    train_loader_sup, val_loader = get_loader(cfg, seed=seed)

    # Optimizer and lr decay scheduler
    cfg_trainer = cfg["trainer"]
    cfg_optim = cfg_trainer["optimizer"]
    times = 10 if "pascal" in cfg["dataset"]["type"] else 1

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"] * times)
        )

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
        load_state(cfg["saver"]["pretrain"], model, key="model_state")

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


    # start Mike's code
    while not plateau_scheduler.is_done():
        epoch += 1
        logging.info("Epoch: {}".format(epoch))

        train_stats.export(args.output_dirpath)  # update metrics data on disk
        train_stats.plot_all_metrics(output_dirpath=args.output_dirpath)

        train_stats.add_global('batch size', cfg["dataset"]["batch_size"])

        # end Mike's code
        # Training
        train(
            model,
            optimizer,
            plateau_scheduler,
            criterion,
            train_loader_sup,
            epoch,
            #tb_logger,
            train_stats
        )

        # Validation and store checkpoint
        prec = validate(model, val_loader, epoch, train_stats, criterion)

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
                    state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth")
                )

            torch.save(model.state_dict(), osp.join(cfg["saver"]["snapshot_dir"], "model-state-dict.pth"))

            # logger.info(
            #     "\033[31m * Currently, the best val result is: {:.2f}\033[0m".format(
            #         best_prec * 100
            #     )
            # )
            # tb_logger.add_scalar("mIoU val", prec, epoch)

            # start Mike's code

            # val_loss = train_stats.get_epoch('val_loss', epoch=epoch) - commented out in Mike's code
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
    train_stats
):
    model.train()

    #data_loader.sampler.set_epoch(epoch)
    data_loader_iter = iter(data_loader)

    #rank, world_size = dist.get_rank(), dist.get_world_size()
    rank = 0

    losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    # start Mike's code
    start_time = time.time()
    per_class_accuracy = []
    # end Mike's code

    batch_end = time.time()
    for step, tensor_dict in enumerate(data_loader):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(data_loader) + step
        # lr = lr_scheduler.get_lr()
        # learning_rates.update(lr[0])
        # lr_scheduler.step()

        image, label = tensor_dict
        batch_size, h, w = label.size()
        image, label = image.cuda(), label.cuda()
        outs = model(image)
        pred = outs["pred"]
        pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

        if "aux_loss" in cfg["net"].keys():
            aux = outs["aux"]
            aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
            loss = criterion([pred, aux], label)
        else:
            loss = criterion(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # gather all loss from different gpus
        reduced_loss = loss.clone().detach()
        #dist.all_reduce(reduced_loss)
        losses.update(reduced_loss.item())

        # start Mike's code
        train_stats.append_accumulate('train_loss', loss.item())
        pred = torch.argmax(pred, dim=1)

        accuracy = torch.mean((pred == label / label.numel()).type(torch.FloatTensor))
        train_stats.append_accumulate('train_accuracy', accuracy.item())
        train_stats.append_accumulate('learning_rates', optimizer.param_groups[0]['lr'])
        # end Mike's code

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
                "LR {lr.val:.5f} ({lr.avg:.5f})\t"
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

            #tb_logger.add_scalar("lr", learning_rates.avg, i_iter)
            #tb_logger.add_scalar("Loss", losses.avg, i_iter)

    # start Mike's code
    train_stats.close_accumulate(epoch, 'train_loss', method='avg')  # this adds the avg loss to the train stats
    train_stats.close_accumulate(epoch, 'train_accuracy', method='avg')
    train_stats.close_accumulate(epoch, 'learning_rates', method='avg')
    train_stats.add(epoch, 'train_wall_time', time.time() - start_time)


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
    train_stats,
    criterion
):
    # start Mike's code
    logging.info('Evaluating model against validation data')
    start_time = time.time()
    # end Mike's code

    model.eval()
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

    for i, iou in enumerate(iou_class):
        logging.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
    logging.info(" * epoch {} mIoU {:.2f}".format(epoch, mIoU * 100))

    return mIoU


if __name__ == "__main__":
    main()
