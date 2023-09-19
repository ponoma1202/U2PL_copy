import argparse
import logging
import os
import os.path as osp
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import shutil
from pytorch_utils import lr_scheduler as lr_scheduler_custom
from pytorch_utils import metadata
import copy
import psutil
import subprocess

from u2pl.dataset.augmentation import generate_unsup_data
from u2pl.dataset.builder import get_loader
from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.loss_helper import (
    compute_contra_memobank_loss,
    compute_unsupervised_loss,
    get_criterion,
)
from u2pl.utils.lr_helper import get_optimizer
from u2pl.utils.utils import (
    AverageMeter,
    intersectionAndUnion,
    label_onehot,
    set_random_seed,
)

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
parser.add_argument("--config", type=str, default="experiments/pascal/1464/ours/config.yaml")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dirpath", type=str, default="./stats")


def main():
    global args, cfg, prototype
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        handlers=[logging.StreamHandler()])

    cfg["exp_path"] = os.path.dirname(args.config)
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    if args.seed is not None:
        print("set random seed to", args.seed)
        set_random_seed(args.seed)

    if not osp.exists(cfg["saver"]["snapshot_dir"]):
        os.makedirs(cfg["saver"]["snapshot_dir"])

    # Create network
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()
    sup_loss_fn = get_criterion(cfg)
    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed)

    # Optimizer
    cfg_optim = cfg["trainer"]["optimizer"]

    params_list = []
    for module in modules_back:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )
    for module in modules_head:
        params_list.append(
            dict(params=module.parameters(), lr=cfg_optim["kwargs"]["lr"])
        )

    optimizer = get_optimizer(params_list, cfg_optim)

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = model_teacher.cuda()

    for p in model_teacher.parameters():
        p.requires_grad = False

    if os.path.exists(args.output_dirpath):
        logging.info("{} directory exists, deleting".format(args.output_dirpath))
        shutil.rmtree(args.output_dirpath)
    os.makedirs(args.output_dirpath)

    plateau_scheduler = lr_scheduler_custom.EarlyStoppingReduceLROnPlateau(optimizer, mode="max")
    train_stats = metadata.TrainingStats()

    best_model = None
    epoch = -1

    # add the file based handler to the logger
    logging.getLogger().addHandler(logging.FileHandler(filename=os.path.join(args.output_dirpath, 'log.txt')))
    train_start_time = time.time()

    # build class-wise memory bank
    memobank = []
    queue_ptrlis = []
    queue_size = []
    for i in range(cfg["net"]["num_classes"]):
        memobank.append([torch.zeros(0, 256)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    # build prototype
    prototype = torch.zeros(
        (
            cfg["net"]["num_classes"],
            cfg["trainer"]["contrastive"]["num_queries"],
            1,
            256,
        )
    ).cuda()

    # Start to train model
    while not plateau_scheduler.is_done():
        epoch += 1
        logging.info("Epoch: {}".format(epoch))

        train_stats.export(args.output_dirpath)  # update metrics data on disk
        train_stats.plot_all_metrics(output_dirpath=args.output_dirpath)
        train_stats.add_global('batch size', cfg["dataset"]["batch_size"])

        # Training
        train(
            model,
            model_teacher,
            optimizer,
            sup_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            epoch,
            memobank,
            queue_ptrlis,
            queue_size,
            train_stats
        )

        # Validation
        if cfg["trainer"]["eval_on"]:
            logging.info("start evaluation")

            if epoch < cfg["trainer"].get("sup_only_epoch", 1):
                validate(model, val_loader, epoch, train_stats, sup_loss_fn)
            else:
                validate(model_teacher, val_loader, epoch, train_stats, sup_loss_fn)

            val_iou = train_stats.get_epoch('val_iou', epoch=epoch)
            plateau_scheduler.step(val_iou)

            # update global metadata stats
            train_stats.add_global('train_wall_time', train_stats.get('train_wall_time', aggregator='sum'))
            train_stats.add_global('val_wall_time', train_stats.get('val_wall_time', aggregator='sum'))
            train_stats.add_global('num_epochs_trained', epoch)

            # handle early stopping when loss converges
            if plateau_scheduler.is_equiv_to_best_epoch:
                logging.info('Updating best model with epoch: {} IoU: {}'.format(epoch, val_iou * 100))
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
    model_teacher,
    optimizer,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    memobank,
    queue_ptrlis,
    queue_size,
    train_stats
):
    global prototype
    ema_decay_origin = cfg["net"]["ema_decay"]

    model.train()
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    batch_times = AverageMeter(10)

    start_time = time.time()
    for step, tensor_l_dict in enumerate(loader_l):
        batch_start = time.time()
        i_iter = epoch * len(loader_l) + step

        image_l, label_l = tensor_l_dict
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        image_u, label_u = next(loader_u_iter)
        image_u = image_u.cuda()
        label_u = label_u.cuda().detach()

        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            # forward
            outs = model(image_l)
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            # supervised loss
            sup_loss = sup_loss_fn(pred, label_l)

            model_teacher.train()
            _ = model_teacher(image_l)

            unsup_loss = 0 * rep.sum()
            contra_loss = 0 * rep.sum()
        else:
            if epoch == cfg["trainer"].get("sup_only_epoch", 1):
                # copy student parameters to teacher
                with torch.no_grad():
                    for t_params, s_params in zip(
                        model_teacher.parameters(), model.parameters()
                    ):
                        t_params.data = s_params.data

            # generate pseudo labels first
            model_teacher.eval()
            pred_u_teacher = model_teacher(image_u)["pred"]
            pred_u_teacher = F.interpolate(
                pred_u_teacher, (h, w), mode="bilinear", align_corners=True
            )
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
            logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)

            accuracy = torch.mean((label_u_aug == label_u / label_u.numel()).type(torch.FloatTensor))
            train_stats.append_accumulate('teacher_pseudo_labeling_accuracy', accuracy.item())

            # apply strong data augmentation: cutout, cutmix, or classmix
            if np.random.uniform(0, 1) < 0.5 and cfg["trainer"]["unsupervised"].get(
                "apply_aug", False
            ):
                image_u_aug, label_u_aug, logits_u_aug = generate_unsup_data(
                    image_u,
                    label_u_aug.clone(),
                    logits_u_aug.clone(),
                    mode=cfg["trainer"]["unsupervised"]["apply_aug"],
                )
            else:
                image_u_aug = image_u

            # forward
            num_labeled = len(image_l)
            image_all = torch.cat((image_l, image_u_aug))
            outs = model(image_all)
            pred_all, rep_all = outs["pred"], outs["rep"]
            pred_l, pred_u = pred_all[:num_labeled], pred_all[num_labeled:]
            pred_l_large = F.interpolate(
                pred_l, size=(h, w), mode="bilinear", align_corners=True
            )
            pred_u_large = F.interpolate(
                pred_u, size=(h, w), mode="bilinear", align_corners=True
            )

            pred_student = torch.argmax(pred_u_large, dim=1)
            accuracy = torch.mean((pred_student == label_u / label_u.numel()).type(torch.FloatTensor))
            train_stats.append_accumulate('student_pseudo_labeling_accuracy', accuracy.item())

            # supervised loss
            sup_loss = sup_loss_fn(pred_l_large, label_l.clone())

            # teacher forward
            model_teacher.train()
            with torch.no_grad():
                out_t = model_teacher(image_all)
                pred_all_teacher, rep_all_teacher = out_t["pred"], out_t["rep"]
                prob_all_teacher = F.softmax(pred_all_teacher, dim=1)
                prob_l_teacher, prob_u_teacher = (
                    prob_all_teacher[:num_labeled],
                    prob_all_teacher[num_labeled:],
                )

                pred_u_teacher = pred_all_teacher[num_labeled:]
                pred_u_large_teacher = F.interpolate(
                    pred_u_teacher, size=(h, w), mode="bilinear", align_corners=True
                )

            # unsupervised loss
            drop_percent = cfg["trainer"]["unsupervised"].get("drop_percent", 100)
            percent_unreliable = (100 - drop_percent) * ((0.5) ** (epoch / 50))
            drop_percent = 100 - percent_unreliable
            unsup_loss = (
                    compute_unsupervised_loss(
                        pred_u_large,
                        label_u_aug.clone(),
                        drop_percent,
                        pred_u_large_teacher.detach(),
                    )
                    * cfg["trainer"]["unsupervised"].get("loss_weight", 1)
            )

            # contrastive loss using unreliable pseudo labels
            if cfg["trainer"].get("contrastive", False):
                cfg_contra = cfg["trainer"]["contrastive"]
                contra_flag = "{}:{}".format(
                    cfg_contra["low_rank"], cfg_contra["high_rank"]
                )
                alpha_t = cfg_contra["low_entropy_threshold"] * ((0.5) ** (epoch / 50))

                with torch.no_grad():
                    prob = torch.softmax(pred_u_large_teacher, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                    low_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(), alpha_t
                    )
                    low_entropy_mask = (
                        entropy.le(low_thresh).float() * (label_u_aug != 255).bool()
                    )

                    high_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(),
                        100 - alpha_t,
                    )
                    high_entropy_mask = (
                        entropy.ge(high_thresh).float() * (label_u_aug != 255).bool()
                    )

                    low_mask_all = torch.cat(
                        (
                            (label_l.unsqueeze(1) != 255).float(),
                            low_entropy_mask.unsqueeze(1),
                        )
                    )

                    low_mask_all = F.interpolate(
                        low_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )
                    # down sample

                    if cfg_contra.get("negative_high_entropy", True):
                        contra_flag += " high"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                    else:
                        contra_flag += " low"
                        high_mask_all = torch.cat(
                            (
                                (label_l.unsqueeze(1) != 255).float(),
                                torch.ones(logits_u_aug.shape)
                                .float()
                                .unsqueeze(1)
                                .cuda(),
                            ),
                        )
                    high_mask_all = F.interpolate(
                        high_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )  # down sample

                    # down sample and concat
                    label_l_small = F.interpolate(
                        label_onehot(label_l, cfg["net"]["num_classes"]),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )
                    label_u_small = F.interpolate(
                        label_onehot(label_u_aug, cfg["net"]["num_classes"]),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )

                if not cfg_contra.get("anchor_ema", False):
                    new_keys, contra_loss = compute_contra_memobank_loss(
                        rep_all,
                        label_l_small.long(),
                        label_u_small.long(),
                        prob_l_teacher.detach(),
                        prob_u_teacher.detach(),
                        low_mask_all,
                        high_mask_all,
                        cfg_contra,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach(),
                    )
                else:
                    prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                        rep_all,
                        label_l_small.long(),
                        label_u_small.long(),
                        prob_l_teacher.detach(),
                        prob_u_teacher.detach(),
                        low_mask_all,
                        high_mask_all,
                        cfg_contra,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach(),
                        prototype,
                    )

                contra_loss = (contra_loss/ cfg["trainer"]["contrastive"].get("loss_weight", 1))

            else:
                contra_loss = 0 * rep_all.sum()

        loss = sup_loss + unsup_loss + contra_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min(1 - 1 / (i_iter - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1) + 1), ema_decay_origin,)
                for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
                    t_params.data = (ema_decay * t_params.data + (1 - ema_decay) * s_params.data)

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        uns_losses.update(reduced_uns_loss.item())

        reduced_con_loss = contra_loss.clone().detach()
        con_losses.update(reduced_con_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        train_stats.append_accumulate('train_loss', loss.item())
        train_stats.append_accumulate('supervised_loss', reduced_sup_loss.item())
        train_stats.append_accumulate('unsupervised_loss', reduced_uns_loss.item())
        train_stats.append_accumulate('contrastive_loss', reduced_con_loss.item())
        train_stats.append_accumulate('learning_rates', optimizer.param_groups[0]['lr'])

        if i_iter % 100 == 0:
            cpu_mem_percent_used = psutil.virtual_memory().percent
            gpu_mem_percent_used, memory_total_info = get_gpu_memory()
            gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]

            logging.info(
                "Iter [{}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "Con {con_loss.val:.3f} ({con_loss.avg:.3f})\t"
                "LR {lr:.5f}\t"
                "cpu_mem: {cpu_mem:2.1f}%\t"
                "gpu_mem: {gpu_mem}% of {total_mem}MiB\t".format(
                    i_iter,
                    batch_time=batch_times,
                    sup_loss=sup_losses,
                    uns_loss=uns_losses,
                    con_loss=con_losses,
                    lr=optimizer.param_groups[0]['lr'],
                    cpu_mem=cpu_mem_percent_used,
                    gpu_mem=gpu_mem_percent_used,
                    total_mem=memory_total_info
                )
            )

    train_stats.close_accumulate(epoch, 'train_loss', method='avg')
    train_stats.close_accumulate(epoch, "supervised_loss", method='avg')
    train_stats.close_accumulate(epoch, "unsupervised_loss", method='avg')
    train_stats.close_accumulate(epoch, "contrastive_loss", method='avg')
    train_stats.close_accumulate(epoch, 'learning_rates', method='avg')

    train_stats.close_accumulate(epoch, 'teacher_pseudo_labeling_accuracy', method='avg')
    train_stats.close_accumulate(epoch, 'student_pseudo_labeling_accuracy', method='avg')
    train_stats.add(epoch, 'train_wall_time', time.time() - start_time)


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
    criterion
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

        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )

        loss = criterion(output, labels)
        train_stats.append_accumulate('val_loss', loss.item())
        pred = torch.argmax(output, dim=1)
        accuracy = torch.mean(
            (pred == labels / labels.numel()).type(torch.FloatTensor))
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