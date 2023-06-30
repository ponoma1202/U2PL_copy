import argparse
import copy
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
import shutil                                                   # For Mike's code
from pytorch_utils import lr_scheduler as lr_scheduler_custom   # For Mike's code
from pytorch_utils import metadata                              # For Mike's code
import copy                                                     # For Mike's code
import psutil                                                     # For Mike's code
import subprocess                                                     # For Mike's code

from u2pl.dataset.augmentation import generate_unsup_data
from u2pl.dataset.builder import get_loader
from u2pl.models.model_helper import ModelBuilder
#from u2pl.utils.dist_helper import setup_distributed
from u2pl.utils.loss_helper import (
    compute_contra_memobank_loss,
    compute_unsupervised_loss,
    get_criterion,
)
from u2pl.utils.lr_helper import get_optimizer, get_scheduler
from u2pl.utils.utils import (
    AverageMeter,
    get_rank,
    get_world_size,
    init_log,
    intersectionAndUnion,
    label_onehot,
    load_state,
    set_random_seed,
)

parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
#Changed config.yaml path. Was referencing ~/U2PL/config.yaml previously.
parser.add_argument("--config", type=str, default="experiments/pascal/1464/ours/config.yaml")
#parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
#parser.add_argument("--port", default=None, type=int)
parser.add_argument("--output_dirpath", type=str, default="./stats")


def main():
    global args, cfg, prototype
    args = parser.parse_args()
    seed = args.seed
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    #logger = init_log("global", logging.INFO)
    #logger.propagate = 0
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",  # Mike's logger
                        handlers=[logging.StreamHandler()])

    cfg["exp_path"] = os.path.dirname(args.config)
    cfg["save_path"] = os.path.join(cfg["exp_path"], cfg["saver"]["snapshot_dir"])

    cudnn.enabled = True
    cudnn.benchmark = True

    #rank, word_size = setup_distributed(port=args.port)

    rank = 0
    # if rank == 0:
    #     logging.info("{}".format(pprint.pformat(cfg)))
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

    # Create network
    model = ModelBuilder(cfg["net"])
    modules_back = [model.encoder]
    if cfg["net"].get("aux_loss", False):
        modules_head = [model.auxor, model.decoder]
    else:
        modules_head = [model.decoder]

    if cfg["net"].get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.cuda()

    sup_loss_fn = get_criterion(cfg)

    train_loader_sup, train_loader_unsup, val_loader = get_loader(cfg, seed=seed)

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

    #local_rank = int(os.environ["LOCAL_RANK"])
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model,
    #     device_ids=[local_rank],
    #     output_device=local_rank,
    #     find_unused_parameters=False,
    # )

    # Teacher model
    model_teacher = ModelBuilder(cfg["net"])
    model_teacher = model_teacher.cuda()
    # model_teacher = torch.nn.parallel.DistributedDataParallel(
    #     model_teacher,
    #     device_ids=[local_rank],
    #     output_device=local_rank,
    #     find_unused_parameters=False,
    # )

    for p in model_teacher.parameters():
        p.requires_grad = False

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
            _, _ = load_state(
                lastest_model, model_teacher, optimizer=optimizer, key="teacher_state"
            )

    elif cfg["saver"].get("pretrain", False):
        load_state(cfg["saver"]["pretrain"], model, key="model_state")
        load_state(cfg["saver"]["pretrain"], model_teacher, key="teacher_state")

    optimizer_start = get_optimizer(params_list, cfg_optim)
    lr_scheduler = get_scheduler(
        cfg_trainer, len(train_loader_sup), optimizer_start, start_epoch=last_epoch
    )

    # start Mike's code
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

    # end Mike's code

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
    #for epoch in range(last_epoch, cfg_trainer["epochs"]):

    # start Mike's code
    while not plateau_scheduler.is_done():
        epoch += 1
        logging.info("Epoch: {}".format(epoch))

        train_stats.export(args.output_dirpath)  # update metrics data on disk
        train_stats.plot_all_metrics(output_dirpath=args.output_dirpath)

        # end Mike's code

        # TODO: Replace lr_scheduler with Mike's plateu_scheduler to test with his code

        # Training
        train(
            model,
            model_teacher,
            optimizer,
            plateau_scheduler,
            #lr_scheduler,
            sup_loss_fn,
            train_loader_sup,
            train_loader_unsup,
            epoch,
            #tb_logger,
            #logger,
            memobank,
            queue_ptrlis,
            queue_size,
            train_stats             # For Mike's code
        )

        # Validation
        if cfg_trainer["eval_on"]:
            if rank == 0:
                logging.info("start evaluation")

            if epoch < cfg["trainer"].get("sup_only_epoch", 1):
                prec = validate(model, val_loader, epoch, train_stats, sup_loss_fn)
            else:
                prec = validate(model_teacher, val_loader, epoch, train_stats, sup_loss_fn)

            if rank == 0:
                state = {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "teacher_state": model_teacher.state_dict(),
                    "best_miou": best_prec,
                }
                if prec > best_prec:
                    best_prec = prec
                    torch.save(
                        state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt_best.pth")
                    )

                torch.save(state, osp.join(cfg["saver"]["snapshot_dir"], "ckpt.pth"))

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
    model_teacher,
    optimizer,
    lr_scheduler,
    sup_loss_fn,
    loader_l,
    loader_u,
    epoch,
    #tb_logger,
    #logger,
    memobank,
    queue_ptrlis,
    queue_size,
    train_stats                 # For Mike's code
):
    global prototype
    ema_decay_origin = cfg["net"]["ema_decay"]

    model.train()

    #loader_l.sampler.set_epoch(epoch)
    #loader_u.sampler.set_epoch(epoch)
    loader_l_iter = iter(loader_l)
    loader_u_iter = iter(loader_u)
    assert len(loader_l) == len(
        loader_u
    ), f"labeled data {len(loader_l)} unlabeled data {len(loader_u)}, imbalance!"

    #rank, world_size = dist.get_rank(), dist.get_world_size()

    sup_losses = AverageMeter(10)
    uns_losses = AverageMeter(10)
    con_losses = AverageMeter(10)
    data_times = AverageMeter(10)
    batch_times = AverageMeter(10)
    learning_rates = AverageMeter(10)

    # start Mike's code
    start_time = time.time()
    # end Mike's code

    batch_end = time.time()
    for step, tensor_l_dict in enumerate(loader_l): #range(len(loader_l)):
        batch_start = time.time()
        data_times.update(batch_start - batch_end)

        i_iter = epoch * len(loader_l) + step
        #lr = lr_scheduler.get_lr()             # For Mike's code
        #learning_rates.update(lr[0])           # For Mike's code
        #lr_scheduler.step()                    # For Mike's code

        image_l, label_l = tensor_l_dict #loader_l_iter.next()
        batch_size, h, w = label_l.size()
        image_l, label_l = image_l.cuda(), label_l.cuda()

        image_u, label_u = next(loader_u_iter) #loader_u_iter.next()             # Get label for "unlabeled" image for accuracy calculation - VP
        image_u, label_u = image_u.cuda(), label_u.cuda()
        # For Mike's code

        if epoch < cfg["trainer"].get("sup_only_epoch", 1):
            contra_flag = "none"
            # forward
            outs = model(image_l)
            pred, rep = outs["pred"], outs["rep"]
            pred = F.interpolate(pred, (h, w), mode="bilinear", align_corners=True)

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred, aux], label_l)
            else:
                sup_loss = sup_loss_fn(pred, label_l)

            model_teacher.train()
            teacher_out = model_teacher(image_l)

            # start Mike's code
            # Teacher's pseudo-labels vs ground truth in supervised environment
            # pred_teach = teacher_out["pred"]
            # pred_teach = F.interpolate(pred_teach, (h, w), mode="bilinear", align_corners=True)
            #
            # pred_teach = torch.argmax(pred_teach, dim=1)
            # accuracy = torch.mean((pred_teach == label_l).type(torch.FloatTensor))
            # train_stats.append_accumulate('teacher_sup_accuracy', accuracy.item())
            # end Mike's code

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
                pred_u_teacher, (h, w), mode="bilinear", align_corners=True)
            pred_u_teacher = F.softmax(pred_u_teacher, dim=1)
            logits_u_aug, label_u_aug = torch.max(pred_u_teacher, dim=1)            # second argument of torch.max is argmax

            # start Mike's code
            # TODO Assess Teacher's pseudo-labeling accuracy
            # Teacher's pseudo-labels vs ground truth in unsupervised environment
            accuracy = torch.mean((label_u_aug == label_u).type(torch.FloatTensor))
            train_stats.append_accumulate('teacher_pseudo_labeling_accuracy', accuracy.item())
            # end Mike's code

            # apply strong data augmentation: cutout, cutmix, or classmix
            if np.random.uniform(0, 1) < 0.5 and cfg["trainer"]["unsupervised"].get(
                "apply_aug", False
            ):
                image_u_aug, label_u_aug, logits_u_aug, label_u = generate_unsup_data(           # changed to label_u_all for accuracy calculation
                    image_u,
                    label_u_aug.clone(),            # changed from label_u_aug => label_u_all
                    logits_u_aug.clone(),
                    label_u,
                    mode=cfg["trainer"]["unsupervised"]["apply_aug"]
                )
                # _, label_u, _ = generate_unsup_data(
                #     # changed to label_u_all for accuracy calculation
                #     image_u,
                #     label_u,  # changed from label_u_aug => label_u_all
                #     logits_u_aug.clone(),
                #     mode=cfg["trainer"]["unsupervised"]["apply_aug"],
                # )
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

            # TODO: Add Pseudo-label student accuracy (student labels vs teacher labels)
            # start Mike's code
            # accuracy = torch.mean((pred_student == label_u_aug).type(torch.FloatTensor))
            # train_stats.append_accumulate('student_vs_teacher_accuracy', accuracy.item())

            # TODO: Add student vs ground truth accuracy
            pred_student = torch.argmax(pred_u_large, dim=1)
            accuracy = torch.mean((pred_student == label_u).type(torch.FloatTensor))
            train_stats.append_accumulate('student_pseudo_labeling_accuracy', accuracy.item())

            # end Mike's code

            # supervised loss
            if "aux_loss" in cfg["net"].keys():
                aux = outs["aux"][:num_labeled]
                aux = F.interpolate(aux, (h, w), mode="bilinear", align_corners=True)
                sup_loss = sup_loss_fn([pred_l_large, aux], label_l.clone())
            else:
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
            percent_unreliable = (100 - drop_percent) * (1 - epoch / cfg["trainer"]["epochs"])
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
            contra_flag = "none"
            if cfg["trainer"].get("contrastive", False):
                cfg_contra = cfg["trainer"]["contrastive"]
                contra_flag = "{}:{}".format(
                    cfg_contra["low_rank"], cfg_contra["high_rank"]
                )
                alpha_t = cfg_contra["low_entropy_threshold"] * (
                    1 - epoch / cfg["trainer"]["epochs"]
                )

                with torch.no_grad():
                    prob = torch.softmax(pred_u_large_teacher, dim=1)
                    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                    # For RELIABLE pixels - VP
                    # Create mask that captures the percentile less than alpha_t (starts at 20% and goes down) - VP
                    low_thresh = np.percentile(
                        entropy[label_u_aug != 255].cpu().numpy().flatten(), alpha_t
                    )
                    low_entropy_mask = (
                        entropy.le(low_thresh).float() * (label_u_aug != 255).bool()
                    )

                    # Create mask that captures the percentile greater than (100 - alpha_t) (starts at 80% and goes up) - VP
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

                if cfg_contra.get("binary", False):
                    contra_flag += " BCE"
                    contra_loss = compute_binary_memobank_loss(
                        rep_all,
                        torch.cat((label_l_small, label_u_small)).long(),
                        low_mask_all,
                        high_mask_all,
                        prob_all_teacher.detach(),
                        cfg_contra,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        rep_all_teacher.detach(),
                    )
                else:
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

           #     dist.all_reduce(contra_loss)
                world_size = 1
                contra_loss = (
                    contra_loss
                    / world_size
                    * cfg["trainer"]["contrastive"].get("loss_weight", 1)
                )

            else:
                contra_loss = 0 * rep_all.sum()

        loss = sup_loss + unsup_loss + contra_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update teacher model with EMA
        if epoch >= cfg["trainer"].get("sup_only_epoch", 1):
            with torch.no_grad():
                ema_decay = min(1 - 1 / (i_iter - len(loader_l) * cfg["trainer"].get("sup_only_epoch", 1) + 1), ema_decay_origin, )
                for t_params, s_params in zip(model_teacher.parameters(), model.parameters()):
                    t_params.data = (ema_decay * t_params.data + (1 - ema_decay) * s_params.data)

        # gather all loss from different gpus
        reduced_sup_loss = sup_loss.clone().detach()
        #dist.all_reduce(reduced_sup_loss)
        sup_losses.update(reduced_sup_loss.item())

        reduced_uns_loss = unsup_loss.clone().detach()
        #dist.all_reduce(reduced_uns_loss)
        uns_losses.update(reduced_uns_loss.item())

        reduced_con_loss = contra_loss.clone().detach()
        #dist.all_reduce(reduced_con_loss)
        con_losses.update(reduced_con_loss.item())

        batch_end = time.time()
        batch_times.update(batch_end - batch_start)

        # start Mike's code
        train_stats.append_accumulate('train_loss', loss.item())
        train_stats.append_accumulate('supervised_loss', reduced_sup_loss.item())
        train_stats.append_accumulate('unsupervised_loss', reduced_uns_loss.item())
        train_stats.append_accumulate('contrastive_loss', reduced_con_loss.item())
        train_stats.append_accumulate('learning_rates', optimizer.param_groups[0]['lr'])
        # end Mike's code

        rank = 0
        if i_iter % 100 == 0 and rank == 0:
            # start Mike's code
            cpu_mem_percent_used = psutil.virtual_memory().percent
            gpu_mem_percent_used, memory_total_info = get_gpu_memory()
            gpu_mem_percent_used = [np.round(100 * x, 1) for x in gpu_mem_percent_used]
            # end Mike's code

            logging.info(
                "[{}][{}] "
                "Iter [{}/{}]\t"
                #"Data {data_time.val:.2f} ({data_time.avg:.2f})\t"
                #"Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t"
                "Sup {sup_loss.val:.3f} ({sup_loss.avg:.3f})\t"
                "Uns {uns_loss.val:.3f} ({uns_loss.avg:.3f})\t"
                "Con {con_loss.val:.3f} ({con_loss.avg:.3f})\t"
                "LR {lr:.5f}\t"
                "cpu_mem: {cpu_mem:2.1f}%\t"
                "gpu_mem: {gpu_mem}% of {total_mem}MiB\t".format(
                    cfg["dataset"]["n_sup"],
                    contra_flag,
                    i_iter,
                    cfg["trainer"]["epochs"] * len(loader_l),
                    data_time=data_times,
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

            # tb_logger.add_scalar("lr", learning_rates.val, i_iter)
            # tb_logger.add_scalar("Sup Loss", sup_losses.val, i_iter)
            # tb_logger.add_scalar("Uns Loss", uns_losses.val, i_iter)
            # tb_logger.add_scalar("Con Loss", con_losses.val, i_iter)

    # start Mike's code
    train_stats.close_accumulate(epoch, 'train_loss', method='avg')  # this adds the avg loss to the train stats
    train_stats.close_accumulate(epoch, "supervised_loss", method='avg')
    train_stats.close_accumulate(epoch, "unsupervised_loss", method='avg')
    train_stats.close_accumulate(epoch, "contrastive_loss", method='avg')
    train_stats.close_accumulate(epoch, 'learning_rates', method='avg')
    #train_stats.close_accumulate(epoch, 'teacher_sup_accuracy', method='avg')
    train_stats.close_accumulate(epoch, 'teacher_pseudo_labeling_accuracy', method='avg')
    # train_stats.close_accumulate(epoch, 'student_vs_teacher_accuracy',method='avg')
    train_stats.close_accumulate(epoch,'student_pseudo_labeling_accuracy', method='avg')
    train_stats.add(epoch, 'train_wall_time', time.time() - start_time)
    # end Mike's code

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
    #logger,
    train_stats,         # for Mike's code
    criterion            # for Mike's code
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

        with torch.no_grad():
            outs = model(images)

        # get the output produced by model_teacher
        output = outs["pred"]
        output = F.interpolate(
            output, labels.shape[1:], mode="bilinear", align_corners=True
        )

        # start Mike's code
        loss = criterion(output, labels)
        train_stats.append_accumulate('val_loss', loss.item())
        pred = torch.argmax(output, dim=1)
        accuracy = torch.mean((pred == labels).type(torch.FloatTensor))
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
