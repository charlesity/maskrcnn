import numpy as np

import math
import sys
import time

import torch
import utils
from matplotlib import pylab as plt

from sklearn.metrics import precision_score, jaccard_score

import warnings

warnings.filterwarnings("ignore")


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )


    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger



def color_masks(mask, threshold):
    color_spectrum = np.random.randint(0, 256, size=len(mask))
    clk_mask = np.zeros(mask[0].shape, dtype=np.int16)
    for idx, eachPredMask in enumerate(mask):
        eachPredMask[eachPredMask > threshold] = color_spectrum[idx]
        clk_mask = np.maximum(eachPredMask.astype(np.int16), clk_mask)
    return clk_mask


@torch.inference_mode()
def evaluate_metric(model, data_loader, device, score_threshold=0.10, mask_threshold =0.50):
    cpu_device = torch.device("cpu")
    model.eval()
    metric = {'precision': [], 'iou': []}
    for images, targets in (data_loader):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        true_targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]

        # process the batches of predictions
        for o_idx, pred in enumerate(outputs):
            aPredMask = pred['masks'].detach().cpu().numpy()
            aTrueMask = true_targets[o_idx]['masks'].detach().cpu().numpy()
            scores = pred['scores'].detach().cpu().numpy()

            aPredMask= aPredMask[scores>score_threshold]
            if not len(aPredMask)>0:
                continue
            input_img = images[o_idx].detach().cpu().numpy()
            clk_pred_mask =color_masks(aPredMask.squeeze(), mask_threshold)
            clk_true_mask = color_masks(aTrueMask.squeeze(), mask_threshold)

            clk_pred_mask[clk_pred_mask>0] = 1
            clk_true_mask[clk_true_mask>0] = 1
            # plt.subplot(1,2,1)
            # plt.imshow(clk_pred_mask)
            # plt.subplot(1,2,2)
            # plt.imshow(clk_true_mask)
            # plt.show()
            try:
                p = precision_score(clk_pred_mask.flatten(), clk_true_mask.flatten(),average="micro")
                metric['precision'].append(p)
            except:
                pass
            try:
                iou = jaccard_score(clk_pred_mask, clk_true_mask, average="micro")
                metric['iou'].append(iou)
            except:
               pass

    mP = np.mean(metric['precision']) if len(metric['precision'])>0 else 'Nan'
    mIou = np.mean(metric['iou']) if len(metric['iou'])>0 else 'Nan'
    return (mP, mIou)


@torch.inference_mode()
def evaluate(model, data_loader, device, score_threshold=0.10, mask_threshold =0.50):
    cpu_device = torch.device("cpu")
    model.eval()
    for images, targets in (data_loader):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        true_targets = [{k: v.to(cpu_device) for k, v in t.items()} for t in targets]

        for o_idx, pred in enumerate(outputs):
            aPredMask = pred['masks'].detach().cpu().numpy()
            aTrueMask = true_targets[o_idx]['masks'].detach().cpu().numpy()
            scores = pred['scores'].detach().cpu().numpy()
            aPredMask= aPredMask[scores>score_threshold]

            input_img = images[o_idx].detach().cpu().numpy()
            clk_pred_mask =color_masks(aPredMask.squeeze(), mask_threshold)
            clk_true_mask = color_masks(aTrueMask.squeeze(), mask_threshold)

            if np.random.binomial(1, .1):
                plt.subplot(1,3,1)
                plt.imshow(input_img.transpose((1,2,0)))
                plt.xticks([])
                plt.yticks([])

                plt.subplot(1,3,2)
                plt.imshow(clk_pred_mask)
                plt.xticks([])
                plt.yticks([])

                plt.subplot(1,3,3)
                plt.imshow(clk_true_mask)
                plt.xticks([])
                plt.yticks([])
                plt.show()
                print()

