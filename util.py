from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import torch
import os
import torch.nn.functional as F
import math

def _convert_to_numpy(list_of_str):
    return np.array(list_of_str.split(' '))

def row_wise_f1_score(labels, preds):
    scores = []

    if isinstance(labels, pd.Series):
        labels = labels.map(_convert_to_numpy)

    if isinstance(preds, pd.Series):
        preds = preds.map(_convert_to_numpy)
    
    for label, pred in zip(labels, preds):
        n = len(np.intersect1d(label, pred))
        score = 2 * n / (len(label)+len(pred))
        scores.append(score)
    return scores, np.mean(scores)


def l2_norm(input, axis=1):
    norm = torch.norm(input,2, axis, True)
    output = torch.div(input, norm)
    return output

try:
    from warmup_scheduler import GradualWarmupScheduler
    class GradualWarmupSchedulerV2(GradualWarmupScheduler):
        def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
            super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
        def get_lr(self):
            if self.last_epoch > self.total_epoch:
                if self.after_scheduler:
                    if not self.finished:
                        self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                        self.finished = True
                    return self.after_scheduler.get_lr()
                return [base_lr * self.multiplier for base_lr in self.base_lrs]
            if self.multiplier == 1.0:
                return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
            else:
                return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
except:
    class GradualWarmupSchedulerV2:
        pass


def weight_file(model_type, fold, stage, loss_type, out_dim):
    return f'{model_type}_fold{fold}_stage{stage}_{loss_type}_outdim{out_dim}.pth'


def search_weight(weight_dir, model_type, fold, stage, loss_type):
    all_ = os.listdir(weight_dir)
    for f in all_:
        if match_weight(f, model_type, fold, stage, loss_type):
            return os.path.join(weight_dir, f)

    return "none"

def match_weight(f, model_type, fold, stage, loss_type):
    m, rest = f.split("_fold")
    match_model_type = model_type in {'auto', m}
    parts = rest.split("_")
    _fold = int(parts[0].replace('fold', ''))
    _stage = int(parts[1].replace('stage', ''))
    _loss_type = parts[2]

    return match_model_type and _fold == fold and _stage == stage and _loss_type == loss_type


def scale_img(img, ratio=1.0, same_shape=False, gs=32):
    # https://github.com/ultralytics/yolov5/blob/1487bc84ff3babfb502dffb5ffbdc7e02fcb1879/utils/torch_utils.py#L247

    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def freeze_bn(model):
    count = 0
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.eval()
            count += 1

    return count
