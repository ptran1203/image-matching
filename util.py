from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import torch
import os

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
    expect = weight_file(model_type, fold, stage, loss_type, 0).split("_outdim")[0]
    for f in all_:
        if f.startswith(expect):
            return os.path.join(weight_dir, f)

    return "none"
