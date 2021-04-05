from typing import Dict, Tuple, Any
from warmup_scheduler import GradualWarmupScheduler
import numpy as np
import pandas as pd

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
