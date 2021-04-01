import os
import cv2
import math
import time
import pickle
import random
import argparse
import albumentations
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.backends import cudnn

from dataset import ShoppeDataset, get_df, get_transforms
from util import f1_score, GradualWarmupSchedulerV2
from models import DenseCrossEntropy, Swish_module
from models import ArcFaceLossAdaptiveMargin, Effnet, RexNet20, ResNest101


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/GLD2')
    parser.add_argument('--train-step', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--init-lr', type=float, default=1e-4)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--start-from-epoch', type=int, default=1)
    parser.add_argument('--stop-at-epoch', type=int, default=999)
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--load-from', type=str, default='')
    parser.add_argument('--groups', type=int, default=0)

    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, criterion):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, target) in bar:

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        _, logits_m = model(data)
        loss = criterion(logits_m, target)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
            
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))

    return train_loss

def val_epoch(model, valid_loader, criterion):

    model.eval()
    val_loss = []
    PRODS_M = []
    PREDS_M = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(valid_loader):
            data, target = data.cuda(), target.cuda()

            feat, logits_m = model(data)

            lmax_m = logits_m.max(1)
            probs_m = lmax_m.values
            preds_m = lmax_m.indices

            PRODS_M.append(probs_m.detach().cpu())
            PREDS_M.append(preds_m.detach().cpu())
            TARGETS.append(target.detach().cpu())

            loss = criterion(logits_m, target)
            val_loss.append(loss.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        PRODS_M = torch.cat(PRODS_M).numpy()
        PREDS_M = torch.cat(PREDS_M).numpy()
        TARGETS = torch.cat(TARGETS)

    acc_m = (PREDS_M == TARGETS.numpy()).mean() * 100.
    y_true = {idx: target if target >=0 else None for idx, target in enumerate(TARGETS)}
    y_pred_m = {idx: (pred_cls, conf) for idx, (pred_cls, conf) in enumerate(zip(PREDS_M, PRODS_M))}
    val_f1_score = f1_score(y_true, y_pred_m)
    return val_loss, acc_m, val_f1_score


def main(args):

    # get dataframe
    df, out_dim = get_df(args.groups)
    print(f"out_dim = {out_dim}")

    # get adaptive margin
    tmp = np.sqrt(1 / np.sqrt(df['label_group'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    # get augmentations
    transforms_train, transforms_val = get_transforms(args.image_size)

    # get train and valid dataset
    df_train = df[df['fold'] != args.fold]
    df_valid = df[df['fold'] == args.fold].reset_index(drop=True)

    dataset_train = ShoppeDataset(df_train, 'train', transform=transforms_train)
    dataset_valid = ShoppeDataset(df_valid, 'val', transform=transforms_val)

    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    # model
    model = ModelClass(args.enet_type, out_dim=out_dim)
    model = model.cuda()

    # loss func
    def criterion(logits_m, target):
        arc = ArcFaceLossAdaptiveMargin(margins=margins, s=80)
        loss_m = arc(logits_m, target, out_dim)
        return loss_m

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)


    # load pretrained
    if len(args.load_from) > 0:
        checkpoint = torch.load(args.load_from,  map_location='cuda:0')
        state_dict = checkpoint['model_state_dict']
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}    
        if args.train_step==1: 
            del state_dict['metric_classify.weight']
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=True)        
        del checkpoint, state_dict
        torch.cuda.empty_cache()
        import gc
        gc.collect()   
    
    # lr scheduler
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs-1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    # train & valid loop
    best_score = 0.
    model_file = os.path.join(args.model_dir, f'{args.kernel_type}_fold{args.fold}.pth')
    for epoch in range(args.start_from_epoch, args.n_epochs+1):

        print(time.ctime(), 'Epoch:', epoch)
        scheduler_warmup.step(epoch - 1)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  shuffle=True, drop_last=True)        

        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, acc_m, f1score = val_epoch(model, valid_loader, criterion)

        content = time.ctime() + ' ' + f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, valid loss: {(val_loss):.5f}, acc_m: {(acc_m):.6f}, gap_m: {(gap_m):.6f}.'
        print(content)
        with open(os.path.join(args.log_dir, f'{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        print('best f1 score ({:.6f} --> {:.6f}). Saving model ...'.format(best_score, f1score))
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, model_file)            
        best_score = f1score

        if epoch == args.stop_at_epoch:
            print(time.ctime(), 'Training Finished!')
            break

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.model_dir, f'{args.kernel_type}_fold{args.fold}_final.pth'))


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.enet_type == 'nest101':
        ModelClass = ResNest101
    elif args.enet_type == 'rex20':
        ModelClass = RexNet20
    else:
        ModelClass = Effnet

    set_seed(0)
    main(args)