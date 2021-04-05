import os
import cv2
import math
import time
import gc
import pickle
import random
import argparse
import albumentations
import numpy as np
import pandas as pd
from tqdm import tqdm
# from tqdm import tqdm_notebook
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
from util import GradualWarmupSchedulerV2, row_wise_f1_score
from models import Effnet, RexNet20, ResNest101, Swish_module
from lossses import ArcFaceLossAdaptiveMargin, TripletLoss

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='/raid/GLD2')
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
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--warmup-epochs', type=int, default=1)

    args, _ = parser.parse_known_args()
    return args


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def search_similiar_images(embeds, test_df, thr=0.5):
    import cupy as cp

    embeds = cp.array(embeds)

    preds = []
    CHUNK = 1024 * 4

    CTS = len(embeds) // CHUNK

    if len(embeds) % CHUNK!=0:
        CTS += 1

    for j in range(CTS):

        a = j * CHUNK
        b = (j + 1) * CHUNK
        b = min(b,len(embeds))
    
        cts = cp.matmul(embeds, embeds[a:b].T).T
        
        for k in range(b-a):
            # print(sorted(cts[k,], reverse=True))
            IDX = cp.where(cts[k,] > thr)[0]
            o = test_df.iloc[cp.asnumpy(IDX)].posting_id.values
            preds.append(o)

    return preds


def generate_embeddings(model, test_loader):
    embeds = []

    with torch.no_grad():
        for img in tqdm(test_loader): 
            img = img.cuda()
            feat, _ = model(img)

            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)
        
    _ = gc.collect()
    image_embeddings = np.concatenate(embeds)
    return image_embeddings


def train_epoch(model, loader, optimizer, criterion):

    model.train()
    train_loss = []
    accs = []

    bar = tqdm(loader)
    for (data, target) in bar:

        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()

        feat, logits_m = model(data)
        loss = criterion(logits_m, feat, target)
        loss.backward()
        optimizer.step()

        lmax_m = logits_m.max(1)
        preds_m = lmax_m.indices

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)

        acc_m = (preds_m.detach().cpu().numpy() == target.detach().cpu().numpy()).mean() * 100
        accs.append(acc_m)
        bar.set_description('loss: %.5f, smth: %.5f, acc: %.5f' % (loss_np, smooth_loss, acc_m))

    return train_loss, accs

def val_epoch(model, valid_loader, criterion, valid_df):

    model.eval()
    embeds = []

    with torch.no_grad():
        for (data, target) in tqdm(valid_loader):
            data, target = data.cuda(), target.cuda()

            feat, _ = model(data)

            embeds.append(feat.detach().cpu().numpy())

    embeds = np.concatenate(embeds)
    preds = search_similiar_images(embeds, valid_df)
    _, val_f1_score = row_wise_f1_score(valid_df.target, preds)
    if val_f1_score == 0.0:
        print('val_f1_score', val_f1_score)
        valid_df_clone = valid_df.copy()
        valid_df_clone[['target', 'preds']].to_csv('/content/valid.csv', index=False)
        raise('Valid 0')

    return val_f1_score


def main(args):

    # get dataframe
    df, out_dim = get_df(args.groups)
    print(list(df.columns.values))
    print(df.head())
    print(f"out_dim = {out_dim}")

    # get adaptive margin
    tmp = np.sqrt(1 / np.sqrt(df['label_group'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    # get augmentations
    transforms_train, transforms_val = get_transforms(args.image_size, args.stage)

    # get train and valid dataset
    df_train = df[df['fold'] != args.fold]
    df_valid = df[df['fold'] == args.fold]

    dataset_train = ShoppeDataset(df_train, 'train', transform=transforms_train)
    dataset_valid = ShoppeDataset(df_valid, 'val', transform=transforms_val)

    print(f'Train on {len(df_train)} images, validate on {len(df_valid)} images')

    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    # model
    model = ModelClass(args.enet_type, out_dim=out_dim)
    model = model.cuda()

    # loss func
    def criterion(logits_m, feat, target):
        arc = ArcFaceLossAdaptiveMargin(margins=margins, s=80)
        loss_m = arc(logits_m, target, out_dim)
        triplet_loss = TripletLoss(0.3)(feat, target)
        return loss_m + triplet_loss

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)


    # load pretrained
    if args.load_from and args.load_from != 'none':
        checkpoint = torch.load(args.load_from,  map_location='cuda:0')
        state_dict = checkpoint['model_state_dict']
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}    
        model.load_state_dict(state_dict, strict=True)
        del checkpoint, state_dict
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Loaded weight from {args.load_from}")

    # lr scheduler
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.n_epochs-1)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=args.warmup_epochs, after_scheduler=scheduler_cosine)

    # train & valid loop
    best_score = -1
    model_file = os.path.join(args.model_dir, f'{args.kernel_type}_fold{args.fold}_stage{args.stage}.pth')
    for epoch in range(args.start_from_epoch, args.n_epochs+1):

        print(time.ctime(), 'Epoch:', epoch)
        scheduler_warmup.step(epoch - 1)

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                                  shuffle=True, drop_last=True)

        train_loss, acc_list = train_epoch(model, train_loader, optimizer, criterion)
        f1score = val_epoch(model, valid_loader, criterion, df_valid)

        content = time.ctime() + ' ' + \
            (
                f'Fold {args.fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f},'
                f' train acc {np.mean(acc_list):.5f}, f1score: {(f1score):.6f}.')

        print(content)
        with open(os.path.join(args.log_dir, f'{args.kernel_type}.txt'), 'a') as appender:
            appender.write(content + '\n')

        if f1score > best_score:
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
    }, model_file)


if __name__ == '__main__':

    args = parse_args()
    print(args)
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