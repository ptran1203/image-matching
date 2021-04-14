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
import cupy as cp
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.backends import cudnn
from dataset import ShoppeDataset, get_df, get_transforms
from util import GradualWarmupSchedulerV2, row_wise_f1_score
from models import EffnetV2, Resnest50
from losses import ArcFaceLossAdaptiveMargin
from losses import TripletLoss
from losses import encode_config, loss_from_config, decode_config
from util import weight_file
from sklearn.preprocessing import LabelEncoder

default_loss_config = encode_config(loss_type='cos', margin=0.6, scale=30, label_smoothing=0.0, triplet=False, cls='CE')

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
    parser.add_argument('--bert', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--load-from', type=str, default='')
    parser.add_argument('--groups', type=int, default=0)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--full', action='store_true')
    
    parser.add_argument('--loss-config', type=str, default=default_loss_config)

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
    embeds = cp.array(embeds)

    preds = []
    scores = []
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
            scores.append(cts[k,][IDX])

    return preds, scores


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
    for image, input_ids, attention_mask, target in bar:

        image, input_ids, attention_mask, target = (
            image.cuda(), input_ids.cuda(),
            attention_mask.cuda(), target.cuda())

        optimizer.zero_grad()

        feat, logits_m = model(image, input_ids, attention_mask, target)
        loss = criterion(feat, logits_m, target)
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
    bar = tqdm(valid_loader)

    with torch.no_grad():
        for image, input_ids, attention_mask, target in bar:
            image, input_ids, attention_mask, target = (
                image.cuda(), input_ids.cuda(),
                attention_mask.cuda(), target.cuda())

            feat, _ = model(image, input_ids, attention_mask)
            # embeds.append(torch.cat([global_feat, local_feat], 1).detach().cpu().numpy())
            embeds.append(feat.detach().cpu().numpy())

    embeds = np.concatenate(embeds)
    preds, _ = search_similiar_images(embeds, valid_df)
    _, val_f1_score = row_wise_f1_score(valid_df.target, preds)

    return val_f1_score


def get_criterion(args, out_dim, margins):
    LossFunction = loss_from_config(args.loss_config, adaptive_margins=margins, classes=out_dim)
    loss_config = decode_config(args.loss_config)
    print(f'Loss: {LossFunction.__class__.__name__}')

    if loss_config.triplet:
        if loss_config.loss_type == 'aarc':
            def criterion(feat, logits, target):
                return LossFunction(logits, target, out_dim) + TripletLoss(0.3)(feat, target)
        else:
            def criterion(feat, logits, target):
                return LossFunction(logits, target) + TripletLoss(0.3)(feat, target)
    else:
        if loss_config.loss_type == 'aarc':
            def criterion(feat, logits, target):
                return LossFunction(logits, target, out_dim)
        else:
            def criterion(feat, logits, target):
                return LossFunction(logits, target)

    return criterion


def main(args):

    # get dataframe
    df = get_df(args.groups)

    # get adaptive margin
    tmp = np.sqrt(1 / np.sqrt(df['label_group'].value_counts().sort_index().values))
    margins = (tmp - tmp.min()) / (tmp.max() - tmp.min()) * 0.45 + 0.05

    # get augmentations
    transforms_train, transforms_val = get_transforms(args.image_size, args.stage)

    # get train and valid dataset
    df_train = df[df['fold'] != args.fold] if not args.full else df
    df_train['label_group'] =  LabelEncoder().fit_transform(df_train.label_group)

    df_valid = df[df['fold'] == args.fold]

    out_dim = df_train.label_group.nunique()
    print(f"out_dim = {out_dim}")
 
    dataset_train = ShoppeDataset(df_train, 'train', transform=transforms_train)
    dataset_valid = ShoppeDataset(df_valid, 'val', transform=transforms_val)

    print(f'Train on {len(df_train)} images, validate on {len(df_valid)} images')

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                shuffle=True, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers)

    loss_config = decode_config(args.loss_config)
    # model
    if args.enet_type == 'resnest50':
        model = Resnest50(out_dim=out_dim, loss_type=loss_config.loss_type, bert=args.bert)
    else:
        model = EffnetV2(args.enet_type, out_dim=out_dim, loss_type=loss_config.loss_type, bert=args.bert)
    model = model.cuda()

    # loss func
    criterion = get_criterion(args, out_dim, margins)

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
    model_file = os.path.join(
        args.model_dir,
        weight_file(args.kernel_type, args.fold, args.stage, loss_config.loss_type, out_dim)
    )
    for epoch in range(args.start_from_epoch, args.n_epochs+1):

        print(time.ctime(), f'Epoch: {epoch}/{args.n_epochs}')
        scheduler_warmup.step(epoch - 1)

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
    set_seed(0)
    main(args)
