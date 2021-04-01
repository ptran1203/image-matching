import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset

DATA_DIR = '/content'

class ShoppeDataset(Dataset):
    def __init__(self, csv, mode='train', transform=None):

        self.csv = csv.reset_index()
        self.mode = mode
        self.transform = transform
        self.img_dir = 'train_images'

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = cv2.imread(os.path.join(DATA_DIR, self.img_dir, row.image))[:,:,::-1]

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)
        if self.mode == 'test':
            return torch.tensor(image)
        else:
            return torch.tensor(image), torch.tensor(row.label_group)


def get_transforms(image_size):
    max_size_cutout = int(image_size * 0.1)
    transforms_train = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.JpegCompression(quality_lower=99, quality_upper=100),
        albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
        albumentations.MedianBlur(blur_limit=(3, 5), p=0.9),
        albumentations.RandomBrightnessContrast(p=0.6),
        albumentations.Cutout(max_h_size=max_size_cutout, max_w_size=max_size_cutout, num_holes=3, p=0.5),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val


def get_df(groups=0):
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

    if groups > 0:
        selected = df.label_group.unique()[:groups]
        df = df[df.label_group.isin(selected)]

    out_dim = df.label_group.nunique()

    return df, out_dim