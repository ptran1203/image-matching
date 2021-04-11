import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from transformers import AutoTokenizer

DATA_DIR = '/content'

if os.path.exists('/content'):
    tokenizer = AutoTokenizer.from_pretrained('/content/bert-base-uncased')
else:
    tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/bert-base-uncased')

class ShoppeDataset(Dataset):
    def __init__(self, csv, mode='train', transform=None,
        image_dir=os.path.join(DATA_DIR, 'train_images'),
    ):

        self.csv = csv.reset_index()
        self.mode = mode
        self.transform = transform
        self.img_dir = image_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        image = cv2.imread(os.path.join(self.img_dir, row.image))[:,:,::-1]
        text = row.title
        text = self.tokenizer(text, padding='max_length', truncation=True, max_length=16, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]
        
        # transform
        image = self.transform(image=image)['image'].astype(np.float32)

        image = image.transpose(2, 0, 1)
        if self.mode == 'test':
            return torch.tensor(image), input_ids, attention_mask
        else:
            return torch.tensor(image), input_ids, attention_mask, torch.tensor(row.label_group)


def get_transforms(image_size, stage=1, norm=True):
    if stage == 1:
        max_size_cutout = int(image_size * 0.2)
        transforms_train = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.1),
            A.JpegCompression(quality_lower=80, quality_upper=100),
            A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=30, border_mode=0, p=0.3),
            A.OneOf([
                A.MedianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=(3, 7)),
                A.GaussNoise(),
            ], p=0.3),
            A.OneOf([
                A.GridDistortion(),
                A.OpticalDistortion(),
            ], p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Cutout(max_h_size=max_size_cutout, max_w_size=max_size_cutout, num_holes=2, p=0.3),
        ]
    elif stage == 2:
        max_size_cutout = int(image_size * 0.2)
        transforms_train = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.3),
            A.JpegCompression(quality_lower=80, quality_upper=100),
            A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=30, border_mode=0, p=0.5),
            A.OneOf([
                A.MedianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=(3, 7)),
                A.GaussNoise(),
            ], p=0.5),
            A.OneOf([
                A.GridDistortion(),
                A.OpticalDistortion(),
            ], p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Cutout(max_h_size=max_size_cutout, max_w_size=max_size_cutout, num_holes=2, p=0.5),
        ]
    else:
        max_size_cutout = int(image_size * 0.25)
        transforms_train = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.JpegCompression(quality_lower=80, quality_upper=100),
            A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=30, border_mode=0, p=0.7),
            A.OneOf([
                A.MedianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=(3, 7)),
                A.GaussNoise(),
            ], p=0.7),
            A.OneOf([
                A.GridDistortion(),
                A.OpticalDistortion(),
            ], p=0.7),
            A.RandomBrightnessContrast(p=0.7),
            A.Cutout(max_h_size=max_size_cutout, max_w_size=max_size_cutout, num_holes=2, p=0.7),
        ]

    transforms_val = [
        A.Resize(image_size, image_size),
    ]

    if norm:
        transforms_train.append(A.Normalize())
        transforms_val.append(A.Normalize())
    else:
        transforms_train.append(A.Normalize(mean=0, std=1))
        transforms_val.append(A.Normalize(mean=0, std=1))

    # transforms_train.append(ToTensorV2(p=1.0))
    # transforms_val.append(ToTensorV2(p=1.0))

    return A.Compose(transforms_train), A.Compose(transforms_val)


def get_df(groups=0):
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

    if groups > 0:
        selected = df.label_group.unique()[:groups]
        df = df[df.label_group.isin(selected)]

    return df