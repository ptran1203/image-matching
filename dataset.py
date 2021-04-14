import os
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from augment.transform import get_transforms

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


def get_df(groups=0):
    df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

    if groups > 0:
        selected = df.label_group.unique()[:groups]
        df = df[df.label_group.isin(selected)]

    return df