# Author: Jingxiao Gu
# Description: Dataloader Code for AliProducts Recognition Competition

import numpy as np
import pandas as pd
from typing import Any, Tuple

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from utils.configs import *

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, mode: str) -> None:
        print('creating data loader - {}'.format(mode))
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.mode = mode

        transforms_list = []

        if self.mode == 'train':
            transforms_list = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomChoice([
                    transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)]),
                    transforms.RandomApply([transforms.RandomAffine(degrees=15, translate=(0.2, 0.2),
                                            scale=(0.8, 1.2), shear=15,
                                            resample=Image.BILINEAR)])
                ])
            ]

        transforms_list.extend([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]),
        ])
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, index: int) -> Any:
        ''' Returns: tuple (sample, target) '''
        subfolder = self.df.label_id.values[index]
        subfolder = '%05d' % subfolder
        filename = self.df.image_id.values[index]

        if self.mode == 'train':
            sample = Image.open(TRAIN_PATH + subfolder + '/' + filename)
        else:
            sample = Image.open(VAL_PATH + subfolder + '/' + filename)

        if sample.mode != 'RGB':
            #print(sample.mode)
            sample = sample.convert('RGB')

        image = self.transforms(sample)
        return image, self.df.class_id.values[index]

    def __len__(self) -> int:
        return self.df.shape[0]

def load_data(mode='train') -> 'Tuple[DataLoader[np.ndarray],DataLoader[np.ndarray], LabelEncoder, LabelEncoder, int]':
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    cudnn.benchmark = True

    # only use classes which have at least MIN_SAMPLES_PER_CLASS samples
    print('loading data...')


    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    counts = train_df.class_id.value_counts()
    selected_classes = counts[counts >= MIN_SAMPLES_PER_CLASS].index
    num_classes = selected_classes.shape[0]
    print('classes with at least N samples:', num_classes)

    train_select_df = train_df.loc[train_df.class_id.isin(selected_classes)].copy()
    print('train_df', train_df.shape)
    val_select_df = val_df.loc[val_df.class_id.isin(selected_classes)].copy()
    print('val_df', val_df.shape)

    label_encoder_train = LabelEncoder()
    label_encoder_train.fit(train_select_df.class_id.values)
    label_encoder_val = LabelEncoder()
    label_encoder_val.fit(val_select_df.class_id.values)
    print('found classes', len(label_encoder_train.classes_))

    assert len(label_encoder_train.classes_) == num_classes
    assert len(label_encoder_val.classes_) == num_classes
    train_select_df.class_id = label_encoder_train.transform(train_select_df.class_id)
    train_dataset = ImageDataset(train_select_df, mode='train')
    val_select_df.class_id = label_encoder_val.transform(val_select_df.class_id)
    val_dataset = ImageDataset(val_select_df, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, drop_last=True)

    return train_loader, val_loader, label_encoder_train, label_encoder_val, num_classes