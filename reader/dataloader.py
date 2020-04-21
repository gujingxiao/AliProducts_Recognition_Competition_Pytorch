# Author: Jingxiao Gu
# Description: Dataloader Code for AliProducts Recognition Competition

import numpy as np
import pandas as pd
from typing import Any, Tuple

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from imgaug import augmenters as iaa

from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import cv2
from utils.configs import *

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame, mode: str) -> None:
        print('creating data loader - {}'.format(mode))
        assert mode in ['train', 'val', 'test']

        self.df = dataframe
        self.mode = mode

        transforms_list = [transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])]

        iaa_seq = iaa.Sequential(
            [
                # iaa.Fliplr(0.4),  # 对50%的图像进行上下翻转
                sometimes(iaa.Crop(percent=(0, 0.06))),   # crop的幅度为0到10%
                sometimes(iaa.Affine(
                    rotate=(-15, 15),
                    cval=(0),
                )),
            ],
        )

        self.transforms = transforms.Compose(transforms_list)
        self.iaa_seq = iaa_seq

    def __getitem__(self, index: int) -> Any:
        ''' Returns: tuple (sample, target) '''
        subfolder = self.df.label_id.values[index]
        subfolder = '%05d' % subfolder
        filename = self.df.image_id.values[index]

        if self.mode == 'train':
            sample = cv2.imread(TRAIN_PATH + subfolder + '/' + filename)

            if DATA_AUGMENTATION == True:
                sample = self.iaa_seq.augment_image(sample)
        else:
            sample = cv2.imread(VAL_PATH + subfolder + '/' + filename)

        # Make border to let the image suit for any scale and ratio
        height, width, channels = sample.shape
        if height != width:
            max_edge = max(height, width)
            left = round((max_edge - width) / 2.0)
            right = max_edge - width - left
            top = round((max_edge - height) / 2.0)
            bottom = max_edge - height - top
            sample = cv2.copyMakeBorder(sample, top, bottom, left, right, cv2.BORDER_CONSTANT)

        sample = cv2.resize(sample, (IMAGE_SIZE, IMAGE_SIZE))
        # cv2.imshow('sample', sample)
        # cv2.waitKey(0)

        image = self.transforms(sample)
        return image, self.df.class_id.values[index], subfolder + '/' + filename

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
    count1 = counts[counts >= MIN_SAMPLES_PER_CLASS]
    selected_classes = count1[count1 <= MAX_SAMPLES_PER_CLASS].index

    num_classes = selected_classes.shape[0]
    print('classes with at least N samples:', num_classes)

    train_select_df = train_df.loc[train_df.class_id.isin(selected_classes)].copy()
    print('train_df', train_select_df.shape)
    val_select_df = val_df.loc[val_df.class_id.isin(selected_classes)].copy()
    print('val_df', val_select_df.shape)

    label_encoder_train = LabelEncoder()
    label_encoder_train.fit(train_select_df.class_id.values)
    label_encoder_val = LabelEncoder()
    label_encoder_val.fit(val_select_df.class_id.values)

    assert len(label_encoder_train.classes_) == num_classes
    assert len(label_encoder_val.classes_) == num_classes
    train_select_df.class_id = label_encoder_train.transform(train_select_df.class_id)
    train_dataset = ImageDataset(train_select_df, mode='train')
    val_select_df.class_id = label_encoder_val.transform(val_select_df.class_id)
    val_dataset = ImageDataset(val_select_df, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

    return train_loader, val_loader, label_encoder_train, label_encoder_val, num_classes


# def load_data(mode='train') -> 'Tuple[DataLoader[np.ndarray],DataLoader[np.ndarray], LabelEncoder, LabelEncoder, int, int]':
#     torch.multiprocessing.set_sharing_strategy('file_descriptor')
#     cudnn.benchmark = True
#
#     # only use classes which have at least MIN_SAMPLES_PER_CLASS samples
#     print('loading data...')
#
#     train_df = pd.read_csv(TRAIN_CSV)
#     val_df = pd.read_csv(VAL_CSV)
#     counts = train_df.class_id.value_counts()
#     count1 = counts[counts >= MIN_SAMPLES_PER_CLASS]
#     selected_classes = count1[count1 <= MAX_SAMPLES_PER_CLASS].index
#
#     num_classes = selected_classes.shape[0]
#     print('classes with at least N samples:', num_classes)
#
#     train_select_df = train_df.loc[train_df.class_id.isin(selected_classes)].copy()
#     print('train_df', train_select_df.shape)
#     val_select_df = val_df.loc[val_df.class_id.isin(selected_classes)].copy()
#     print('val_df', val_select_df.shape)
#
#     label_encoder_train = LabelEncoder()
#     label_encoder_train.fit(train_select_df.class_id.values)
#     label_encoder_val = LabelEncoder()
#     label_encoder_val.fit(val_select_df.class_id.values)
#
#     label_encoder_train_level3 = LabelEncoder()
#     label_encoder_train_level3.fit(train_select_df.level3.values)
#     label_encoder_val_level3 = LabelEncoder()
#     label_encoder_val_level3.fit(val_select_df.level3.values)
#     print('found classes', len(label_encoder_train.classes_))
#     print('found level3 classes', len(label_encoder_train_level3.classes_))
#     num_level3_classes = len(label_encoder_train_level3.classes_)
#
#     assert len(label_encoder_train.classes_) == num_classes
#     assert len(label_encoder_val.classes_) == num_classes
#     train_select_df.class_id = label_encoder_train.transform(train_select_df.class_id)
#     train_select_df.level3 = label_encoder_train.transform(train_select_df.level3)
#     train_dataset = ImageDataset(train_select_df, mode='train')
#     val_select_df.class_id = label_encoder_val.transform(val_select_df.class_id)
#     val_select_df.level3 = label_encoder_val.transform(val_select_df.level3)
#     val_dataset = ImageDataset(val_select_df, mode='val')
#
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
#                               shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
#                               shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
#
#     return train_loader, val_loader, label_encoder_train, label_encoder_val, num_classes, num_level3_classes