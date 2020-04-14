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
                sometimes(iaa.Crop(percent=(0, 0.1))),   # crop的幅度为0到10%

                sometimes(iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-10, 10),
                    shear=(-10, 10),
                    order=[0, 1],
                    cval=(0, 255),
                )),

                # 使用下面的0个到5个之间的方法去增强图像。注意SomeOf的用法
                iaa.SomeOf(
                    (0, 1),
                   [

                       # 锐化处理
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                       # 每个像素随机加减-10到10之间的数
                       iaa.Add((-10, 10), per_channel=0.5),

                       # 将整个图像的对比度变为原来的一半或者二倍
                       iaa.contrast.LinearContrast((0.8, 1.2), per_channel=0.5),
                   ],

                   random_order=True  # 随机的顺序把这些操作用在图像上
                )
            ],
            random_order=True  # 随机的顺序把这些操作用在图像上
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


        if DATA_AUGMENTATION == True and self.mode == 'train':
            sample = self.iaa_seq.augment_image(sample)
        sample = cv2.resize(sample, (IMAGE_SIZE, IMAGE_SIZE))
        # cv2.imshow('sample', sample)
        # cv2.waitKey(0)

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
    count1 = counts[counts >= MIN_SAMPLES_PER_CLASS]
    selected_classes = count1[count1 <= MAX_SAMPLES_PER_CLASS].index

    class2 = []
    if USE_SELECT == True:
        select_df = pd.read_csv(SELECT_CLASS)
        select_good = np.array(select_df)

        for index in range(len(select_good)):
            if select_good[index][1] >= 0.75:
                class2.append(select_good[index][0])

        class1 = list(selected_classes)
        good_class = list(set(class1).intersection(set(class2)))
        num_classes = len(good_class)
        print('classes with at least N samples:', num_classes)
        train_select_df = train_df.loc[train_df.class_id.isin(good_class)].copy()
        print('train_df', train_select_df.shape)
        val_select_df = val_df.loc[val_df.class_id.isin(good_class)].copy()
        print('val_df', val_select_df.shape)

    else:
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