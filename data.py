import os
from pathlib import Path
from typing import Optional, Callable

import cv2
import torchvision.transforms as tf
import torchvision.transforms.functional as F
from torch import torch

from utils import build_imglist


class WaterBaseDataSet(torch.utils.data.Dataset): 
    ''' based dataset of water detection 
    @params data_path:str (data path) 
    --data_path
      --train
        --images
        --labels
      --val
        --images
      --test
        --images
    @params data_process: (str)->torch.Tensor data preprocessing function
    @params left, right:int (data index range of start and end)
    '''
    def __init__(self, data_path:str, data_process:Callable[[str], torch.Tensor], left=0, right=None, augment = True) -> None:
        super(WaterBaseDataSet, self).__init__()
        self.data_path = data_path
        self.images, self.labels = self.build_data()
        right = len(self.images) if right is None else right
        self.images = self.images[left:right]
        if self.labels is not None: 
            self.labels = self.labels[left:right]
        self.data_process = data_process
        self.augment = augment

    ''' build images and labels
    implemented:
    self.images = ...
    self.labels = ...
    '''
    # def build_data(self) -> tuple[list[str], Optional[list[str]]]:
    def build_data(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index) -> dict:
        name = Path(self.images[index]).stem
        image = self.images[index]
        data, mask = self.data_process(image)
        assert isinstance(data, torch.Tensor) or isinstance(data, tuple)
        label = self.labels[index] if self.labels else []
        if isinstance(label, str):
            label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
            label = torch.tensor(label)
        if self.augment:
            p1, p2, p3, p4 = 0.5, 0.5, 0.5, 0.1
            shape = label.shape[-1]
            if torch.rand(1) < p1:
                data = F.hflip(data)
                label = F.hflip(label)
                mask = F.hflip(mask)
            if torch.rand(1) < p2:
                data = F.vflip(data)
                label = F.vflip(label)
                mask = F.vflip(mask)
            if torch.rand(1) < p4:
                i, j, h, w = tf.RandomCrop.get_params(data, (shape, shape))
                data = F.crop(data, i, j, h, w)
                label = F.crop(label, i, j, h, w)
                mask = F.crop(mask, i, j, h, w)
                data = F.resize(data, shape)
                label = F.resize(label, shape)
                mask = F.resize(mask, shape)
            if torch.rand(1) < p3:
                x, y, h, w, v = tf.RandomErasing.get_params(data, scale=(0.02, 0.33), ratio=(0.3, 3.3))
                data = F.erase(data, x, y, h ,w, v)
                label = F.erase(label, x, y, h ,w, torch.tensor(0.))
        return {'data': data, 'mask': mask, 'label': label, 'name': name}

class WaterTrainDataSet(WaterBaseDataSet):
    def build_data(self):
        train_path = os.path.join(self.data_path, 'train', 'images')
        label_path = os.path.join(self.data_path, 'train', 'labels')
        images = build_imglist(train_path)
        labels = build_imglist(label_path)
        return (images, labels)


class WaterPredictDataSet(WaterBaseDataSet):
    def build_data(self):
        images = build_imglist(self.data_path)
        return (images, None)