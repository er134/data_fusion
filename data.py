import os
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
from torch.utils.data.dataset import ConcatDataset, Dataset
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
    def __init__(self, data_path:str, data_process:Callable[[str], torch.Tensor], left=0, right=None, augment = True, resize=False, cut=False) -> None:
        super(WaterBaseDataSet, self).__init__()
        self.data_path = data_path
        self.images, self.labels = self.build_data()
        right = len(self.images) if right is None else right 
        self.images = self.images[left:right]
        if self.labels is not None: 
            self.labels = self.labels[left:right]
        self.data_process = data_process
        self.augment = augment
        self.resize = resize
        self.cut = cut

    ''' build images and labels
    implemented:
    self.images = ...
    self.labels = ...
    '''
    # def build_data(self) -> tuple[list[str], Optional[list[str]]]:
    def build_data(self):
        raise NotImplementedError
    
    def __len__(self) -> int:
        if self.cut:
            return len(self.images) * 5
        return len(self.images) 

    def __getitem__(self, index) -> dict:
        if self.cut:
            range = index % 5
            index = index // 5
        name = Path(self.images[index]).stem
        image = self.images[index]
        data, mask = self.data_process(image)
        assert isinstance(data, torch.Tensor) or isinstance(data, tuple)
        label = self.labels[index] if self.labels else []
        if isinstance(label, str):
            label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
            label = torch.tensor(label).unsqueeze(0)
        if self.augment:
            p1, p2, p3, p4 = 0.5, 0.5, 0.5, 0.
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
                i, j, h, w = tf.RandomCrop.get_params(data, (256, 256))
                data = F.crop(data, i, j, 256, 256)
                label = F.crop(label, i, j, 256, 256)
                mask = F.crop(mask, i, j, 256, 256)
            if torch.rand(1) < p3:
                x, y, h, w, v = tf.RandomErasing.get_params(data, scale=(0.02, 0.33), ratio=(0.3, 3.3))
                data = F.erase(data, x, y, h ,w, v)
                label = F.erase(label, x, y, h ,w, torch.tensor(0.))
        if self.resize:
            _, h, w = (2, 512, 512)#data.shape
            data = F.resize(data, (h//2, w//2), antialias=True)
            # label = F.resize(label, (h//2, w//2), antialias=True) 
            # mask = F.resize(mask, (h//2, w//2), antialias=True)
        if self.cut:
            _, h, w = data.shape
            match range:
                case 0:
                    data = F.resize(data, (h//2, w//2), antialias=True) 
                    label = F.resize(label, (h//2, w//2), antialias=True) 
                    mask = F.resize(mask, (h//2, w//2), antialias=True) 
                case 1:
                    data = F.crop(data, 0, 0, 256, 256)
                    label = F.crop(label, 0, 0, 256, 256)
                    mask = F.crop(mask, 0, 0, 256, 256)
                case 2:
                    data = F.crop(data, 0, 256, 256, 256)
                    label = F.crop(label, 0, 256, 256, 256)
                    mask = F.crop(mask, 0, 256, 256, 256)
                case 3:
                    data = F.crop(data, 256, 0, 256, 256)
                    label = F.crop(label, 256, 0, 256, 256)
                    mask = F.crop(mask, 256, 0, 256, 256)
                case 4:
                    data = F.crop(data, 256, 256, 256, 256)
                    label = F.crop(label, 256, 256, 256, 256)
                    mask = F.crop(mask, 256, 256, 256, 256)
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
    
class WaterSemiDataSet(WaterBaseDataSet):
    def build_data(self):
        return ([], None)