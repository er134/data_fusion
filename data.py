import os
from pathlib import Path
from typing import Optional, Callable

import cv2
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
    def __init__(self, data_path:str, data_process:Callable[[str], torch.Tensor], left=0, right=None) -> None:
        super(WaterBaseDataSet, self).__init__()
        self.data_path = data_path
        self.images, self.labels = self.build_data()
        right = len(self.images) if right is None else right
        self.images = self.images[left:right]
        if self.labels is not None: 
            self.labels = self.labels[left:right]
        self.data_process = data_process

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
        data = self.data_process(image)
        assert isinstance(data, torch.Tensor) or isinstance(data, tuple)
        label = self.labels[index] if self.labels else []
        if isinstance(label, str):
            label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
            label = torch.tensor(label)
        return {'data': data, 'label': label, 'name': name}

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