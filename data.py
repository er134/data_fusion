import os
import glob
from pathlib import Path

import cv2
import numpy as np
from osgeo import gdal
from osgeo.gdal import Dataset
import torchvision.transforms as tf
from sklearn import preprocessing
from torch import torch

from utils import build_imglist

class WaterDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, left=0, right=None) -> None:
        super(WaterDataSet, self).__init__()
        train_path = os.path.join(data_path, 'train', 'images')
        label_path = os.path.join(data_path, 'train', 'labels')
        self.trains = build_imglist(train_path)
        self.labels = build_imglist(label_path)
        assert len(self.trains) == len(self.labels)
        right = len(self.trains) if right is None else right
        self.trains = self.trains[left:right]
        self.labels = self.labels[left:right]
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> dict:
        gdal.UseExceptions()
        name = Path(self.trains[index]).stem
        image = self.trains[index]
        label = self.labels[index]
        image:Dataset = gdal.Open(image)
        image_data = image.ReadAsArray()
        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        data = image_data.astype(dtype=float)[(0,1),:,:]
        for i in range(data.shape[0]):
            data[ i, :, :] = preprocessing.scale(data[ i, :, :])
        data = torch.tensor(data, dtype=float)
        label = torch.tensor(label).unsqueeze(0)
        del image
        return {'data': data, 'label':label, 'name':name}
    
class WaterTrainDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path, left=0, right=None) -> None:
        super(WaterTrainDataSet, self).__init__()
        train_path = os.path.join(data_path, 'train', 'images')
        label_path = os.path.join(data_path, 'train', 'labels')
        self.trains = build_imglist(train_path)
        self.labels = build_imglist(label_path)
        assert len(self.trains) == len(self.labels)
        right = len(self.trains) if right is None else right
        self.trains = self.trains[left:right]
        self.labels = self.labels[left:right]
        # mean = np.array([1773.20873858, 381.89445823, 157.24579071, 159.12466477, 31.49188123, 3.68448185])
        # std = np.array([2781.19743946, 687.20408303, 337.99843602, 340.17951273, 17.91570697, 17.93172077])
        mean = [6.99297937, 5.50701671]   
        std = [1.00755637, 0.98871325]     
        self.compose = tf.Compose([tf.Normalize(mean=mean, std=std)])
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> dict:
        name = Path(self.trains[index]).stem
        image = self.trains[index]
        label = self.labels[index]
        data = np.load(image)
        data[data == -9999] = 0
        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)
        data = np.log(data[(0,1),:,:]+1)
        data = self.compose(torch.tensor(data, dtype=float))
        label = torch.tensor(label).unsqueeze(0)
        return {'data': data, 'label':label, 'name':name}


class WaterPredictDataSet(torch.utils.data.Dataset):
    def __init__(self, data_path) -> None:
        super(WaterPredictDataSet, self).__init__()
        self.images = build_imglist(data_path)
        mean = np.array([1773.20873858, 381.89445823, 157.24579071, 159.12466477, 31.49188123, 3.68448185])
        std = np.array([2781.19743946, 687.20408303, 337.99843602, 340.17951273, 17.91570697, 17.93172077])        
        self.compose = tf.Compose([tf.Normalize(mean=mean, std=std)])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index) -> dict:
        gdal.UseExceptions()
        name = Path(self.images[index]).stem
        image = self.images[index]
        image:Dataset = gdal.Open(image)
        image_data = image.ReadAsArray()
        data = image_data.astype(dtype=float)#[:,:,:]
        for i in range(data.shape[0]):
            data[ i, :, :] = preprocessing.scale(data[ i, :, :])
        # data = self.compose(torch.tensor(image_data, dtype=float))
        del image
        return {'data': data, 'name':name}

if __name__ == "__main__":
    all = np.zeros((2), dtype=float)
    dataset = WaterTrainDataSet(r'./data/npy')
    for train_data in dataset:
        data = train_data['data']
        data[data == -9999] = 0
        mean_single = np.nanmean(np.log(data[(0,1),:,:]+1), (1, 2))
        all += mean_single
        print(f'mean: {mean_single}   name: {train_data["name"]}')
    mean = all / len(dataset)
    std = np.zeros((2, 512, 512), dtype=float)
    for train_data in dataset:
        data = train_data['data'].astype(dtype=float)
        data[data == -9999] = 0
        data = np.log(data[(0,1),:,:]+1)
        for index in range(data.shape[0]):
            data[index] -= mean[index]
        std += np.power(data,2)
    std = std.sum(axis=(1,2))
    std /= (len(dataset)*512*512-1)
    std = np.sqrt(std)
    print(mean, std)

        # for i, data_single in enumerate(data):
        #     path = f'./data/{i}/'
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        #     cv2.imwrite(f'{path}{name}.png',data_single)
