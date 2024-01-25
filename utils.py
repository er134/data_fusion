import os
import glob
from typing import Optional
from pathlib import Path

import cv2
import torch
import numpy as np
from tqdm import tqdm
from osgeo import gdal
from osgeo.gdal import Dataset
from sklearn.metrics import confusion_matrix

# image suffixes
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm', 'npy'


def build_imglist(path):

    # *.txt file with img/vid/dir on each line
    if isinstance(path, str) and Path(path).suffix == '.txt':
        path = Path(path).read_text().rsplit()
    files = []

    for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
        # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
        p = str(Path(p).absolute())
        if '*' in p:
            files.extend(sorted(glob.glob(p, recursive=True)))  # glob
        elif os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
        elif os.path.isfile(p):
            files.append(p)  # files
        else:
            raise FileNotFoundError(f'{p} does not exist')

    return [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]


class metrics():

    def __init__(self):
        self.class_index = [0, 1]
        self.eps = 1e-15
        self.refresh()

    def refresh(self):
        num = len(self.class_index)
        # accumulated confuse matrix, float-type
        self.confusion_matrix = np.zeros((num, num))
        self.matrix = self.calculate(self.confusion_matrix)

    def calCM_once(self, predict, target):  # Calculate confuse matrix for a mini-batch
        if len(predict.shape) == 3:
            predict = predict.squeeze(0)
            target = target.squeeze(0)
        elif len(predict.shape) == 1:
            cm = confusion_matrix(target, predict)
            self.confusion_matrix += cm
            return
        for p, t in zip(predict, target):
            p = np.int32(p.reshape(-1))
            t = np.int32(t.reshape(-1))
            cm = confusion_matrix(t, p)
            self.confusion_matrix += cm

    def calculate(self, confusion_m):
        FP = confusion_m.sum(axis=0) - np.diag(confusion_m)
        FN = confusion_m.sum(axis=1) - np.diag(confusion_m)
        TP = np.diag(confusion_m)
        TN = confusion_m.sum() - (FP + FN + TP)
        return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

    def F1_score(self):
        return 2*self.precision()*self.recall()/(self.precision()+self.recall()+self.eps)

    def precision(self):
        return self.matrix['TP']/(self.matrix['TP']+self.matrix['FP']+self.eps)

    def recall(self):
        return self.matrix['TP']/(self.matrix['TP']+self.matrix['FN']+self.eps)

    def update(self):
        self.matrix = self.calculate(self.confusion_matrix)
        return {'f1': self.F1_score()[1], 'precision': self.precision()[1], 'recall': self.recall()[1]}

def cal_slope(data_path:str, output_path:str):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    imgs = build_imglist(data_path)
    for img in imgs:
        img_data:Dataset = gdal.Open(img)
        dem:Dataset = img_data.GetRasterBand(3).GetDataset()
        dem.Set
        name = Path(img).stem
        options = gdal.DEMProcessingOptions
        gdal.DEMProcessing(f'{output_path}/{name}.tif', dem, 'slope')

def metrics_stats(pred_path, true_path):

    indexes = metrics()
    pred_images = build_imglist(pred_path)
    true_images = build_imglist(true_path)
    assert len(pred_images) == len(true_images) 
    with tqdm(total=len(pred_images), unit='img') as pbar:
        for pred_image, true_image in zip(pred_images, true_images):
            pred = cv2.imread(pred_image, cv2.IMREAD_GRAYSCALE) > 0
            true = cv2.imread(true_image, cv2.IMREAD_GRAYSCALE) > 0
            indexes.calCM_once(np.expand_dims(pred, 0), np.expand_dims(true,0))
            pbar.update(1)

    results = indexes.update()
    print(f'Total {len(true_images)} images has been calculated.')

    return {'f1': results['f1'][-1], 'pr': results['precision'][-1], 're': results['recall'][-1]}

def loss_stats(pred_path, true_path):

    indexes = metrics()
    pred_images = build_imglist(pred_path).sort()
    true_images = build_imglist(true_path).sort()
    assert len(pred_images) == len(true_images) 
    with tqdm(total=len(pred_images), unit='img') as pbar:
        for pred_image, true_image in zip(pred_images, true_images):
            pred = cv2.imread(pred_image, cv2.IMREAD_GRAYSCALE) > 0
            true = cv2.imread(true_image, cv2.IMREAD_GRAYSCALE) > 0
            indexes.calCM_once(pred.unsqueeze(0), true.unsqueeze(0))
            pbar.update(1)

    results = indexes.update()
    print(f'Total {len(true_images)} images has been calculated.')

    return results['f1'], results['precision'], results['recall']

if __name__  == '__main__':
    r = metrics_stats(r'E:\data_fusion\results\result_sar_water_predict_1\perdict', r'E:\data_fusion\Track1\train\labels')
    print(r)