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
        self.confusion_matrix = np.zeros((num, num), dtype=float)

    def calCM_once(self, predict, target):  # Calculate confuse matrix for a mini-batch
        t = target.flatten()
        p = predict.flatten()
        cm = confusion_matrix(t, p, labels= [0, 1])
        self.confusion_matrix += cm

    def calculate(self):
        TN, FP, FN, TP = self.confusion_matrix.ravel()
        return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}

    def F1_score(self):
        return 2*self.precision()*self.recall()/(self.precision()+self.recall()+self.eps)

    def precision(self):
        return self.matrix['TP']/(self.matrix['TP']+self.matrix['FP']+self.eps)

    def recall(self):
        return self.matrix['TP']/(self.matrix['TP']+self.matrix['FN']+self.eps)

    def update(self):
        self.matrix = self.calculate()
        return {'f1': self.F1_score(), 'precision': self.precision(), 'recall': self.recall()}

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
    pred_images = build_imglist(pred_path)[-200:]
    true_images = build_imglist(true_path)[-200:]
    assert len(pred_images) == len(true_images) 
    with tqdm(total=len(pred_images), unit='img') as pbar:
        for pred_image, true_image in zip(pred_images, true_images):
            pred = cv2.imread(pred_image, cv2.IMREAD_GRAYSCALE) > 0
            true = cv2.imread(true_image, cv2.IMREAD_GRAYSCALE) > 0
            indexes.calCM_once(pred.ravel(), true.ravel())
            pbar.update(1)

    results = indexes.update()
    print(f'Total {len(true_images)} images has been calculated.')

    return {'f1': results['f1'], 'pr': results['precision'], 're': results['recall']}

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

def CutMix(image, label, mask, alpha=1.0):
    """
    inputs:
        image: batch of input images, shape (N, C, H, W)
        dem: batch of input dem data, shape (N, H, W)
        ...
        target: batch of target images, shape (N, H, W)
        alpha: regularization hyperparameter
    outputs:
        image: batch of mixed input images, shape (N, C, H, W)
        dem: batch of mixed input dem data, shape (N, H, W)
        ...
        target: batch of mixed target images, shape (N, H, W)
        lam: Mixing ratio
    """
    indices = torch.randperm(image.size(0))
    shuffled_image = image[indices]
    shuffled_label = label[indices]
    shuffled_mask = mask[indices]

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)

    image[:, :, bbx1:bbx2, bby1:bby2] = shuffled_image[:, :, bbx1:bbx2, bby1:bby2]
    label[:, bbx1:bbx2, bby1:bby2] = shuffled_label[:, bbx1:bbx2, bby1:bby2]
    mask[:, bbx1:bbx2, bby1:bby2] = shuffled_mask[:, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)

    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

if __name__  == '__main__':
    r = metrics_stats(r'./results/resize256_aug_sdwi_multihead_semi_1/perdict', r'./data/npy/train/labels')
    print(r)