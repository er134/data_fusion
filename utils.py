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
    pred_path = build_imglist(pred_path)
    true_path = build_imglist(true_path)
    assert len(pred_path) == len(true_path)
    with tqdm(total=len(pred_path), unit='img') as pbar:
        for i in range(len(pred_path)):
            Pred_path = Path(pred_path[i]).stem[:-5]
            True_path = Path(true_path[i]).stem

            if Pred_path == True_path:
                j = i
            else:
                is_exist = False
                for j in range(len(true_path)):
                    if Pred_path == Path(true_path[j]).stem:
                        is_exist = True
                        break
                assert is_exist

            pred = cv2.imread(pred_path[i], cv2.IMREAD_GRAYSCALE) > 0
            true = cv2.imread(true_path[j], cv2.IMREAD_GRAYSCALE) > 0
            # output = pred & np.logical_not(true)
            # cv2.imwrite(rf"C:\Users\Administrator\Documents\Study\SummerCompetition\FastSAM\train11\precision\{True_path}.png",output.astype(dtype=np.uint8)*255)
            indexes.calCM_once(pred, true)
            pbar.update(1)

    results = indexes.update()
    print(f'Total {len(true_path)} images has been calculated.')

    return results['accuracy'], results['kappa'], results['iou'], \
        results['miou'], results['precision'], results['recall']

if __name__  == '__main__':
    cal_slope('./Track1/train/images', './data/slope')