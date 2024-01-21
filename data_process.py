import cv2
import numpy as np
from osgeo import gdal
from osgeo.gdal import Dataset
import torch
import torchvision.transforms as tf
from sklearn import preprocessing

'''
@params image_path:str image path
@params compose: normlize
'''
def tif_only_sar(image_path, compose:tf.Compose) -> torch.Tensor:
    image: Dataset = gdal.Open(image_path)
    image_data = image.ReadAsArray()
    data = np.log(image_data[(0, 1), :, :])
    data = compose(torch.tensor(data, dtype=float))
    return data

def tif_only_sar_water(image_path, compose:tf.Compose)-> torch.Tensor:
    gdal.UseExceptions()
    image: Dataset = gdal.Open(image_path)
    image_data = image.ReadAsArray()
    data = image_data[(0, 1, 5), :, :]
    data[(0, 1), :, :] = np.log(data[(0, 1), :, :]+1)
    data = compose(torch.tensor(data, dtype=float))
    return data

def npy_only_sar_water(image_path, compose:tf.Compose)-> torch.Tensor:
    image_data = np.load(image_path)
    data = image_data[(0, 1, 5), :, :]
    data[(0, 1), :, :] = np.log(data[(0, 1), :, :]+1)
    data = compose(torch.tensor(data, dtype=float))
    return data


