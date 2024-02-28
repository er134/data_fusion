import cv2
import numpy as np
from osgeo import gdal
from osgeo.gdal import Dataset
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from sklearn import preprocessing

'''
@params image_path:str image path
@params compose: normlize
'''
def npy_sar_class(image_path, compose:tf.Compose):
    image_data = np.load(image_path)
    data = image_data[(0, 1), :, :]
    data = np.log(data+1)
    sdwi = np.expand_dims(data[0, :, :] + data[1,:,:] + np.log(10), 0)
    data = np.concatenate((data, sdwi), axis=0)
    data = compose(torch.tensor(data, dtype=float))
    mask = torch.tensor(image_data[4, :, :])
    return data, mask

def tif_sar_class(image_path, compose:tf.Compose):
    gdal.UseExceptions()
    image: Dataset = gdal.Open(image_path)
    image_data = image.ReadAsArray()
    data = image_data[(0, 1), :, :]
    data = np.log(data+1)
    sdwi = np.expand_dims(data[0, :, :] + data[1,:,:] + np.log(10), 0)
    data = np.concatenate((data, sdwi), axis=0)
    data = compose(torch.tensor(data, dtype=float))
    mask = image_data[4, :, :]
    mask = torch.tensor(mask)
    return data, mask