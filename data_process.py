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
    data = compose(torch.tensor(data, dtype=float))
    mask = torch.tensor(image_data[4, :, :])
    return data, mask

def tif_sar_class(image_path, compose:tf.Compose):
    gdal.UseExceptions()
    image: Dataset = gdal.Open(image_path)
    image_data = image.ReadAsArray()
    data = image_data[(0, 1), :, :]
    data = np.log(data+1)
    data = compose(torch.tensor(data, dtype=float))
    mask = torch.tensor(image_data[4, :, :])
    return data, mask