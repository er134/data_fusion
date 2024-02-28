import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data import WaterTrainDataSet
from data_process import npy_sar_class
from utils import build_imglist, metrics
import torchvision.transforms as tf
from sklearn.metrics import PrecisionRecallDisplay

data_path_list = ['./data/Track1',  # 原始数据
                  './data/npy',  # 原始数据的numpy格式
                  './data/Track2'
                  ] 
data_path = data_path_list[1]

mean = [6.99297937, 5.50701671, 14.8026]#, 3.68448185]
std = [1.00755637, 0.98871325, 1.8027]#, 17.93172077]

compose = tf.Compose([tf.Normalize(mean, std, inplace=True)])
dp = lambda image:npy_sar_class(image, compose)
imgs = build_imglist('/home/er134/data_fusion/results/resize256_aug_sdwi_multihead_10/perdict')
ds_valid = WaterTrainDataSet(data_path, dp, -200, augment=False, resize=True)

me = metrics()
num = 0
for i, data in enumerate(ds_valid):
    mask = data['mask'].numpy()
    label = data['label'].numpy().squeeze()
    name = data['name']
    mask_cls = mask == 40#~((mask == 10) | (mask == 30) | (mask == 40) | (mask == 90))
    num += np.count_nonzero(mask_cls)
    img = cv2.imread(imgs[i + 1431], cv2.IMREAD_GRAYSCALE)
    me.calCM_once(img[mask_cls], label[mask_cls])
print(me.update())
print(f'num: {num}')


# gray = 256
# metrics_all = list()
# for i in range(gray):
#     metrics_all.append(metrics())

# pr = np.zeros(gray)
# re = np.zeros(gray)
# with tqdm(total=200, desc=f'Cal PR', unit='img') as pbar:
#     for i, data in enumerate(ds_valid):
#         mask = data['mask'].numpy()
#         label = data['label'].numpy().squeeze()
#         name = data['name']
#         img = cv2.imread(imgs[i], cv2.IMREAD_GRAYSCALE)
#         mask_cls = mask != 80
#         for k in range(gray):
#             result = img >= k
#             metrics_all[k].calCM_once(result[mask_cls], label[mask_cls])            
#         pbar.update(1)

# for k in range(gray):
#     metrics_all[k].update()
#     pr[k] = metrics_all[k].precision()
#     re[k] = metrics_all[k].recall()

# np.save('pr_no80.npy', pr)
# np.save('re_no80.npy', re) 

# pr = np.load('pr_no80.npy')
# re = np.load('re_no80.npy')

# for k in range(gray):
#     pr[k] = re[k]
#     re[k] = k/255

# disp = PrecisionRecallDisplay(pr, re)
# disp.plot()

# plt.show()

