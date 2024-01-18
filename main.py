import os
from typing import Union, Optional


import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from osgeo import gdal
from tqdm import tqdm

from models import UNet, SegNext

from data import WaterDataSet, WaterPredictDataSet, WaterTrainDataSet
from loss import BCEFocalLoss, FocalLoss
from utils import metrics


class WaterDetection:
    def __init__(self, data_path, save_path, model: Union[str, nn.Module], epochs=100, batchsize=8, lr=1e-3) -> None:
        self.data_path = data_path
        self.save_path = save_path
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.lr = lr
        self.batchsize = batchsize
        if isinstance(model, str):
            self.model = torch.load(model, map_location=self.device)
        else:
            self.model = model.to(device=self.device)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f'Create Running Directory: {self.save_path}')
        else:
            print(f'Running Directory already exists: {self.save_path}')

    def train(self, ds_train=None, ds_valid=None):

        if ds_train is None:
            ds_train = WaterDataSet(self.data_path)
        train_loader = DataLoader(ds_train, batch_size=self.batchsize,
                                  shuffle=True, num_workers=0, pin_memory=True)
        if ds_valid is not None:
            valid_loader = DataLoader(ds_valid, batch_size=self.batchsize,
                                      shuffle=True, num_workers=0, pin_memory=True)
            n_valid = len(ds_valid)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=0, last_epoch=-1)
        criterion = BCEFocalLoss()
        log_dir = os.path.join(self.save_path, "Report.csv")
        n_train = len(ds_train)
        best_loss, best_epoch, indexes = 100, -1, metrics()
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            indexes.refresh()
            loss_num = 0.
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{self.epochs}:train',
                      unit='img') as pbar:
                for batch in train_loader:

                    imag = batch['data'].to(
                        device=self.device, dtype=torch.float32)
                    goal = batch['label'].to(
                        device=self.device, dtype=torch.long)

                    pred = self.model(imag)
                    pred = F.sigmoid(pred)
                    loss = criterion(pred, goal)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    true_size = imag.shape[0]
                    pred_label = pred >= 0.5
                    indexes.calCM_once(
                        pred_label.cpu().numpy(), goal.cpu().numpy())
                    loss_num += loss.item() * true_size
                    pbar.update(true_size)
            lr_scheduler.step()
            loss_num /= n_train
            indexes_num = indexes.update()
            precision = indexes_num['precision']
            recall = indexes_num['recall']
            f1 = indexes_num['f1']
            print(
                f'loss = {loss_num}, precision = {precision}, recall = {recall}, f1 = {f1}')
            if ds_valid is not None:
                indexes.refresh()
                loss_num = 0
                with torch.no_grad():
                    self.model.eval()
                    with tqdm(total=n_valid, desc=f'Epoch {epoch}/{self.epochs}:vaild',
                              unit='img') as pbar:
                        for batch in valid_loader:

                            imag = batch['data'].to(
                                device=self.device, dtype=torch.float32)
                            goal = batch['label'].to(
                                device=self.device, dtype=torch.long)     # dim: N*1

                            pred = self.model(imag)
                            pred = F.sigmoid(pred)
                            loss = criterion(pred, goal)

                            true_size = imag.shape[0]
                            pred_label = pred >= 0.5
                            indexes.calCM_once(
                                pred_label.cpu().numpy(), goal.cpu().numpy())
                            loss_num += loss.item() * true_size
                            pbar.update(true_size)
                loss_num /= n_valid
                indexes_num = indexes.update()
                precision = indexes_num['precision'][1]
                recall = indexes_num['recall'][1]
                f1 = indexes_num['f1'][1]
                print(
                    f'loss = {loss_num}, precision = {precision}, recall = {recall}, f1 = {f1}')

            if epoch % 1 == 0:
                df = pd.DataFrame([epoch, loss_num, indexes_num['f1'][1],
                                   indexes_num['precision'][1], indexes_num['recall'][1]]).transpose()
                if not os.path.exists(log_dir):
                    df.to_csv(log_dir, mode='a', index=False, header=[
                              'epoch', 'loss', 'f1', 'Precision', 'Recall'])
                else:
                    df.to_csv(log_dir, mode='a', index=False, header=False)

                wgt_dir = os.path.join(self.save_path, "model_epoch_")
                if loss_num < best_loss:
                    best_loss = loss_num
                    if os.path.exists(wgt_dir + '{}.pt'.format(best_epoch)):
                        os.remove(wgt_dir + '{}.pt'.format(best_epoch))
                    torch.save(self.model.state_dict(),
                               wgt_dir + '{}.pt'.format(epoch))
                    best_epoch = epoch
                    print(f'Checkpoint Epoch-{epoch} saved !')

    def predict(self, batchsize=None, data_path=None):
        if data_path is None:
            data_path = os.path.join(self.data_path, 'val', 'images')
        save_path = os.path.join(self.save_path, 'perdict')
        save_view_path = os.path.join(self.save_path, 'perdict_view')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_view_path):
            os.makedirs(save_view_path)
        ds = WaterPredictDataSet(data_path)
        data_loader = DataLoader(ds, batch_size=self.batchsize if batchsize is None else batchsize,
                                 num_workers=0, pin_memory=True)
        n_test = len(ds)
        with torch.no_grad():
            self.model.eval()
            with tqdm(total=n_test, desc=f'Predict Images', unit='img') as pbar:
                for batch in data_loader:

                    imag = batch['data'].to(
                        device=self.device, dtype=torch.float32)

                    name = batch['name']
                    pred = self.model(imag)
                    pred = F.sigmoid(pred)

                    true_size = imag.shape[0]
                    pred_label = pred >= 0.5

                    for image_single, name_single in zip(pred_label, name):
                        image = image_single.cpu().numpy().astype(np.uint8).squeeze(0)
                        cv2.imwrite(f'{save_path}/{name_single}.png', image)
                        cv2.imwrite(
                            f'{save_view_path}/{name_single}.png', image*255)
                    pbar.update(true_size)


if __name__ == "__main__":
    data_path_list = ['./Track1', #原始数据
                      './data/npy', #原始数据的numpy格式
                      ]
    data_path = data_path_list[1]
    ds_train = WaterTrainDataSet(data_path, 0, -200)
    ds_valid = WaterTrainDataSet(data_path, -200)
    net = UNet(2, 1)
    net.load_state_dict(torch.load('./results/result8/model_epoch_93.pt'))
    model = WaterDetection(data_path, './results/result10', net,
                           batchsize=4, lr =0.001, epochs=200)
    model.train(ds_train, ds_valid)
    # model.predict(batchsize=1)
