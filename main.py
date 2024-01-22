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

from models import UNet

from data import WaterPredictDataSet, WaterTrainDataSet
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
            ds_train = WaterTrainDataSet(self.data_path)
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
        criterion = FocalLoss(gamma=2, alpha=0.07)
        log_dir = os.path.join(self.save_path, "Report.csv")
        n_train = len(ds_train)
        best_loss, best_f1, best_loss_epoch, best_f1_epoch, indexes = 100, 0, 0, -1, metrics()
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
                    # pred = torch.sigmoid(pred)
                    loss = criterion(pred, goal)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    true_size = imag.shape[0]
                    pred_label = pred.argmax(1)
                    indexes.calCM_once(
                        pred_label.cpu().numpy(), goal.cpu().numpy())
                    loss_num += loss.item() * true_size
                    pbar.update(true_size)
            lr_scheduler.step()
            loss_num /= n_train
            indexes_num = indexes.update()
            precision = indexes_num['precision'][1]
            recall = indexes_num['recall'][1]
            f1 = indexes_num['f1'][1]
            f1_all = f1
            loss_all = loss_num
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
                            # pred = torch.sigmoid(pred)
                            loss = criterion(pred, goal)

                            true_size = imag.shape[0]
                            pred_label = pred.argmax(1)
                            indexes.calCM_once(
                                pred_label.cpu().numpy(), goal.cpu().numpy())
                            loss_num += loss.item() * true_size
                            pbar.update(true_size)
                loss_num /= n_valid
                indexes_num = indexes.update()
                precision = indexes_num['precision'][1]
                recall = indexes_num['recall'][1]
                f1 = indexes_num['f1'][1]
                f1_all += f1
                print(
                    f'loss = {loss_num}, precision = {precision}, recall = {recall}, f1 = {f1}')
                loss_all = (loss_all+loss_num)/2

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
                    if os.path.exists(f'{wgt_dir}{best_loss_epoch}_loss.pt'):
                        os.remove(f'{wgt_dir}{best_loss_epoch}_loss.pt')
                    torch.save(self.model.state_dict(),
                               f'{wgt_dir}{epoch}_loss.pt')
                    best_loss_epoch = epoch
                    print(f'Checkpoint Epoch-{epoch} saved !')
                if f1_all > best_f1:
                    best_f1 = f1_all
                    if os.path.exists(f'{wgt_dir}{best_f1_epoch}_f1.pt'):
                        os.remove(f'{wgt_dir}{best_f1_epoch}_f1.pt')
                    torch.save(self.model.state_dict(),
                               f'{wgt_dir}{epoch}_f1.pt')                   
                    best_f1_epoch = epoch
                    print(f'Checkpoint Epoch-{epoch} saved !')
                if os.path.exists(f'{wgt_dir}last.pt'):
                        os.remove(f'{wgt_dir}last.pt')
                torch.save(self.model.state_dict(),
                            f'{wgt_dir}last.pt')

    def predict(self, data_process, batchsize=None, data_path=None):
        if data_path is None:
            data_path = os.path.join(self.data_path, 'val', 'images')
        save_path = os.path.join(self.save_path, 'perdict')
        save_view_path = os.path.join(self.save_path, 'perdict_view')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_view_path):
            os.makedirs(save_view_path)
        ds = WaterPredictDataSet(data_path,data_process)
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
                    pred_label = pred >=0.05

                    for img_pro, image_single, name_single in zip(pred, pred_label, name):
                        image = image_single.cpu().numpy().astype(np.uint8).squeeze(0)
                        pro = img_pro.cpu().numpy() *255
                        pro = pro.astype(np.uint8).squeeze(0)
                        cv2.imwrite(f'{save_path}/{name_single}.png', image)
                        cv2.imwrite(
                            f'{save_view_path}/{name_single}.png', pro)
                    pbar.update(true_size)
    
