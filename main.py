import gc
import os
from pathlib import Path
from typing import Union, Optional


import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as ttf
from torch.utils.data import DataLoader
from osgeo import gdal
from tqdm import tqdm

from models import UNet

from data import WaterPredictDataSet, WaterTrainDataSet
from loss import BCEFocalLoss, FocalLoss, ModifiedOhemLoss
from utils import metrics, CutMix


class WaterDetection:
    def __init__(self, data_path, save_path, model, epochs=100, batchsize=8, num_workers=0, lr=1e-3) -> None:
        self.data_path = data_path
        self.save_path = save_path
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.epochs = epochs
        self.lr = lr
        self.batchsize = batchsize
        self.num_workers = num_workers
        if isinstance(model, str):
            self.model = torch.load(model, map_location=self.device)
        elif isinstance(model, dict):
            for key in model.keys():
                model[key] = model[key].to(device=self.device)
            self.model = model
        else:
            self.model = model.to(device=self.device)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
            print(f'Create Running Directory: {self.save_path}')
        else:
            print(f'Running Directory already exists: {self.save_path}')

    def train(self, ds_train=None, ds_valid=None, resize = True):

        if ds_train is None:
            ds_train = WaterTrainDataSet(self.data_path)
        train_loader = DataLoader(ds_train, batch_size=self.batchsize,
                                  shuffle=True, num_workers=self.num_workers, pin_memory=True)
        if ds_valid is not None:
            valid_loader = DataLoader(ds_valid, batch_size=self.batchsize,
                                      shuffle=True, num_workers=self.num_workers, pin_memory=True)
            n_valid = len(ds_valid)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.epochs, eta_min=0, last_epoch=-1)
        criterion = ModifiedOhemLoss()
        valid_log_dir = os.path.join(self.save_path, "Valid_Report.csv")
        train_log_dir = os.path.join(self.save_path, "Train_Report.csv")
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
                        device=self.device, dtype=torch.float32)
                    mask = batch['mask'].to(
                        device=self.device, dtype=torch.long)
                    
                    pred = self.model(imag)

                    pred_label = torch.zeros_like(goal, dtype=int)
                    if isinstance(pred, list):
                        for i, item in enumerate(pred):
                            item = torch.sigmoid(item)
                            pred[i] = F.interpolate(item, scale_factor=2, mode='nearest')
                        keys = [10, 30, 40, 90]
                        mask_all = torch.zeros_like(goal, dtype=int)
                        loss = 0.
                        for i, key in enumerate(keys):
                            mask_cls = mask == key
                            mask_cls = mask_cls.unsqueeze(1)
                            loss += criterion(pred[i], goal, mask_cls)
                            pred_label |= ((pred[i] >= 0.5) & mask_cls)
                            mask_all |= mask_cls
                        mask_all.unsqueeze(1)
                        loss += criterion(pred[4], goal, ~mask_all)
                        pred_label |= ((pred[4] >= 0.5) & ~mask_all)
                    else:
                        pred = torch.sigmoid(pred)
                        pred = F.interpolate(pred, scale_factor=2, mode='nearest')
                        pred_label = pred >= 0.5
                        loss = criterion(pred, goal)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    true_size = imag.shape[0]
                    indexes.calCM_once(
                        pred_label[mask_all > 0].cpu().numpy(), goal[mask_all > 0].cpu().numpy())
                    loss_num += loss.item() * true_size
                    pbar.update(true_size)
            # lr_scheduler.step()
            loss_num /= n_train
            indexes_num = indexes.update()
            f1_all = indexes_num['f1']
            loss_all = loss_num
            print(f'loss = {loss_num}, {indexes_num}')
            if epoch % 1 == 0:
                self.save_accuracy(epoch, loss_num, indexes_num, train_log_dir)

            if ds_valid is not None:
                indexes.refresh()
                loss_num = 0.
                with torch.no_grad():
                    self.model.eval()
                    with tqdm(total=n_valid, desc=f'Epoch {epoch}/{self.epochs}:vaild',
                              unit='img') as pbar:
                        for batch in valid_loader:

                            imag = batch['data'].to(
                                device=self.device, dtype=torch.float32)
                            goal = batch['label'].to(
                                device=self.device, dtype=torch.float32)
                            mask = batch['mask'].to(
                                device=self.device, dtype=torch.long)

                            pred = self.model(imag)

                            pred_label = torch.zeros_like(goal, dtype=int)
                            if isinstance(pred, list):
                                for i, item in enumerate(pred):
                                    item = torch.sigmoid(item)
                                    pred[i] = F.interpolate(item, scale_factor=2, mode='nearest')
                                keys = [10, 30, 40, 90]
                                mask_all = torch.zeros_like(goal, dtype=int)
                                loss = 0.
                                for i, key in enumerate(keys):
                                    mask_cls = mask == key
                                    mask_cls = mask_cls.unsqueeze(1)
                                    loss += criterion(pred[i], goal, mask_cls)
                                    pred_label |= ((pred[i] >= 0.5) & mask_cls)
                                    mask_all |= mask_cls
                                mask_all.unsqueeze(1)
                                loss += criterion(pred[4], goal, ~mask_all)
                                pred_label |= ((pred[4] >= 0.5) & ~mask_all)
                            else:
                                pred = torch.sigmoid(pred)
                                pred = F.interpolate(pred, scale_factor=2, mode='nearest')
                                pred_label = pred >= 0.5
                                loss = criterion(pred, goal)

                            true_size = imag.shape[0]
                            indexes.calCM_once(
                                pred_label[mask_all > 0].cpu().numpy(), goal[mask_all > 0].cpu().numpy())
                            loss_num += loss.item() * true_size
                            pbar.update(true_size)
                loss_num /= n_valid
                indexes_num = indexes.update()
                f1 = indexes_num['f1']
                f1_all = f1
                print(f'loss = {loss_num}, {indexes_num}')
                loss_all = (loss_all+loss_num)/2

            if epoch % 1 == 0:
                self.save_accuracy(epoch, loss_num, indexes_num, valid_log_dir)
                wgt_dir = os.path.join(self.save_path, "model_epoch_")
                if loss_num < best_loss:
                    best_loss = loss_num
                    self.update_checkpoint(
                        f'{wgt_dir}{best_loss_epoch}_loss.pt', f'{wgt_dir}{epoch}_loss.pt')
                    best_loss_epoch = epoch
                if f1_all > best_f1:
                    best_f1 = f1_all
                    self.update_checkpoint(
                        f'{wgt_dir}{best_f1_epoch}_f1.pt', f'{wgt_dir}{epoch}_f1.pt')
                    best_f1_epoch = epoch
                self.update_checkpoint(
                    f'{wgt_dir}last.pt', f'{wgt_dir}last.pt', False)

    def predict(self, data_process=None, batchsize=None, dataset=None, data_path=None, resize=False, TTA=True, save_path=None):
        if data_path is None:
            data_path = os.path.join(self.data_path, 'val', 'images')
        if save_path is None:
            save_path = os.path.join(self.save_path, 'perdict')
        save_view_path = os.path.join(self.save_path, 'perdict_view')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_view_path):
            os.makedirs(save_view_path)
        if dataset is None:
            ds = WaterPredictDataSet(data_path, data_process, augment=False, resize=resize)
        else:
            ds = dataset
        data_loader = DataLoader(ds, batch_size=self.batchsize if batchsize is None else batchsize,
                                 num_workers=0, pin_memory=True)
        n_test = len(ds)
        with torch.no_grad():
            if isinstance(self.model, dict):
                keys = self.model.keys()
                for key in keys:
                    self.model[key].eval()
            else:
                self.model.eval()
            with tqdm(total=n_test, desc=f'Predict Images', unit='img') as pbar:
                for batch in data_loader:
                    imag = batch['data'].to(
                        device=self.device, dtype=torch.float32)
                    mask = batch['mask'].to(
                        device=self.device, dtype=torch.long)
                    name = batch['name']
                    true_size = imag.shape[0]

                    if isinstance(self.model, dict):
                        b, _, h, w = imag.shape
                        pred_label = torch.zeros(
                            (b, h, w),
                            dtype=int,
                            device=self.device)
                        keys = self.model.keys()
                        mask_all = torch.zeros(
                            (b, h, w),
                            dtype=int,
                            device=self.device)
                        for key in keys:
                            mask_cls = mask == key
                            pred = self.model[key](imag)
                            mask_label = mask_cls & pred.argmax(1)
                            pred_label |= mask_label
                            mask_all |= mask_cls
                        pred = self.model[-1](imag)
                        mask_label = ~mask_all & pred.argmax(1)
                        pred_label |= mask_label
                        mask_water = mask == 80
                        pred_label |= mask_water
                    elif isinstance(self.model, nn.Module):
                        pred = self.model(imag)
                        if TTA:
                            img_h = ttf.hflip(imag)
                            img_v = ttf.vflip(imag)
                            img_hv = ttf.vflip(ttf.hflip(imag))
                            pred_h = self.model(img_h)
                            pred_v = self.model(img_v)
                            pred_hv = self.model(img_hv)
                            if isinstance(pred, list):
                                for i in range(5):
                                    pred[i] += ttf.hflip(pred_h[i])
                                    pred[i] += ttf.vflip(pred_v[i])
                                    pred[i] += ttf.hflip(ttf.vflip(pred_hv[i]))
                                    pred[i] /= 4
                            else:
                                pred += ttf.hflip(pred_h)
                                pred += ttf.vflip(pred_v)
                                pred += ttf.hflip(ttf.vflip(pred_hv))
                                pred /= 4
                        if isinstance(pred, list):
                            for i, item in enumerate(pred):
                                item = torch.sigmoid(item)
                                pred[i] = F.interpolate(item, scale_factor=2, mode='bilinear', antialias=True)
                            keys = [10, 30, 40, 90]
                            thres = [0.9, 0.9, 0.9, 0.9]
                            mask_all = torch.zeros_like(pred[0], dtype=int)
                            pred_label = torch.zeros_like(pred[0], dtype=int)
                            for i, key in enumerate(keys):
                                mask_cls = mask == key
                                mask_cls = mask_cls.unsqueeze(1)
                                pred_label |= ((pred[i] >= thres[i]) & mask_cls)
                                mask_all |= mask_cls
                            pred_label |= ((pred[4] >= 0.9) & ~mask_all)
                        else:
                            pred = torch.sigmoid(pred)
                            if resize:
                                pred = F.interpolate(pred, scale_factor=2, mode='bilinear')
                            pred_label = pred >= 0.5
                        # mask_water = mask == 80
                        # pred_label |= mask_water

                    for image_single, name_single in zip(pred_label, name):
                        image = image_single.cpu().numpy().astype(np.uint8).squeeze(0)
                        cv2.imwrite(f'{save_path}/{name_single}.png', image)
                        cv2.imwrite(
                            f'{save_view_path}/{name_single}.png', image * 255)
                    pbar.update(true_size)

    def save_accuracy(self, epoch, loss, metric, log_dir):
        df = pd.DataFrame([epoch, loss, metric['f1'],
                           metric['precision'], metric['recall']]).transpose()
        if not os.path.exists(log_dir):
            df.to_csv(log_dir, mode='a', index=False, header=[
                'epoch', 'loss', 'f1', 'Precision', 'Recall'])
        else:
            df.to_csv(log_dir, mode='a', index=False, header=False)

    def update_checkpoint(self, old_name, new_name, is_print=True):
        if os.path.exists(old_name):
            os.remove(old_name)
        torch.save(self.model.state_dict(), new_name)
        if is_print:
            print(f'Checkpoint {Path(new_name).stem} saved !')
