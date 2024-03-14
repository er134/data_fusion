import os
import copy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import WaterSemiDataSet, WaterTrainDataSet
from data_process import npy_sar_class
from loss import ModifiedOhemLoss
from main import WaterDetection
from models import UNet, MultiHead
from utils import build_imglist, metrics

class SemiWaterDetection(WaterDetection):
    def train(self, ds_train=None, ds_valid=None, resize=True):
        if ds_train is None:
            ds_train = WaterTrainDataSet(self.data_path)
        train_loader = DataLoader(ds_train, batch_size=self.batchsize,
                                  shuffle=True, num_workers=self.num_workers, pin_memory=True)
        if ds_valid is not None:
            ds_valid_copy = copy.deepcopy(ds_valid)
            valid_loader = DataLoader(ds_valid_copy, batch_size=self.batchsize,
                                      shuffle=True, num_workers=self.num_workers, pin_memory=True)
            n_valid = len(ds_valid)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

        criterion = ModifiedOhemLoss()
        valid_log_dir = os.path.join(self.save_path, "Valid_Report.csv")
        train_log_dir = os.path.join(self.save_path, "Train_Report.csv")
        n_train = len(ds_train)
        best_loss, best_f1, best_loss_epoch, best_f1_epoch, indexes = 100, 0, 0, -1, metrics()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            indexes.refresh()
            loss_num = 0.            
            img_num = 0
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

                    certain_mask = goal <= 1
                    certain_mask = certain_mask.view(-1)
                    goal[goal>1] = 1
                    pred_label = torch.zeros_like(goal, dtype=int)
                    if isinstance(pred, list):
                        for i, item in enumerate(pred):
                            item = torch.sigmoid(item)
                            if resize:
                                pred[i] = F.interpolate(item, scale_factor=2, mode='nearest')
                        keys = [10, 30, 40, 90]
                        mask_all = torch.zeros_like(goal, dtype=int)
                        loss = 0.
                        for i, key in enumerate(keys):
                            mask_cls = mask == key
                            mask_cls = mask_cls.unsqueeze(1)
                            loss += (criterion(pred[i], goal, mask_cls) * certain_mask).mean()
                            pred_label |= ((pred[i] >= 0.5) & mask_cls)
                            mask_all |= mask_cls
                        mask_all.unsqueeze(1)
                        loss += (criterion(pred[4], goal, ~mask_all) * certain_mask).mean()
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
                    img_num += true_size
                    if img_num <= n_train:
                        indexes.calCM_once(
                            pred_label.cpu().numpy(), goal.cpu().numpy())
                    loss_num += loss.item() * true_size
                    pbar.update(true_size)
            loss_num /= n_train
            indexes_num = indexes.update()
            f1_all = indexes_num['f1']
            loss_all = loss_num
            print(f'loss = {loss_num}, {indexes_num}')
            if epoch % 1 == 0:
                self.save_accuracy(epoch, loss_num, indexes_num, train_log_dir)
                
            # if ds_valid is not None:
            #     indexes.refresh()
            #     loss_num = 0.
            #     with torch.no_grad():
            #         self.model.eval()
            #         with tqdm(total=n_valid, desc=f'Epoch {epoch}/{self.epochs}:vaild',
            #                   unit='img') as pbar:
            #             for batch in valid_loader:

            #                 imag = batch['data'].to(
            #                     device=self.device, dtype=torch.float32)
            #                 goal = batch['label'].to(
            #                     device=self.device, dtype=torch.float32)
            #                 mask = batch['mask'].to(
            #                     device=self.device, dtype=torch.long)

            #                 pred = self.model(imag)

            #                 certain_mask = goal <= 1
            #                 goal[goal>1] = 1
            #                 pred_label = torch.zeros_like(goal, dtype=int)
            #                 if isinstance(pred, list):
            #                     for i, item in enumerate(pred):
            #                         item = torch.sigmoid(item)
            #                         pred[i] = F.interpolate(item, scale_factor=2, mode='nearest')
            #                     keys = [10, 30, 40, 90]
            #                     mask_all = torch.zeros_like(goal, dtype=int)
            #                     loss = 0.
            #                     for i, key in enumerate(keys):
            #                         mask_cls = mask == key
            #                         mask_cls = mask_cls.unsqueeze(1)
            #                         loss += (criterion(pred[i], goal, mask_cls) * certain_mask).mean()
            #                         pred_label |= ((pred[i] >= 0.5) & mask_cls)
            #                         mask_all |= mask_cls
            #                     mask_all.unsqueeze(1)
            #                     loss += (criterion(pred[4], goal, ~mask_all) * certain_mask).mean()
            #                     pred_label |= ((pred[4] >= 0.5) & ~mask_all)
            #                 else:
            #                     pred = torch.sigmoid(pred)
            #                     pred = F.interpolate(pred, scale_factor=2, mode='nearest')
            #                     pred_label = pred >= 0.5
            #                     loss = criterion(pred, goal)

            #                 true_size = imag.shape[0]
            #                 indexes.calCM_once(
            #                     pred_label.cpu().numpy(), goal.cpu().numpy())
            #                 loss_num += loss.item() * true_size
            #                 pbar.update(true_size)
            # loss_num /= n_valid
            # indexes_num = indexes.update()
            # f1 = indexes_num['f1']
            # f1_all = f1
            # print(f'loss = {loss_num}, {indexes_num}')
            # loss_all = (loss_all+loss_num)/2

            wgt_dir = os.path.join(self.save_path, "model_epoch_")

            if epoch % 50 == 1 and epoch <= 1:
                self.semi_predict(dataset=ds_valid, resize=True, thres=0.9, save_path='./temp')
                print('updated train dataloader')
                ds_valid.labels = build_imglist('./temp')
                self.model = UNet(3, 1, 96, multi_head=True).to(device=self.device)
                # self.model = UNet(3, 1, 64, multi_head=True)
                # self.model.load_state_dict(torch.load(f'{wgt_dir}{best_loss_epoch}_f1.pt'))
                # self.model.to(device=self.device)
                # self.model.outc = MultiHead(64, 1).to(device=self.device)
                train_loader = DataLoader(ds_train + ds_valid, batch_size=self.batchsize,
                                shuffle=True, num_workers=self.num_workers, pin_memory=True)
                optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
                optimizer.zero_grad()
                n_train = len(ds_train) + len(ds_valid)

            if epoch % 1 == 0:
                self.save_accuracy(epoch, loss_num, indexes_num, valid_log_dir)
                if loss_num < best_loss:
                    best_loss = loss_all
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
    def semi_predict(self, dataset, resize=False, save_path=None, thres=0.9):
        if save_path is None:
            save_path = os.path.join(self.save_path, 'perdict')
        save_view_path = os.path.join(self.save_path, 'perdict_view')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_view_path):
            os.makedirs(save_view_path)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True)
        n_test = len(dataset)
        with torch.no_grad():
            self.model.eval()
            with tqdm(total=n_test, desc=f'Predict Images', unit='img') as pbar:
                for batch in data_loader:
                    imag = batch['data'].to(
                        device=self.device, dtype=torch.float32)
                    mask = batch['mask'].to(
                        device=self.device, dtype=torch.long)
                    name = batch['name']
                    true_size = imag.shape[0]
                    
                    pred = self.model(imag)
                    if isinstance(pred, list):
                        for i, item in enumerate(pred):
                            item = torch.sigmoid(item)
                            pred[i] = F.interpolate(item, scale_factor=2, mode='bilinear', antialias=True)
                        keys = [10, 30, 40, 90]
                        mask_all = torch.zeros_like(pred[0], dtype=bool)
                        pred_label = torch.zeros_like(pred[0], dtype=int)
                        for i, key in enumerate(keys):
                            mask_cls = mask == key
                            mask_cls = mask_cls.unsqueeze(1)
                            pos = ((pred[i] >= thres) & mask_cls)
                            neg = ((pred[i] <= (1-thres)) & mask_cls)
                            uncer = ((~pos) & (~neg) & mask_cls)
                            pred_label[pos] = 1
                            pred_label[neg] = 0
                            pred_label[uncer] = 2
                            mask_all |= mask_cls
                        pos = ((pred[4] >= thres) & ~mask_all)
                        neg = ((pred[4] <= (1-thres)) & ~mask_all)
                        uncer = ((~pos) & (~neg) & ~mask_all)
                        pred_label[pos] = 1
                        pred_label[neg] = 0
                        pred_label[uncer] = 2
                    else:
                        pred = torch.sigmoid(pred)
                        if resize:
                            pred = F.interpolate(pred, scale_factor=2, mode='bilinear')
                        pred_label = pred >= thres


                    for image_single, name_single in zip(pred_label, name):
                        image = image_single.cpu().numpy().astype(np.uint8).squeeze(0)
                        cv2.imwrite(f'{save_path}/{name_single}.png', image)
                        cv2.imwrite(
                            f'{save_view_path}/{name_single}.png', image * 255)
                    pbar.update(true_size)