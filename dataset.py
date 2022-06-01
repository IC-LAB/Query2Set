# coding: utf-8
import os
import math
import scipy
import random
import numpy as np
from cv2 import cv2
from glob import glob
from PIL import Image
from copy import deepcopy
from PIL.ImageOps import invert

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import torch
import torch.nn.functional as F
import torchvision.transforms as TT
from torch.utils.data import Dataset, DataLoader

class Query2setFixedLenDataset(Dataset):
    def __init__(self, data_path, img_size=200, s_len=4, positive_ratio=0.5, is_train=True):
        self.data = self.loadData(data_path)
        self.img_size = img_size
        self.s_len = s_len
        self.positive_ratio = positive_ratio
        self.is_train = is_train
    
    def __len__(self):
        return math.floor(len(self.data)/self.positive_ratio)
    
    def __getitem__(self, idx):
        try:
            q, s, label = self.loadItem(idx%len(self.data))
        except:
            print('>>>> Load Error: ' + self.data[idx%len(self.data)][0])
            q, s, label = self.loadItem(0)
        return q, s, label
    
    def loadData(self, data_path):
        if isinstance(data_path, str):
            if os.path.isdir(data_path):
                data = []
                finger_list = glob(os.path.join(data_path,'*', '*'))
                finger_list.sort()
                for finger_path in finger_list:
                    file_list = glob(os.path.join(finger_path, '*.png'))
                    file_list.sort()
                    data.append(file_list)
                return data
            else:
                return []
        else:
            return []
    
    def loadItem(self, idx):
        # Positive
        if(random.uniform(0,1)<self.positive_ratio):
            label = 1
            # load images (PIL.Image)
            q_path = random.choice(self.data[idx])
            q = Image.open(q_path)
            s_list = deepcopy(self.data[idx])
            s_list.remove(q_path)
            s = [Image.open(x) for x in s_list]
        # Negative
        else:
            label = 0
            # load images (PIL.Image)
            q_path = random.choice(self.data[idx])
            q = Image.open(q_path)
            s_idx_list = list(range(len(self.data)))
            s_idx_list.remove(idx)
            s_idx = random.choice(s_idx_list)
            s_list = random.sample(self.data[s_idx], len(self.data[s_idx])-1)
            s = [Image.open(x) for x in s_list]
        
        # Fixed Length s
        shuffle_s = deepcopy(s)
        random.shuffle(shuffle_s)
        sorted_s = deepcopy(s)
        sorted_s.sort(key=lambda x:x.size[0]*x.size[1], reverse=True)
        set_len = len(shuffle_s)
        for _ in range(int(self.s_len/set_len)):
            shuffle_s.extend(sorted_s)
        s = shuffle_s[:self.s_len]

        # Invert
        q = invert(q)
        s = [invert(x) for x in s]

        # Data Augmentation
        if(self.is_train):
            train_transform = TT.Compose([
                TT.RandomHorizontalFlip(),
                TT.RandomVerticalFlip(),
                TT.RandomRotation(30, expand=False),
                TT.RandomCrop(self.img_size, pad_if_needed=True, padding_mode='constant', fill=(0,0,0))
            ])
            q = train_transform(q)
            s = [train_transform(x) for x in s]
        
        tensor_transform = TT.Compose([
            TT.Grayscale(),
            TT.ToTensor()
        ])
        q = tensor_transform(q)
        s = [tensor_transform(x) for x in s]
        label = torch.tensor(label, dtype=torch.int64)

        # Invert
        q = 1.0 - q
        s = [1.0-x for x in s]
        
        return q, s, label
    
    def createIterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=8
            )

            for item in sample_loader:
                yield item
    def __init__(self, data_path, img_size=200, s_len=4, positive_ratio=0.5, is_train=True):
        self.data = self.loadData(data_path)
        self.img_size = img_size
        self.s_len = s_len
        self.positive_ratio = positive_ratio
        self.is_train = is_train
    
    def __len__(self):
        return math.floor(len(self.data)/self.positive_ratio)
    
    def __getitem__(self, idx):
        try:
            q, s, label = self.loadItem(idx%len(self.data))
        except:
            print('>>>> Load Error: ' + self.data[idx%len(self.data)][0])
            q, s, label = self.loadItem(0)
        return q, s, label
    
    def loadData(self, data_path):
        if isinstance(data_path, str):
            if os.path.isdir(data_path):
                data = []
                finger_list = glob(os.path.join(data_path,'*'))
                finger_list.sort()
                for finger_path in finger_list:
                    file_list = glob(os.path.join(finger_path, '*.bmp'))
                    file_list.sort()
                    data.append(file_list)
                return data
            else:
                return []
        else:
            return []
    
    def loadItem(self, idx):
        # Positive
        if(random.uniform(0,1)<self.positive_ratio):
            label = 1
            # load images (PIL.Image)
            q_path = random.choice(self.data[idx])
            q = Image.open(q_path)
            s_list = deepcopy(self.data[idx])
            s_list.remove(q_path)
            s = [Image.open(x) for x in s_list]
        # Negative
        else:
            label = 0
            # load images (PIL.Image)
            q_path = random.choice(self.data[idx])
            q = Image.open(q_path)
            s_idx_list = list(range(len(self.data)))
            s_idx_list.remove(idx)
            s_idx = random.choice(s_idx_list)
            s_list = random.sample(self.data[s_idx], len(self.data[s_idx])-1)
            s = [Image.open(x) for x in s_list]
        
        # Fixed Length s
        shuffle_s = deepcopy(s)
        random.shuffle(shuffle_s)
        sorted_s = deepcopy(s)
        sorted_s.sort(key=lambda x:x.size[0]*x.size[1], reverse=True)
        s = (shuffle_s + sorted_s + sorted_s)[:self.s_len]

        # Data Transform
        tensor_transform = TT.Compose([
            TT.RandomCrop(self.img_size),
            TT.Grayscale(),
            TT.ToTensor()
        ])
        q = tensor_transform(q)
        s = [tensor_transform(x) for x in s]
        label = torch.tensor(label, dtype=torch.int64)
        
        return q, s, label
    
    def createIterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=8
            )

            for item in sample_loader:
                yield item

if __name__=='__main__':
    data_path = './dataset/fingerprint_query2set/train'
    query2set_dataset = Query2setFixedLenDataset(data_path)
    dataset_iterator = query2set_dataset.createIterator(1)
    for i in range(100):
        q, s, label = next(dataset_iterator)
        fig_num = len(s) + 1
        fig, axs = plt.subplots(1, fig_num, figsize=(fig_num*3, 3))
        axs[0].imshow(q[0,0].cpu().numpy(), cmap='gray')
        axs[0].set_title(str(label[0].cpu().numpy()))
        axs[0].axis('off')
        for j in range(fig_num-1):
            axs[j+1].imshow(s[j][0,0].cpu().numpy(), cmap='gray')
            axs[j+1].set_title('s' + str(j))
            axs[j+1].axis('off')
        fig.savefig('./sample/train_data/%03d.png'%i, bbox_inches='tight', dpi=fig.dpi, pad_inches=0.0)
        plt.close()