# coding: utf-8
import os
import random
import numpy as np
from cv2 import cv2

import torch
import torch.nn as nn
import torch.optim as optim

from network import Q2SUNet

class Q2SUModel(nn.Module):
    def __init__(self):
        super(Q2SUModel, self).__init__()

        # init models
        self.q2s_net = Q2SUNet()

        # init optimizer
        self.optimizer = optim.Adam(
            params=self.q2s_net.parameters(),
            lr=0.0001,
            betas=(0.0, 0.9)
        )
    
    def process(self, q, s, label, transform_loss_weight=0.0):
        self.optimizer.zero_grad()
        # forward
        pre_y, transformed_s, fm_q, fm_s, fused_fm_s, am_stack = self.q2s_net(q, s)
        # loss
        d = torch.mean(nn.MSELoss(reduce=False)(fused_fm_s, fm_q), [1,2,3])
        transform_loss = torch.mean(label*(d**2) + (1-label)*(torch.clamp_max(0.2-d,0)**2))
        loss = nn.CrossEntropyLoss()(pre_y, label) + transform_loss_weight*transform_loss
        return pre_y, transformed_s, loss
    
    def forward(self, q, s):
        pre_y, _, _, _, _, _ = self.q2s_net(q, s)
        return pre_y
    
    def backward(self, loss=None):
        loss.backward()
        self.optimizer.step()
    
    def load(self, checkpoint_path, epoch=0):
        self.q2s_net = torch.load(os.path.join(checkpoint_path, 'q2snet_{0:04d}.pth'.format(epoch)))
    
    def save(self, checkpoint_path, epoch=0):
        torch.save(self.q2s_net, os.path.join(checkpoint_path, 'q2snet_{0:04d}.pth'.format(epoch)))
