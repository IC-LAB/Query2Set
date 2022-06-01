# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize

class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=7),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.regressor = nn.Sequential(
            nn.Linear(32*46*46, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 3*2)
        )
        self.regressor[3].weight.data.zero_()
        self.regressor[3].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))
    
    def get_theta(self, img1, img2):
        x = torch.cat([img1,img2], 1)
        x = self.localization(x)
        x = x.view(-1, 32*46*46)
        theta = self.regressor(x)
        theta = theta.view(-1, 2, 3)
        return theta

    def forward(self, q, s):
        transformed_s = []
        for si in s:
            theta = self.get_theta(q, si)
            grid = F.affine_grid(theta, si.size(), align_corners=False)
            transformed_si = F.grid_sample(si, grid, align_corners=False)
            transformed_s.append(transformed_si)
        return transformed_s
    
    def get_transformed_m(self, q, s):
        transformed_m = []
        for si in s:
            theta = self.get_theta(q, si)
            grid = F.affine_grid(theta, si.size(), align_corners=False)
            mi = torch.ones_like(si)
            transformed_mi = F.grid_sample(mi, grid, align_corners=False)
            transformed_m.append(transformed_mi)
        return transformed_m

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 8, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, q, s):
        transformed_s = []
        for si in s:
            x = torch.cat([q,si], dim=1)
            conv1 = self.conv1(x) #(B, 8, 100, 100)
            conv2 = self.conv2(conv1) #(B, 16, 50, 50)
            conv3 = self.conv3(conv2) #(B, 32, 25, 25)
            conv4 = self.conv4(conv3) #(B, 32, 13, 13)
            deconv3 = self.deconv3(conv4) #(B, 32, 25, 25)
            deconv2 = self.deconv2(torch.cat([deconv3,conv3], dim=1)) #(B, 16, 50, 50)
            deconv1 = self.deconv1(torch.cat([deconv2,conv2], dim=1)) #(B, 8, 100, 100)
            output = self.deconv0(torch.cat([deconv1,conv1], dim=1)) #(B, 1, 200, 200)
            transformed_s.append(output)
        return transformed_s

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=7, stride=2),
            nn.BatchNorm2d(4),
            nn.ReLU(True),
            nn.Conv2d(4, 8, kernel_size=5, stride=2),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
    
    def forward(self, img):
        return self.extractor(img)

class Q2SAttention(nn.Module):
    def __init__(self):
        super(Q2SAttention, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.conv2 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def spatial_attention(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv2(x))
        return attention_map

    def forward(self, q, s):
        fms = [self.conv1(torch.cat([q, x],dim=1)) for x in s]
        ams = [self.spatial_attention(x).repeat(1,8,1,1) for x in fms]
        s_stack = torch.stack(s, dim=0)
        am_stack = torch.stack(ams, dim=0)
        fused_s = (s_stack * am_stack).mean(0)
        return fused_s, am_stack

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Flatten(),
            nn.Linear(3200, 32),
            nn.ReLU(True),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        return self.classifier(x)

class Q2SUNet(nn.Module):
    def __init__(self):
        super(Q2SUNet, self).__init__()
        self.transformer = UNet()
        self.extractor = Extractor()
        self.q2s_attention = Q2SAttention()
        self.classifier = Classifier()
    
    def forward(self, q, s):
        transformed_s = self.transformer(q, s)
        fm_q = self.extractor(q)
        fm_s = [self.extractor(x) for x in transformed_s]
        fused_fm_s, am_stack = self.q2s_attention(fm_q, fm_s)
        x = torch.cat([fm_q, fused_fm_s], dim=1)
        pre_y = self.classifier(x)
        return pre_y, transformed_s, fm_q, fm_s, fused_fm_s, am_stack
