# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 21:11:55 2023

@author: WiCi
"""

import math
import pylab
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import h5py
import torch.utils.data as Data
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(1,2),mode='bicubic',align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.3)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
class _Res_Block(nn.Module):
    def __init__(self):
        super(_Res_Block, self).__init__()
        n_feat=96
        kernel_size = 3
        self.res_conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=1)
        self.res_conv2 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=1)
        # self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.bt = nn.BatchNorm2d(n_feat)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, a=1):

        y = self.bt(self.relu(self.res_conv1(x)))
        y = self.res_conv2(y)
        y *= a
        y = torch.add(y, x)
        # y = self.relu(y)
        return y

    

class MDSR(nn.Module):
    def __init__(self):
        super(MDSR, self).__init__()

        self.conv0 = nn.Conv2d(2, 96, (3, 3), (1, 1), padding=1)
        self.conv51 = nn.Conv2d(96, 2, (3, 3), (1, 1), (1, 1))
        self.conv52 = nn.Conv2d(96, 2, (3, 3), (1, 1), (1, 1))

        self.LReLus = nn.ReLU()
        m_body = [
            _Res_Block() for _ in range(14)
        ]
            
        m_tail1 = [
            up_conv(96, 96) for _ in range(2)
        ]
        
        m_tail2 = [
            up_conv(96, 96) for _ in range(2)
        ]

        self.body = nn.Sequential(*m_body)
        # self.tail = nn.Sequential(*m_tail)
        
        self.tail1 = nn.Sequential(*m_tail1)
        self.tail2 = nn.Sequential(*m_tail2)
       
    def forward(self, x):
#        print(x.size())
        out = self.conv0(x)

        
        out = self.body(out) # 256 x 24 x 4
        out1 = self.tail1(out)
        out2 = self.tail2(out)
        
        out1 = self.conv51(out1)
        out2 = self.conv52(out2)
        
        return out1,out2
        