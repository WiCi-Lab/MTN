# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:14:45 2021

@author: 5106
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

np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BasicBlock(nn.Module):

    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1, 1))
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            self.shrinkage
        )
        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(negative_slope=0.2)(self.residual_function(x) + self.shortcut(x))
    
class resBlock(nn.Module):

    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion),
        )
        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(negative_slope=0.2,inplace=True)(self.residual_function(x) + self.shortcut(x))

class convBlock(nn.Module):

    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        #shortcut

    def forward(self, x):
        return nn.LeakyReLU(negative_slope=0.2)(self.residual_function(x))

class Shrinkage(nn.Module):
    def __init__(self,  channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        # average = torch.mean(x, dim=1, keepdim=True)
        average = x
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2).unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(1,2),mode='bicubic'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),

            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2)

        )

    def forward(self, x):
        x = self.up(x)
        return x
    
import torch.nn.functional as F

class MTLModel(torch.nn.Module):
    def __init__(self):
        super(MTLModel, self).__init__()
        
        n_feat1=64
        n_feat2=64
        kernel_size=3
        
        self.in_channels = 64

        self.conv1 = nn.Conv2d(2, n_feat2, kernel_size=3, padding=1, bias=False)
        self.ups = nn.Upsample(size=(32,128),mode='bicubic',align_corners=True)

        self.conv4_x = nn.Sequential(*[BasicBlock(n_feat2, n_feat2, stride=1)
                                                        for _ in range(3)])
        
        self.tower1 = nn.Sequential(
            BasicBlock(n_feat2,n_feat2,stride=1),
            BasicBlock(n_feat2,n_feat2,stride=1),

            BasicBlock(n_feat2,n_feat2,stride=1),

            nn.Conv2d(n_feat2, 2, kernel_size, stride=(1,1),padding=1, bias=True)
        )
        
        
        self.tower2 = nn.Sequential(
            
            BasicBlock(n_feat2,n_feat2,stride=1),
            BasicBlock(n_feat2,n_feat2,stride=1),
            BasicBlock(n_feat2,n_feat2,stride=1),

            nn.Conv2d(n_feat2, 2, kernel_size, stride=(1,1),padding=1, bias=True),
        )
        
        

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make rsnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual shrinkage block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a rsnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)



    def forward(self, x1):
        x1 = self.ups(x1)

        h_shared = self.conv1(x1)
        h_shared = self.conv4_x(h_shared)
                
        out1 = self.tower1(h_shared)
        out2 = self.tower2(h_shared)
        
        return out1, out2