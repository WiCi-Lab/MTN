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
        return nn.LeakyReLU(negative_slope=0.2)(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Shrinkage(nn.Module):
    def __init__(self,  channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(gap_size)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            # nn.BatchNorm1d(channel),
            # nn.LeakyReLU(negative_slope=0.2),
            # nn.Linear(channel, channel),
            nn.Sigmoid(),
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel),
            # nn.BatchNorm1d(channel),
            # nn.LeakyReLU(negative_slope=0.2),
            # nn.Linear(channel, channel),
            nn.Tanh(),
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
        
        x1 = self.gap(x_raw)
        x1 = torch.flatten(x1, 1)
        # average = torch.mean(x, dim=1, keepdim=True)
        x1 = self.fc1(x1)
        x1 = x1.unsqueeze(2).unsqueeze(2)
        
        x = torch.mul(average, x)
        x = x.unsqueeze(2).unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        
        
        x = x_raw * x1.expand_as(x_raw)+x
        
        
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # Squeeze
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=True),
            # nn.LeakyReLU(negative_slope=0.3),
            # nn.Linear(channel // reduction, channel, bias=True),
            nn.Tanh()
            # nn.Sigmoid()
        )

    def forward(self, x):
        
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class _Res_Blocka(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_Res_Blocka, self).__init__()
        
        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.res_conb = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        
        self.ca = SELayer(out_ch)

    def forward(self, x,al=1):

        y = self.relu(self.res_conv(x))
        y = self.relu(self.res_conb(y))
        y = self.ca(y)
        y *= al
        y = torch.add(y, x)
        return y

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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=False , drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=False)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        #self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return x
    
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

class STLModel(torch.nn.Module):
    def __init__(self):
        super(STLModel, self).__init__()
        
        n_feat1=64
        n_feat22=64
        n_feat2=64
        kernel_size=3
        
        self.in_channels = 64

        self.conv1 = nn.Conv2d(2, n_feat22, kernel_size=3, padding=1, bias=False)
        # self.conv2 = nn.Conv2d(n_feat22, n_feat2, kernel_size=3, padding=1, bias=False)

        # self.conv3_x = BasicBlock(n_feat22,n_feat22,stride=1)
        # self.conv4_x = BasicBlock(n_feat22,n_feat22,stride=1)
        # self.conv5_x = BasicBlock(n_feat2,n_feat2,stride=1)
        # self.conv6_x = BasicBlock(n_feat2,n_feat2,stride=1)
        
        self.conv3_x = nn.Sequential(*[BasicBlock(n_feat2, n_feat22, stride=1)
                                                       for _ in range(2)])
        self.conv4_x = nn.Sequential(*[BasicBlock(n_feat2, n_feat22, stride=1)
                                                        for _ in range(4)])
        self.conv5_x = nn.Sequential(*[BasicBlock(n_feat2, n_feat22, stride=1)
                                                        for _ in range(4)])
        
        d_model = 16*8*4
        d_model1 = 16*8*4*8
        nhead = 8
        d_hid = 128 
        d_hid1 = 128 
        dropout = 0.0
        nlayers = 1
        nlayers1 = 2
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        # self.transformer_encoder1 = nn.TransformerEncoder(encoder_layers, nlayers)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layers, nlayers1)

        
        self.tower1 = nn.Sequential(
            BasicBlock(n_feat2,n_feat2,stride=1),
            up_conv(n_feat2,n_feat2),
            # BasicBlock(n_feat2,n_feat2,stride=1),
            up_conv(n_feat2,n_feat2),
            # BasicBlock(n_feat2,n_feat2,stride=1),
            up_conv(n_feat2,n_feat2),
            # BasicBlock(n_feat2,n_feat2,stride=1),
            convBlock(n_feat2,n_feat2,stride=1),
            # BasicBlock(n_feat2,n_feat2,stride=1),
            # BasicBlock(n_feat2,n_feat2,stride=1),
            # nn.Conv2d(n_feat2, n_feat1, kernel_size, stride=(1,1),padding=1, bias=True),
            # nn.BatchNorm2d(n_feat1),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(n_feat2, 2, kernel_size, stride=(1,1),padding=1, bias=True)
        )
        
        # self.reweight = Mlp(n_feat2, n_feat2 // 2, n_feat22 *2)
        

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
        

        h_shared = self.conv1(x1)
        h_shared = self.conv3_x(h_shared)
        # h_shared = rearrange(h_shared, 'b c eig f -> b c (eig f)')
        # # h_shared = self.fc1(h_shared)
        # h_shared = self.transformer_encoder(h_shared)
        # h_shared = rearrange(h_shared, 'b c (eig f) -> b c eig f', eig=16)
        # h_shared = rearrange(h_shared, 'b c eig f -> b c (eig f)')
        # # h_shared = self.fc1(h_shared)
        # h_shared = self.transformer_encoder1(h_shared, None)
        # h_shared = rearrange(h_shared, 'b c (eig f) -> b c eig f', eig=32)
        
        # B, C, H, W = h_shared.shape
        # # C= C//3
        # a = h_shared
        # a = a.flatten(2)
        # a = a.mean(2)    
        # a1=a
        
        # a = self.reweight(a)
        # a = a.reshape(B, C, 2)
        # a =a.permute(2, 0, 1)
        # a = a.softmax(dim=0)
        # a=a.unsqueeze(3)
        # a=a.unsqueeze(4)
        
        # h_shared0 = h_shared
        h_shared = self.conv4_x(h_shared)
        h_shared = rearrange(h_shared, 'b c eig f -> b c (eig f)')
        # h_shared = self.fc1(h_shared)
        h_shared = self.transformer_encoder2(h_shared, None)
        h_shared = rearrange(h_shared, 'b c (eig f) -> b c eig f', eig=32)
        h_shared = self.conv5_x(h_shared)
        
        # h_shared = rearrange(h_shared, 'b c eig f -> b c (eig f)')
        # # h_shared = self.fc1(h_shared)
        # h_shared = self.transformer_encoder2(h_shared, None)
        # h_shared = rearrange(h_shared, 'b c (eig f) -> b c eig f', eig=32)
        
        # h_shared0 = h_shared
        # h_shared = self.conv4_x(h_shared)
        # h_shared = self.conv5_x(h_shared)
        # h_shared = self.conv6_x(h_shared)
        # h_shared = self.conv2(h_shared)
        
        # h_shared = h_shared * a[0] + h_shared * a[1]
                
        out1 = self.tower1(h_shared)

        return out1
    
class MTLModel(torch.nn.Module):
    def __init__(self):
        super(MTLModel, self).__init__()
        
        n_feat2=64
        n_feat22 = 64
        kernel_size=3
        
        self.in_channels = 64

        self.conv1 = nn.Conv2d(2, n_feat2, kernel_size=3, padding=1, bias=False)
        
        
        self.conv3_x = nn.Sequential(*[BasicBlock(n_feat2, n_feat2)
                                                        for _ in range(2)])
        # self.conv3_x = nn.Sequential(*[BasicBlock(in_places=n_feat2, places=32)
        #                                                 for _ in range(2)])
        self.conv4_x = nn.Sequential(*[BasicBlock(n_feat2, n_feat22, stride=1)
                                                        for _ in range(4)])
        self.conv5_x = nn.Sequential(*[BasicBlock(n_feat2, n_feat22, stride=1)
                                                        for _ in range(4)])
        
        d_model = 32*16
        # # d_model1 = 16*8*4*8
        nhead = 8
        d_hid = 128 
        # # d_hid1 = 128*4 
        dropout = 0.0
        # nlayers1 = 2
        nlayers2 = 2
        # encoder_layer1 = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        # # encoder_layer3 = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)

        # self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer1, nlayers1)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer2, nlayers2)
        
        self.tower1 = nn.Sequential(
            # BasicBlock(n_feat2,n_feat2,stride=1),
            BasicBlock(n_feat2,n_feat2,stride=1),
            # nn.Conv2d(n_feat22, n_feat2, kernel_size=3, padding=1, bias=False),
            up_conv(n_feat2,n_feat2),
            # BasicBlock(n_feat2,n_feat2,stride=1),
            up_conv(n_feat2,n_feat2),
            # BasicBlock(n_feat2,n_feat2,stride=1),
            up_conv(n_feat2,n_feat2),
            convBlock(n_feat2,n_feat2,stride=1),
            # convBlock(n_feat2,n_feat2,stride=1),
            # convBlock(n_feat2,n_feat2,stride=1),
            # nn.Conv2d(n_feat2, n_feat1, kernel_size, stride=(1,1),padding=1, bias=True),
            # nn.BatchNorm2d(n_feat1),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(n_feat2, 2, kernel_size, stride=(1,1),padding=1, bias=True)
        )
        
        self.reweight = Mlp(n_feat2, n_feat2 // 3, n_feat22 *3)
        self.reweight1 = Mlp(n_feat2, n_feat2 // 3, n_feat22 *3)

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
        

        h_shared = self.conv1(x1)
        # h_shared = self.ups(h_shared)
        h_shared = self.conv3_x(h_shared)
        
        B, C, H, W = h_shared.shape
        # C= C//3
        a = h_shared
        a = a.flatten(2)
        a = a.mean(2)    
        a1=a
        
        a = self.reweight(a)
        a = a.reshape(B, C, 3)
        a =a.permute(2, 0, 1)
        a = a.softmax(dim=0)
        a=a.unsqueeze(3)
        a=a.unsqueeze(4)
        
        b = self.reweight1(a1)
        b = b.reshape(B, C, 3)
        b = b.permute(2, 0, 1)
        b = b.softmax(dim=0)
        b = b.unsqueeze(3)
        b = b.unsqueeze(4)
        
        h_shared1 = self.conv4_x(h_shared)
        h_shared3 = self.conv5_x(h_shared)
        
        h_shared2 = rearrange(h_shared, 'b c eig f -> b c (eig f)')
        h_shared2 = self.transformer_encoder2(h_shared2)
        h_shared2 = rearrange(h_shared2, 'b c (eig f) -> b c eig f', eig=32)
        
        
        y1 = h_shared1 * a[0] + h_shared2 * a[1] + h_shared3 * a[2] 
                
        out1 = self.tower1(y1)
        return out1
    
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


# class up_conv(nn.Module):
#     """
#     Up Convolution Block
#     """
#     def __init__(self, in_ch, out_ch):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
# #            nn.ConvTranspose2d(in_ch , out_ch, kernel_size=3, stride=2, padding=1,bias=True),
#             nn.Upsample(scale_factor=(1, 2),mode='nearest'),
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             # nn.InstanceNorm2d(out_ch),
#             # nn.BatchNorm2d(out_ch),
#             # nn.LeakyReLU(negative_slope=0.3)
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.up(x)
#         return x

    
class FeatureExtraction(nn.Module):
    def __init__(self, level):
        super(FeatureExtraction, self).__init__()
        if level==1:
            self.conv0 = nn.Conv2d(2, 96, (3, 3), (1, 1), padding=1)
        else:
            self.conv0 = nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1)
        # self.res1 = _Res_Block()
        # self.res2 = _Res_Block()
        self.res3 = _Res_Block()
        # self.RCAB = RCAB(conv=nn.Conv2d,n_feat=64, kernel_size=3, reduction=16)
        # self.conv1 = nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1)
        # self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1)
        # self.conv3 = nn.Conv2d(64, 64, (3, 3), (1, 1), padding=1)
        # self.conv4 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(96, 2, (3, 3), (1, 1), (1, 1))
        # self.convt_F = nn.Upsample(size=None, scale_factor=(1, 2), mode='nearest', align_corners=None)
        # self.convt_F = nn.ConvTranspose2d(64, 64, (4, 4), (1, 2), padding=(1,1))
        # self.LReLus = nn.LeakyReLU(negative_slope=0.2)
        self.LReLus = nn.ReLU()
        # self.sig = nn.Sigmoid()
#        self.convt_F.weight.data.copy_(bilinear_upsample_weights(4, self.convt_F.weight))

        # m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            _Res_Block() for _ in range(8)
        ]
        # m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        # tail = nn.Sequential(
        #     nn.Upsample(size=None, scale_factor=(1, 2), mode='nearest', align_corners=None),
        #     nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
        #     nn.ReLU()
        #     )
            
        m_tail = [
            up_conv(96, 96) for _ in range(2)
        ]
        # self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        # self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
       
    def forward(self, x):
#        print(x.size())
        out = self.conv0(x)
#        print(out.size())
        # out1 = out
        # for i in range(3):
        #     out = self.RCAB(out) # 256 x 24 x 4
        # out +=out1
        # for i in range(2):
        #     out = self.res2(out) # 256 x 24 x 4
        # out +=out1
        
        out = self.body(out) # 256 x 24 x 4
        out = self.tail(out)
        # out = self.body(out) # 256 x 24 x 4
        # out +=out1
        # out = self.LReLus(self.conv1(out))
        # out = self.LReLus(self.conv2(out))
        # out = self.LReLus(self.conv3(out))
        # out = self.LReLus(self.conv4(out))
        # out = self.LReLus(self.conv5(out))
        # for i in range(3):
        #     out = self.LReLus(self.conv4(self.convt_F(out)))
        # out = self.LReLus(self.conv4(self.convt_F(out)))
        # out = self.LReLus(self.conv4(self.convt_F(out)))
        # out = self.LReLus(self.conv4(self.convt_F(out)))
        # out = self.LReLus(self.conv4(self.convt_F(out)))
        # out = self.LReLus(self.conv4(out))
        # out = self.LReLus(self.conv5(out))
        # out = self.LReLus(self.convt_F(out))
        # out = self.LReLus(self.conv4(out))
        # # out = self.LReLus(self.conv5(out))
        # # out = self.LReLus(self.convt_F(out))
        # # out = self.LReLus(self.conv4(out))
        # out = self.LReLus(self.convt_F(out))
        # out = self.LReLus(self.conv4(out))
        out = self.conv5(out)
        
        return out


# class ImageReconstruction(nn.Module):
#     def __init__(self):
#         super(ImageReconstruction, self).__init__()
#         self.conv_R = nn.Conv2d(64, 2, (3, 3), (1, 1), padding=1)
#         self.convt_I = nn.ConvTranspose2d(2, 2, (4, 4), (2, 2), padding=1)
#         self.convt_I.weight.data.copy_(bilinear_upsample_weights(4, self.convt_I.weight))
#         self.conv_1 = nn.Conv2d(2, 2, (3, 3), (1, 2), padding=1)
        
        
#     def forward(self, LR, convt_F):
#         convt_I = self.conv_1(self.convt_I(LR))
#         conv_R = self.conv_R(convt_F)
        
#         HR = convt_I+conv_R
#         return HR
        
        
class LasSRN(nn.Module):
    def __init__(self):
        super(LasSRN, self).__init__()
        self.FeatureExtraction1 = FeatureExtraction(level=1)
        # self.FeatureExtraction2 = FeatureExtraction(level=2)
        # self.FeatureExtraction3 = FeatureExtraction(level=3)
        # self.ImageReconstruction1 = ImageReconstruction()
        # self.ImageReconstruction2 = ImageReconstruction()
        # self.ImageReconstruction3 = ImageReconstruction()



    def forward(self, LR):
        
        convt_F1 = self.FeatureExtraction1(LR)
#        print(convt_F1.size())
        
        return convt_F1

class MDSR(nn.Module):
    def __init__(self):
        super(MDSR, self).__init__()

        self.conv0 = nn.Conv2d(2, 96, (3, 3), (1, 1), padding=1)
        self.conv51 = nn.Conv2d(96, 2, (3, 3), (1, 1), (1, 1))
        self.conv52 = nn.Conv2d(96, 2, (3, 3), (1, 1), (1, 1))

        self.LReLus = nn.ReLU()
        m_body = [
            _Res_Block() for _ in range(7)
        ]
            
        m_tail1 = [
            up_conv(96, 96) for _ in range(3)
        ]
        
        m_tail2 = [
            up_conv(96, 96) for _ in range(3)
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
        