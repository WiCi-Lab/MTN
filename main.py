# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:14:45 2023

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

# Import the channel estimation model
from MTN import channel_est
from MDSR import MDSR
from DRSN import MTLModel

# Basic setup
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NMSE performance metric
def NMSE(x, x_hat):
    x_real = np.reshape(x[:, 0, :, :], (len(x), -1))
    x_imag = np.reshape(x[:, 1, :, :], (len(x), -1))
    x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    x_C = x_real  + 1j * (x_imag )
    x_hat_C = x_hat_real  + 1j * (x_hat_imag )
    power = np.sum(abs(x_C) ** 2, axis=1)
    mse = np.sum(abs(x_C - x_hat_C) ** 2, axis=1)
    nmse = np.mean(mse / power)
    return nmse

# Joint loss function
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, model):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num))).to(device)

    def forward(self, inputs, targets):
        
        outputs = self.model(inputs[0])
        precision1 = torch.exp(-self.log_vars[0]).to(device)
        loss1 = torch.sum(precision1 * (targets[0] - outputs[0]) ** 2. + self.log_vars[0], -1)

        precision2 = torch.exp(-self.log_vars[1]).to(device)
        loss2= torch.sum(precision2 * (targets[1] - outputs[1]) ** 2. + self.log_vars[1], -1)
        loss = 0.5*loss1+1.5*loss2

        loss = torch.mean(loss)

        return loss, self.log_vars.data.tolist()
    
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params) 

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + self.params[i]
        return loss_sum
    
# Please switch the desired model: MTN, STN, MDSR, DRSN
model = channel_est().to(device)

# Dataset construction, please load the desired file path
# Training data
class MyDataset(Dataset):
    def __init__(self):
        
        path="inHmix_28_32_128_K2_32pilot.mat"
        with h5py.File(path, 'r') as file:
            train_h1 = np.transpose(np.array(file['output_da1']))
            train_h1 = train_h1.transpose([0,3,1,2])
            train_h2 = np.transpose(np.array(file['output_da2']))
            train_h2 = train_h2.transpose([0,3,1,2])

        with h5py.File(path, 'r') as file:
            train_y1 = np.transpose(np.array(file['input_da']))
            train_y1 = train_y1.transpose([0,3,1,2])
        
        self.X = train_y1.astype(np.float32)
        self.Y1 = train_h1.astype(np.float32)
        self.Y2 = train_h2.astype(np.float32)

        del file
        self.len = len(self.X)

    def __len__(self):
        # return len(self.X)
        return self.len

    def __getitem__(self, idx):
        x = self.X[idx]
        y1 = self.Y1[idx]
        y2 = self.Y2[idx]
        return (x, y1, y2)
    
# Vadilation data
class MyDataValset(Dataset):
    def __init__(self):
        
        path="inHmix_28_32_128_K2_32pilot.mat"
        with h5py.File(path, 'r') as file:
            train_h1 = np.transpose(np.array(file['output_da_test1']))
            train_h1 = train_h1.transpose([0,3,1,2])
            train_h2 = np.transpose(np.array(file['output_da_test2']))
            train_h2 = train_h2.transpose([0,3,1,2])

        with h5py.File(path, 'r') as file:
            train_y1 = np.transpose(np.array(file['input_da_test']))
            train_y1 = train_y1.transpose([0,3,1,2])
        
        self.X = train_y1.astype(np.float32)
        self.Y1 = train_h1.astype(np.float32)
        self.Y2 = train_h2.astype(np.float32)

        del file
        self.len = len(self.X)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.X[idx]
        y1 = self.Y1[idx]
        y2 = self.Y2[idx]
        return (x, y1, y2)

# Test data    
class MyDataset1(Dataset):
    def __init__(self):
        
        path="inHmix_28_32_128_test_K2_32pilot.mat"
        with h5py.File(path, 'r') as file:
            train_h1 = np.transpose(np.array(file['Hd1']))
            train_h1 = train_h1.transpose([0,3,1,2])
            train_h2 = np.transpose(np.array(file['Hd2']))
            train_h2 = train_h2.transpose([0,3,1,2])
            
        with h5py.File(path, 'r') as file:
            train_y1 = np.transpose(np.array(file['Yd']))
            train_y1 = train_y1.transpose([0,3,1,2])
        
        self.X = train_y1.astype(np.float32)
        self.Y1 = train_h1.astype(np.float32)
        self.Y2 = train_h2.astype(np.float32)

        del file
        self.len = len(self.X)

    def __len__(self):
        # return len(self.X)
        return self.len

    def __getitem__(self, idx):
        x = self.X[idx]
        y1 = self.Y1[idx]
        y2 = self.Y2[idx]
        return (x, y1, y2)

# Training hyper-parameter
BATCH_SIZE=32
train_dataset = MyDataset()
train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True) 

Val_dataset = MyDataValset()
val_loader = DataLoader(dataset=Val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True) 

test_BATCH_SIZE=100
test_dataset = MyDataset1()
test_loader = DataLoader(dataset=test_dataset,
                            batch_size=test_BATCH_SIZE,
                            shuffle=False,drop_last=True) 

mtl = MultiTaskLossWrapper(2, model)
loss_func = nn.L1Loss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4,nesterov=True)

epochs = 50 
cost1tr = []
cost2tr = []
cost1D = []
cost2D = []
cost1ts = []
cost2ts = []
costtr = []
costD = []
costts = []
tr_nmse3 = []
tr_nmse4 = []
nm1=[]
nm2=[]

# Cosine learning decay schedule
def adjust_learning_rate(optimizer, epoch,learning_rate_init,learning_rate_final):
    lr = learning_rate_final + 0.5*(learning_rate_init-learning_rate_final)*(1+math.cos((epoch*3.14)/epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# Training loops
for it in range(epochs):
    lr = adjust_learning_rate(optimizer, it,1e-1,1e-5)
    model.train()
    epoch_cost = 0
    epoch_cost1 = 0
    epoch_cost2 = 0
    mb_size = 32
    iteration =0
    for i, (x, y1,y2) in enumerate(train_loader):
        iteration = iteration+1
        XE, YE1, YE2= x.to(device), y1.to(device), y2.to(device)
        
        Yhat1, Yhat2 = model(XE)
        
        l1 = 1*loss_func(Yhat1, YE1)    
        l2 = 1*loss_func(Yhat2, YE2)

        awl = AutomaticWeightedLoss(2)
        loss = awl(l1, l2)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_cost = epoch_cost + (loss / BATCH_SIZE)
        
        
    costtr.append(epoch_cost/len(train_loader))

    print('Iter-{}; Total loss: {:.4}'.format(it, loss.item()))
    
    # Model vadilation
    with torch.no_grad():
        model.eval()
        tr_nmse1 = []
        tr_nmse2 = []
         
        for i, (x, y1,y2) in enumerate(val_loader):
            XE, YE1, YE2 = x.to(device), y1.to(device), y2.to(device)
            
            Yhat1, Yhat2 = model(XE)
            
            
            l1 = loss_func(Yhat1, YE1)    
            l2 = loss_func(Yhat2, YE2)
            loss =  (l1 + l2)/2

            
            
            epoch_cost = epoch_cost + (loss / test_BATCH_SIZE)
            
            epoch_cost1 = epoch_cost1 + (l1 / test_BATCH_SIZE)
            epoch_cost2 = epoch_cost2 + (l2 / test_BATCH_SIZE)
            nmsei1=np.zeros([YE1.shape[0], 1])
            nmsei2=np.zeros([YE1.shape[0], 1])
            for i1 in range(YE1.shape[0]):
                nmsei1[i1] = np.sum(np.square(np.abs(Yhat1[i1,:].cpu().detach().numpy()-YE1[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(YE1[i1,:].cpu().detach().numpy())))
                nmsei2[i1] = np.sum(np.square(np.abs(Yhat2[i1,:].cpu().detach().numpy()-YE2[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(YE2[i1,:].cpu().detach().numpy())))
                
            tr_nmse1.append(np.mean(nmsei1))
            tr_nmse2.append(np.mean(nmsei2))
            
        nm1.append(np.mean(tr_nmse1))
        nm2.append(np.mean(tr_nmse2))
        cost1D.append(epoch_cost1/len(val_loader))
        cost2D.append(epoch_cost1/len(val_loader))
        costD.append(epoch_cost1/len(val_loader)) 

        print('Iter-{}; NMSE_R1: {:.4}'.format(it, 10*np.log10(np.mean(tr_nmse1))))
        print('Iter-{}; NMSE_R2: {:.4}'.format(it, 10*np.log10(np.mean(tr_nmse2))))
        
        # Model saving
        if (it+1) % 10 == 0:
            net_g_model_out_path = "model/epoch_{}.pth".format(it)
            torch.save(model, net_g_model_out_path)

# Model Testing
test1_nmse=[]
test2_nmse=[]
test3_nmse=[]
test4_nmse=[]
nmse1_snr=[]
nmse2_snr=[]
nmse3_snr=[]
nmse4_snr=[]

with torch.no_grad():
    model.eval()
    for i, (x, y1,y2) in enumerate(test_loader):
        XE, YE1, YE2= x.to(device), y1.to(device), y2.to(device)
        
        Yhat1, Yhat2 = model(XE)
        
        nmsei1=np.zeros([YE1.shape[0], 1])
        nmsei2=np.zeros([YE1.shape[0], 1])
        for i1 in range(YE1.shape[0]):
            nmsei1[i1] = np.sum(np.square(np.abs(Yhat1[i1,:].cpu().detach().numpy()-YE1[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(YE1[i1,:].cpu().detach().numpy())))
            nmsei2[i1] = np.sum(np.square(np.abs(Yhat2[i1,:].cpu().detach().numpy()-YE2[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(YE2[i1,:].cpu().detach().numpy())))
        nmse1 =np.mean(nmsei1)
        nmse2 =np.mean(nmsei2)
        
        
        test1_nmse.append(nmse1)
        test2_nmse.append(nmse2)
        
        # Saving channel estimation results under different SNR conditions
        if (i+1)%10==0:
            nmse1_snr.append(np.mean(test1_nmse))
            nmse2_snr.append(np.mean(test2_nmse))
            test1_nmse=[]
            test2_nmse=[]

# Ploting NMSE perforamnce under different SNR conditions
nmse1_db=10*np.log10(nmse1_snr)
nmse2_db=10*np.log10(nmse2_snr)
snrs = np.linspace(-10,20,7)
pilots = np.linspace(32,128,4)
DNMSE=[]
HNMSE=[]
plt.plot(snrs, nmse1_db,ls='-', marker='+', c='black',label='R1')
plt.plot(snrs, nmse2_db,ls='--', marker='o', c='black',label='R2')
plt.legend()
plt.grid(True) 
plt.xlabel('SNR/dB')
plt.ylabel('NMSE/dB')
plt.show()
