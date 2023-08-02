# -*- coding: utf-8 -*-
"""
Created on Fri May 19 22:38:05 2023

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
from STN import channel_est
from benchmarks import LasSRN



np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

model = channel_est().to(device)

# In the STL framework, the indiviudal input data is required
class MyDataset(Dataset):
    def __init__(self):
        
        path="inHmix_28_32_128_S_32pilot.mat"
        with h5py.File(path, 'r') as file:
            train_h1 = np.transpose(np.array(file['output_da1']))
            train_h1 = train_h1.transpose([0,3,1,2])
            train_h2 = np.transpose(np.array(file['output_da2']))
            train_h2 = train_h2.transpose([0,3,1,2])

        with h5py.File(path, 'r') as file:
            train_y1 = np.transpose(np.array(file['input_da1']))
            train_y1 = train_y1.transpose([0,3,1,2])
            train_y2 = np.transpose(np.array(file['input_da2']))
            train_y2 = train_y2.transpose([0,3,1,2])
        
        self.X1 = train_y1.astype(np.float32)
        self.X2 = train_y2.astype(np.float32)
        self.Y1 = train_h1.astype(np.float32)
        self.Y2 = train_h2.astype(np.float32)

        del file
        self.len = len(self.X1)

    def __len__(self):
        # return len(self.X)
        return self.len

    def __getitem__(self, idx):
        x1 = self.X1[idx]
        x2 = self.X2[idx]
        y1 = self.Y1[idx]
        y2 = self.Y2[idx]
        return (x1, x2, y1, y2)
    
class MyDataValset(Dataset):
    def __init__(self):
        
        path="inHmix_28_32_128_S_32pilot.mat"
        with h5py.File(path, 'r') as file:
            train_h1 = np.transpose(np.array(file['output_da_test1']))
            train_h1 = train_h1.transpose([0,3,1,2])
            train_h2 = np.transpose(np.array(file['output_da_test2']))
            train_h2 = train_h2.transpose([0,3,1,2])

        with h5py.File(path, 'r') as file:
            train_y1 = np.transpose(np.array(file['input_da_test1']))
            train_y1 = train_y1.transpose([0,3,1,2])
            train_y2 = np.transpose(np.array(file['input_da_test2']))
            train_y2 = train_y2.transpose([0,3,1,2])
        
        self.X1 = train_y1.astype(np.float32)
        self.X2 = train_y2.astype(np.float32)
        self.Y1 = train_h1.astype(np.float32)
        self.Y2 = train_h2.astype(np.float32)

        del file
        self.len = len(self.X1)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x1 = self.X1[idx]
        x2 = self.X2[idx]
        y1 = self.Y1[idx]
        y2 = self.Y2[idx]
        return (x1, x2, y1, y2)
    
class MyDataset1(Dataset):
    def __init__(self):
        
        path="inHmix_28_32_128_test_S_32pilot.mat"
        with h5py.File(path, 'r') as file:
            train_h1 = np.transpose(np.array(file['Hd1']))
            train_h1 = train_h1.transpose([0,3,1,2])
            train_h2 = np.transpose(np.array(file['Hd2']))
            train_h2 = train_h2.transpose([0,3,1,2])

        with h5py.File(path, 'r') as file:
            train_y1 = np.transpose(np.array(file['Yd1']))
            train_y1 = train_y1.transpose([0,3,1,2])
            train_y2 = np.transpose(np.array(file['Yd2']))
            train_y2 = train_y2.transpose([0,3,1,2])
        
        self.X1 = train_y1.astype(np.float32)
        self.X2 = train_y2.astype(np.float32)
        self.Y1 = train_h1.astype(np.float32)
        self.Y2 = train_h2.astype(np.float32)

        del file
        self.len = len(self.X1)

    def __len__(self):
        # return len(self.X)
        return self.len

    def __getitem__(self, idx):
        x1 = self.X1[idx]
        x2 = self.X2[idx]
        y1 = self.Y1[idx]
        y2 = self.Y2[idx]
        return (x1, x2, y1, y2)

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

tr_nmse2 = []
tr_nmse3 = []
tr_nmse4 = []
nm1=[]
nm2=[]

def adjust_learning_rate(optimizer, epoch,learning_rate_init,learning_rate_final):
    lr = learning_rate_final + 0.5*(learning_rate_init-learning_rate_final)*(1+math.cos((epoch*3.14)/epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

for it in range(epochs):
    # In this framework, the initial learning rate need proper adjustment.
    lr = adjust_learning_rate(optimizer, it,5e-2,1e-5)
    model.train()
    epoch_cost = 0
    epoch_cost1 = 0
    epoch_cost2 = 0
    mb_size = 32
    iteration =0
    for i, (x1, x2, y1, y2) in enumerate(train_loader):
        iteration = iteration+1
        XE1, XE2, YE1, YE2= x1.to(device), x2.to(device), y1.to(device), y2.to(device)
        
        # Please select the desired input-output pair
        Yhat1 = model(XE1)
        # Yhat2 = model(XE2)
        
        l1 = loss_func(Yhat1, YE1)
        # l2 = loss_func(Yhat2, YE2)
        loss = l1
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_cost = epoch_cost + (loss / BATCH_SIZE)
        
    costtr.append(epoch_cost/len(train_loader))

    print('Iter-{}; Total loss: {:.4}'.format(it, loss.item()))

    with torch.no_grad():
        model.eval()
        tr_nmse1 = []
        for i, (x1, x2, y1, y2) in enumerate(val_loader):
            XE1, XE2, YE1, YE2= x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            
            Yhat1 = model(XE1)
            # Yhat2 = model(XE2)
            
            l1 = loss_func(Yhat1, YE1)    
            loss =  l1
            
            
            epoch_cost = epoch_cost + (loss / test_BATCH_SIZE)
            
            epoch_cost1 = epoch_cost1 + (l1 / test_BATCH_SIZE)
            nmsei1=np.zeros([YE1.shape[0], 1])
            nmsei2=np.zeros([YE2.shape[0], 1])
            for i1 in range(YE1.shape[0]):
                nmsei1[i1] = np.sum(np.square(np.abs(Yhat1[i1,:].cpu().detach().numpy()-YE1[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(YE1[i1,:].cpu().detach().numpy())))
                # nmsei1[i1] = np.sum(np.square(np.abs(Yhat2[i1,:].cpu().detach().numpy()-YE2[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(YE2[i1,:].cpu().detach().numpy())))

                
            tr_nmse1.append(np.mean(nmsei1))
            
        nm1.append(np.mean(tr_nmse1))
        cost1D.append(epoch_cost1/len(val_loader))

        print('Iter-{}; NMSE_R1: {:.4}'.format(it, 10*np.log10(np.mean(tr_nmse1))))
        
        if (it+1) % 10 == 0:
            net_g_model_out_path = "model/epoch_{}.pth".format(it)
            torch.save(model, net_g_model_out_path)

test_BATCH_SIZE=100
test_dataset = MyDataset1()
test_loader = DataLoader(dataset=test_dataset,
                            batch_size=test_BATCH_SIZE,
                            shuffle=False,drop_last=True)


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
    for i, (x1, x2, y1, y2) in enumerate(test_loader):
        XE1, XE2, YE1, YE2= x1.to(device), x2.to(device), y1.to(device), y2.to(device)
        
        Yhat1 = model(XE1)
        
        nmsei1=np.zeros([YE1.shape[0], 1])
        for i1 in range(YE1.shape[0]):
            nmsei1[i1] = np.sum(np.square(np.abs(Yhat1[i1,:].cpu().detach().numpy()-YE1[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(YE1[i1,:].cpu().detach().numpy())))
        nmse1 =np.mean(nmsei1)

        
        test1_nmse.append(nmse1)
        if (i+1)%10==0:
            nmse1_snr.append(np.mean(test1_nmse))
            test1_nmse=[]

nmse1_db=10*np.log10(nmse1_snr)

snrs = np.linspace(-10,20,7)
pilots = np.linspace(32,128,4)
DNMSE=[]
HNMSE=[]


plt.plot(snrs, nmse1_db,ls='-', marker='+', c='black',label='R1')
plt.legend()
plt.grid(True) 
plt.xlabel('SNR/dB')
plt.ylabel('NMSE/dB')
plt.show()
