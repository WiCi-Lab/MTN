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
from model_MLP12 import channel_est
# from model_UNet import channel_est
from singleTrans import LasSRN

#
#train_set = Data.TensorDataset(X_train, y_train) 
#val_set = Data.TensorDataset(X_val, y_val) 
#test_set = Data.TensorDataset(X_test, y_test) 
#
#batchsize = 16
#
#train_loader = torch.utils.data.DataLoader(train_set, batch_size= batchsize, shuffle=True)
#val_loader = torch.utils.data.DataLoader(val_set, batch_size= batchsize, shuffle=True)
#test_loader = torch.utils.data.DataLoader(test_set, batch_size= batchsize, shuffle=True)
#mb_size=64
#minibatches = random_mini_batches(train_y1, train_h1, train_h2, mb_size)
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
        self.params = torch.nn.Parameter(params) #parameters的封装使得变量可以容易访问到

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 * torch.exp(-self.params[i]) * loss + self.params[i]
# +1避免了log 0的问题  log sigma部分对于整体loss的影响不大
        return loss_sum
    
def criterion(y_pred, y_true, log_vars):
  loss = 0
  for i in range(len(y_pred)):
    precision = torch.exp(-log_vars[i])
    diff = (y_pred[i]-y_true[i])**2. ## mse loss function
    loss += torch.sum(precision * diff + log_vars[i], -1)
  return torch.mean(loss)

#train_data = TrainData(feature_num, X, Y1, Y2)
#train_data_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

model = channel_est().to(device)
# model_path = "model/MMOE_model_P1_epoch_19.pth"
# model = torch.load(model_path).to(device) 

# # model_path = "MTL_model1_epoch100.pth"
# # model = torch.load(model_path).to(device) 

from thop import profile
inputest1 = torch.randn(1, 2, 32, 16).cuda()
# inputest2 = torch.randn(1, 2, 64, 4).cuda()

flops, params = profile(model, inputs=(inputest1,))

from thop import clever_format
flops, params = clever_format([flops, params], "%.3f")
print('flops: ', flops, 'params: ', params)


class MyDataset(Dataset):
    def __init__(self):
        
        path="inHmix_28_32_128_K2_32pilot.mat"
        with h5py.File(path, 'r') as file:
            train_h1 = np.transpose(np.array(file['output_da1']))
            train_h1 = train_h1.transpose([0,3,1,2])
            train_h2 = np.transpose(np.array(file['output_da2']))
            train_h2 = train_h2.transpose([0,3,1,2])
            # train_h3 = np.transpose(np.array(file['output_da3']))
            # train_h3 = train_h3.transpose([0,3,1,2])
            # train_h4 = np.transpose(np.array(file['output_da4']))
            # train_h4 = train_h4.transpose([0,3,1,2])
            # test_h1 = np.transpose(np.array(file['output_da_test1']))
            # test_h1 = test_h1.transpose([0,3,1,2])
            # test_h2 = np.transpose(np.array(file['output_da_test2']))
            # test_h2 = test_h2.transpose([0,3,1,2])
        #    train_h1 = np.transpose(np.array(file['output_train1']))
        #    train_h2 = np.transpose(np.array(file['output_train2']))
        #    test_h1 = np.transpose(np.array(file['output_test1']))
        #    test_h2 = np.transpose(np.array(file['output_test2']))
            
        #    test_snr = np.transpose(np.array(file['Testsnr']))
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
    
class MyDataset1(Dataset):
    def __init__(self):
        
        path="inHmix_28_32_128_test_K2_32pilot.mat"
        with h5py.File(path, 'r') as file:
            train_h1 = np.transpose(np.array(file['Hd1']))
            train_h1 = train_h1.transpose([0,3,1,2])
            train_h2 = np.transpose(np.array(file['Hd2']))
            train_h2 = train_h2.transpose([0,3,1,2])
            # test_h1 = np.transpose(np.array(file['output_da_test1']))
            # test_h1 = test_h1.transpose([0,3,1,2])
            # test_h2 = np.transpose(np.array(file['output_da_test2']))
            # test_h2 = test_h2.transpose([0,3,1,2])
        #    train_h1 = np.transpose(np.array(file['output_train1']))
        #    train_h2 = np.transpose(np.array(file['output_train2']))
        #    test_h1 = np.transpose(np.array(file['output_test1']))
        #    test_h2 = np.transpose(np.array(file['output_test2']))
            
        #    test_snr = np.transpose(np.array(file['Testsnr']))
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

BATCH_SIZE=32
train_dataset = MyDataset()
train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True)  # shuffle 标识要打乱顺序

test_BATCH_SIZE=100
test_dataset = MyDataset1()
test_loader = DataLoader(dataset=test_dataset,
                            batch_size=test_BATCH_SIZE,
                            shuffle=False,drop_last=True)  # shuffle 标识要打乱顺序

mtl = MultiTaskLossWrapper(2, model)
mtl
loss_func = nn.MSELoss().to(device)

# https://github.com/keras-team/keras/blob/master/keras/optimizers.py
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-07,weight_decay=1e-5)
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
tr_nmse1 = []
tr_nmse2 = []
tr_nmse3 = []
tr_nmse4 = []
nm1=[]
nm2=[]

def adjust_learning_rate(optimizer, epoch,learning_rate_init,learning_rate_final):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    # learning_rate_init = 1e-4
    # learning_rate_final = 1e-7
    lr = learning_rate_final + 0.5*(learning_rate_init-learning_rate_final)*(1+math.cos((epoch*3.14)/epochs))
    # lr = 0.00003* (1+math.cos(float(epoch)/TOTAL_EPOCHS*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
# model_path = "MTL_model_epoch_exam{}.pth"
# # torch.save(model, model_path)
# model = torch.load(model_path).to(device) 

for it in range(epochs):
    # adjust_learning_rate(optimizer, it)
    lr = adjust_learning_rate(optimizer, it,1e-1,1e-5)
    model.train()
    epoch_cost = 0
    epoch_cost1 = 0
    epoch_cost2 = 0
    mb_size = 32
    iteration =0
    for i, (x, y1,y2) in enumerate(train_loader):
        iteration = iteration+1
    # for iteration, minibatch in enumerate(minibatches, 1):
        XE, YE1, YE2= x.to(device), y1.to(device), y2.to(device)
        
        Yhat1, Yhat2 = model(XE)
#        Yhat1_view = Yhat1.view(-1,w1,e1,r1)
#        Yhat2_view = Yhat2.view(-1,w3,e3,r3)
#        
#        plt.imshow(XE[1,0,:,:].cpu().detach().numpy(),cmap='gray')
#        plt.imshow(Yhat1[1,0,:,:].cpu().detach().numpy(),cmap='gray')
#        plt.imshow(Yhat2[1,0,:,:].cpu().detach().numpy(),cmap='gray')
#        plt.imshow(YE1[1,0,:,:].cpu().detach().numpy(),cmap='gray')
#        plt.imshow(YE2[1,0,:,:].cpu().detach().numpy(),cmap='gray')
        
        l1 = 1*loss_func(Yhat1, YE1)    
        l2 = 1*loss_func(Yhat2, YE2)
        # loss =  (0.6*l1 + 1.4*l2)/2
        awl = AutomaticWeightedLoss(2)
        loss = awl(l1, l2)
        # inputX = [XE]
        # outputY = [YE1, YE2]
        # loss, log_vars = mtl(inputX, outputY)
        # loss.to(device)
        
        # if iteration%2==0:
        #     loss=loss_func(Yhat1, YE1)  
        # else:
        #     loss=loss_func(Yhat2, YE2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_cost = epoch_cost + (loss / BATCH_SIZE)
        
        # epoch_cost1 = epoch_cost1 + (l1 / num_minibatches)
        # epoch_cost2 = epoch_cost2 + (l2 / num_minibatches)
    costtr.append(torch.mean(epoch_cost))
    # cost1tr.append(torch.mean(epoch_cost1))
    # cost2tr.append(torch.mean(epoch_cost2))  
    print('Iter-{}; Total loss: {:.4}'.format(it, loss.item()))

    with torch.no_grad():
        model.eval()
        
#        Yhat1D, Yhat2D = model(X_train1[:500,:,:,:].to(device),X_train2[:500,:,:,:].to(device))
#        Y1_valid,Y2_valid= Y1_train[:500].to(device),Y2_train[:500].to(device)
        for i, (x, y1,y2) in enumerate(test_loader):
            XE, YE1, YE2 = x.to(device), y1.to(device), y2.to(device)
            
            Yhat1, Yhat2 = model(XE)
            
    #        Yhat1_view = Yhat1.view(-1,w1,e1,r1)
    #        Yhat2_view = Yhat2.view(-1,w3,e3,r3)
    #        
    #        plt.imshow(XE[1,0,:,:].cpu().detach().numpy(),cmap='gray')
    #        plt.imshow(Yhat1[1,0,:,:].cpu().detach().numpy(),cmap='gray')
    #        plt.imshow(Yhat2[1,0,:,:].cpu().detach().numpy(),cmap='gray')
    #        plt.imshow(YE1[1,0,:,:].cpu().detach().numpy(),cmap='gray')
    #        plt.imshow(YE2[1,0,:,:].cpu().detach().numpy(),cmap='gray')
            
            l1 = loss_func(Yhat1, YE1)    
            l2 = loss_func(Yhat2, YE2)
            loss =  (l1 + l2)/2
            # inputX = [XE]
            # outputY = [YE1, YE2]
            # loss, log_vars = mtl(inputX, outputY)
            # loss.to(device)
            
            
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
        cost1D.append(torch.mean(epoch_cost1))
        cost2D.append(torch.mean(epoch_cost2))
        costD.append(torch.mean(epoch_cost)) 
#        
        # Yhat1D, Yhat2D = model(X_valid1.to(device))
        # Y1_valid,Y2_valid= Y1_valid.to(device),Y2_valid.to(device)
        
        
        # l1D = loss_func(Yhat1D, Y1_valid)
        # l2D = loss_func(Yhat2D, Y2_valid)
        # cost1D.append(l1D)
        # cost2D.append(l2D)
        # costD.append((l1D+l2D)/2)
        
        # nmsei1=np.zeros([qt, 1])
        # nmsei2=np.zeros([qt, 1])
        # for i1 in range(qt):
        #     nmsei1[i1] = np.sum(np.square(np.abs(Yhat1D[i1,:].cpu().detach().numpy()-Y1_valid[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(Y1_valid[i1,:].cpu().detach().numpy())))
        #     nmsei2[i1] = np.sum(np.square(np.abs(Yhat1D[i1,:].cpu().detach().numpy()-Y1_valid[i1,:].cpu().detach().numpy()))) / np.sum(np.square(np.abs(Y1_valid[i1,:].cpu().detach().numpy())))
        # tr_nmse1 = np.mean(nmsei1)
        # tr_nmse2 = np.mean(nmsei2)
        # nm1.append(tr_nmse1)
        # nm2.append(tr_nmse2)
        print('Iter-{}; NMSE_R1: {:.4}'.format(it, 10*np.log10(np.mean(tr_nmse1))))
        print('Iter-{}; NMSE_R2: {:.4}'.format(it, 10*np.log10(np.mean(tr_nmse2))))
        
        if (it+1) % 10 == 0:
            net_g_model_out_path = "model/U-MLP_model_MK2_P4_epoch_{}.pth".format(it)
            torch.save(model, net_g_model_out_path)

# plt.plot(torch.tensor(costtr, device = 'cpu'),'-r',label='Tarining')
# plt.plot(torch.tensor(costD, device = 'cpu'),'-b',label='validation')
# plt.legend(loc='upper right')
# plt.ylabel('total cost')
# plt.xlabel('epochs')
# plt.show()

# plt.plot(torch.tensor(cost1tr, device = 'cpu'), '-r', torch.tensor(cost1D, device = 'cpu'), '-b')
# plt.ylabel('task 1 cost')
# plt.xlabel('epochs')
# plt.show() 

# plt.plot(torch.tensor(cost2tr, device = 'cpu'),'-r', torch.tensor(cost2D, device = 'cpu'),'-b')
# plt.ylabel('task 2 cost')
# plt.xlabel('epochs')
# plt.show()

# np.save('MTL_testcost.npy',torch.tensor(costD, device = 'cpu'))
# np.save('MTL_testcost1_8.npy',torch.tensor(cost1D, device = 'cpu'))
# np.save('MTL_testcost2_8.npy',torch.tensor(cost2D, device = 'cpu'))

# import scipy.io as sio # mat
# sio.savemat('MTL_testcost1.mat', {'a':torch.tensor(cost1D, device = 'cpu')})
# sio.savemat('MTL_testcost2.mat', {'a':torch.tensor(cost2D, device = 'cpu')})

# model_path = "model/U-MLP_model_K2_P4_epoch_49.pth"
# model = torch.load(model_path).to(device) 

# model_path = "model/MMOE_model_K4_P8_epoch_{}.pth"
# torch.save(model, model_path)

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# test_BATCH_SIZE=100
# test_dataset = MyDataset1()
# test_loader = DataLoader(dataset=test_dataset,
#                             batch_size=test_BATCH_SIZE,
#                             shuffle=False,drop_last=True)  # shuffle 标识要打乱顺序

# model.eval()
# #Yhat1D,Yhat2D = model(X1_valid.to(device),X2_valid.to(device))
# #Y1_valid,Y2_valid= Y1_valid.to(device),Y2_valid.to(device)

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
        
        
        # nmse1 = NMSE(YE1.cpu().detach().numpy(), Yhat1.cpu().detach().numpy())
        # nmse2 = NMSE(YE2.cpu().detach().numpy(), Yhat2.cpu().detach().numpy())
        # nmse3 = NMSE(YE3.cpu().detach().numpy(), Yhat3.cpu().detach().numpy())
        # nmse4 = NMSE(YE4.cpu().detach().numpy(), Yhat4.cpu().detach().numpy())
        
        test1_nmse.append(nmse1)
        test2_nmse.append(nmse2)
        if (i+1)%10==0:
            nmse1_snr.append(np.mean(test1_nmse))
            nmse2_snr.append(np.mean(test2_nmse))
            test1_nmse=[]
            test2_nmse=[]

nmse1_db=10*np.log10(nmse1_snr)
nmse2_db=10*np.log10(nmse2_snr)

# nmse1_db=10*np.log10(np.mean(nmse1_snr))
# nmse2_db=10*np.log10(np.mean(nmse2_snr))
# nmse3_db=10*np.log10(np.mean(nmse3_snr))
# nmse4_db=10*np.log10(np.mean(nmse4_snr))
snrs = np.linspace(-10,20,7)
pilots = np.linspace(32,128,4)
DNMSE=[]
HNMSE=[]
# for snr in range(9):
#     DNMSE.append(np.mean(nmse1_db[0+snr:54+snr:9]))
    
# snrNMSE=[]
# for snr in range(9):
#     HNMSE.append(np.mean(nmse2_db[0+snr:54+snr:9]))


plt.plot(snrs, nmse1_db,ls='-', marker='+', c='black',label='R1')
plt.plot(snrs, nmse2_db,ls='--', marker='o', c='black',label='R2')
plt.legend()
#plt.plot(pilots, nmse1_db)
plt.grid(True) 
plt.xlabel('SNR/dB')
plt.ylabel('NMSE/dB')
plt.show()


# import scipy.io as sio # mat
# sio.savemat('MT_H2.mat', {'a':nmse1_db})
# sio.savemat('MR_H2.mat', {'a':nmse2_db})

# MT = np.load('MT_testcost.npy',allow_pickle=True)
# MTCNN= np.load('MTCNN_testcost.npy',allow_pickle=True)
# # MTRes = np.load('MTRes_testcost.npy',allow_pickle=True)
# MTDFT= np.load('MTResDFT_testcost.npy',allow_pickle=True)
# MTDFTL1= np.load('MTResDFT_testcostL1.npy',allow_pickle=True)

# epos = np.linspace(0,100,100)
# plt.plot(epos, MT,ls='-', marker='+', c='black',label='MTL')
# plt.plot(epos, MTCNN,ls='-', marker='o', c='red',label='MTLR')
# # plt.plot(epos, MTRes,ls='-', marker='d', c='gray',label='MTRes')
# plt.plot(epos, MTDFT,ls='-', marker='s', c='blue',label='MTCNN')
# plt.plot(epos, MTDFTL1,ls='-', marker='s', c='gray',label='MTCNNL1')
# plt.legend()
# #plt.plot(pilots, nmse1_db)
# plt.grid(True) 
# plt.xlabel('SNR/dB')
# plt.ylabel('NMSE/dB')
# plt.show()

# sio.savemat('fl_outdoor3bit_H.mat', {'a':nmse1_db})
# sio.savemat('fl_outdoor3bit_D.mat', {'a':nmse2_db})
#
#def count_parameters(model):
#    return sum(p.numel() for p in model.parameters() if p.requires_grad)
#print('parameters_count:',count_parameters(model))
#
##from torchstat import stat
##stat(model.cpu(), input_size=[(2, 64, 16),(2,64,4)])
#from torchsummary import summary
#summary(model, [(2, 64, 16),(2,64,4)])
#
#from thop import profile
#inputest1 = torch.randn(1, 2, 64, 16).cuda()
#inputest2 = torch.randn(1, 2, 64, 4).cuda()
#
#flops, params = profile(model, inputs=(inputest1,inputest2))
#
#from thop import clever_format
#flops, params = clever_format([flops, params], "%.3f")
#print('flops: ', flops, 'params: ', params)
#
#c=np.load('STCNN_cost_G.npy',allow_pickle=True)
#d=np.load('STCNN_testcost_G.npy',allow_pickle=True)
#
#e=np.load('STCNN_cost_H.npy',allow_pickle=True)
#f=np.load('STCNN_testcost_H.npy',allow_pickle=True)
#
#g=np.load('MTCNN_cost.npy',allow_pickle=True)
#h=np.load('MTCNN_testcost.npy',allow_pickle=True)
#
#fig, ax = plt.subplots(1, 1)
#count = np.linspace(0,100,100)
#ax.plot(count, c, ls='-', marker='+',c='black', label='Training loss(STCNN-G)')
#ax.plot(count, d, ls='-', marker='o',c='red', label='Validation loss(STCNN-G)')
#ax.plot(count, e, ls='--', marker='+',c='gold', label='Training loss(STCNN-H)')
#ax.plot(count, f, ls='--', marker='o',c='blue', label='Validation loss(STCNN-H)')
##ax.plot(count, g, ls=':', marker='+',c='green', label='Training loss(MTCNN)')
##ax.plot(count, h, ls=':', marker='o',c='gray', label='Validation loss(MTCNN)')
#plt.legend()
#plt.ylabel('MSE')
#plt.xlabel('epochs')
#plt.show()

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('parameters_count:',count_parameters(model))

# # from torchstat import stat
# # stat(model.cpu(), input_size=[2, 64, 256])
# from torchsummary import summary
# summary(model, [(2, 64, 256)])

# # from thop import profile
# # inputest1 = torch.randn(1, 2, 64, 256).cuda()
# # flops, params = profile(model, inputs=(inputest1))

# # from thop import clever_format
# # flops, params = clever_format([flops, params], "%.3f")
# # print('flops: ', flops, 'params: ', params)

# import torch
# from ptflops import get_model_complexity_info

# with torch.cuda.device(0):
#   net = model
#   macs, params = get_model_complexity_info(net, (2, 64, 256), as_strings=True,
#                                             print_per_layer_stat=True, verbose=True)
#   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#   print('{:<30}  {:<8}'.format('Number of parameters: ', params))