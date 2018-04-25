#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:30:07 2018

@author: cyranaouameur
"""

#import libs
#from datasets.MNIST import load_MNIST, test_MNIST

try:
    import matplotlib
    from matplotlib import pyplot as plt
except:
    import sys
    sys.path.append("/usr/local/lib/python3.6/site-packages/")
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

import numpy as np
import sys

import torch
import torch.optim as optim
from torch.autograd import Variable

from VAE.Conv_VAE import Conv_VAE, conv_loss
sys.path.append('./aciditools/')
from aciditools.utils.dataloader import DataLoader
from aciditools.drumLearning import importDataset #Should work now



#Compute transforms and load data
task = 'instrument'

dataset = importDataset()
dataset.data = np.array(dataset.data)
dataset.metadata[task] = np.array(dataset.metadata[task])
in_shape = dataset.get(0).shape
mb_size = 100
dataloader = DataLoader(dataset, mb_size, task) 

#Define the parameters of:
    #The Conv Layers: [in_channels, out_channels, kernel_size, stride, padding]
conv1 = [1, 8, (20,10), (10,5), (2,2)]
conv2 = [8, 16, (10,5), (4,4), (0,2)]
conv = [conv1, conv2]
    #The Deconv Layers: [in_channels, out_channels, kernel_size, stride, padding, output_padding]
deconv1 = [16, 8, (10,5), (4,4), (0,2), (1,1)]
deconv2 = [8, 1, (13,9), (10,5), (0,2), (0,0)]
deconv = [deconv1, deconv2]

    #The MLP hidden Layers : [[in_dim,hlayer1_dim], [hlayer1_dim,hlayer2_dim], ...] 
h_dims = [[2016, 512]]
z_dim = 32

#Hyper parameters : non-linearity? batchnorm? dropout?
nnLin = ['relu','none'] #[h_act, out_act] with 'relu' or 'tanh' or 'elu' or 'none'
use_cuda = torch.cuda.is_available()
use_bn = True
dropout = 0.2 # False or a prob between 0 and 1
final_beta = 1
wu_time = 100
use_tensorboard = True #True, False or 'Full' (histograms)
  
#initialize the model and use cuda if available
vae = Conv_VAE(conv, h_dims, z_dim, deconv, nnLin, use_bn, dropout)
if use_cuda :
    vae.cuda()
if use_tensorboard:
    from tensorboardX import SummaryWriter
    vae.writer = SummaryWriter()
#%%Training routine

nb_epochs = 1000
vae.train()
optimizer = optim.Adam(vae.parameters(), lr=0.0001)

for epoch in range(nb_epochs):

#BETA WU
    beta = final_beta * min(1,epoch/wu_time)
########

    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_KL = 0.0
    
    for i, data in enumerate(dataloader) : #TODO : enumerate chie dans la colle OK
        optimizer.zero_grad()
        
        #1. get the inputs and wrap them in Variable
        raw_inputs, labels = data
        pre_process = np.real(raw_inputs)
        pre_process = np.log(pre_process)
        pre_process, labels = torch.from_numpy(pre_process).float(), torch.from_numpy(labels)
        if use_cuda:
            pre_process = pre_process.cuda()
        pre_process = pre_process.view(mb_size,1,in_shape[0], in_shape[1]) #TODO : verif
        x, labels = Variable(pre_process), Variable(labels)
        
        
        #2. Forward data
        rec_mu, rec_logvar, z_mu, z_logvar = vae.forward(x)

        #3. Compute losses (+validation loss)
        recon_loss, kl_loss = conv_loss(x, rec_mu, rec_logvar, z_mu, z_logvar)
        loss = recon_loss + beta*kl_loss
        
        epoch_loss += loss.data[0]
        epoch_recon += recon_loss.data[0]
        epoch_KL += kl_loss.data[0]
        
        #4. Backpropagate
        loss.backward()
        optimizer.step()

#4. TODO at the end of the epoch :
    
    epoch_size = i+1
    
    #Tensorboard log
    if use_tensorboard:
        
        vae.writer.add_scalars('avglosses', {'loss': epoch_loss/epoch_size,
                                           'Recon_loss': epoch_recon/epoch_size,
                                           'KL_loss': epoch_KL/epoch_size},
                                            epoch+1)
        if use_tensorboard == 'Full':
            for name, param in vae.named_parameters():
                vae.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch+1)
  
        
    #Saving images
    if np.mod(epoch,50) == 0:       
        fig = plt.figure()
        for idx in range(1,6):
            plt.subplot(2,5,idx)
            plt.imshow(raw_inputs[idx].clone().cpu())
            plt.subplot(2,5,5+idx)
            plt.imshow(rec_mu[idx].clone().cpu().data)
        fig.savefig('./results/images/check_epoch'+str(epoch)+'.png' )
            
    #Print stats
    print('[End of epoch %d] \n beta : %.3f \n loss: %.3f \n recon_loss: %.3f \n KLloss: %.3f \n -----------------' %
              (epoch + 1,
               beta,
               epoch_loss/epoch_size, 
               epoch_recon/epoch_size, 
               epoch_KL/epoch_size ))

#5.ToDo when everything is finished
print("MERCI DE VOTRE PATIENCE MAITRE. \n J'AI FINI L'ENTRAINEMENT ET JE NE SUIS QU'UNE VULGAIRE MACHINE ENTIEREMENT SOUMISE.")