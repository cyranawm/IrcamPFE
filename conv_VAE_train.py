#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:30:07 2018

@author: cyranaouameur
"""

#import libs
#from datasets.MNIST import load_MNIST, test_MNIST

import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable

from VAE.Conv_VAE import Conv_VAE, conv_loss
from utils.dataloader import DataLoader
from aciditools.drumLearning import importDataset

try:
    import matplotlib
    from matplotlib import pyplot as plt
except:
    import sys
    sys.path.append("/usr/local/lib/python3.6/site-packages/")
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

#Compute transforms and load data
rootDirectory = ""
dataset = importDataset()
mb_size = 100
dataloader = DataLoader(dataset, mb_size)

#Define the parameters of:
    #The Conv Layers: [in_channels, out_channels, kernel_size, stride, padding]
conv1 = []
conv2 = []
conv = [conv1, conv2]
    #The Deconv Layers: [in_channels, out_channels, kernel_size, stride, padding, output_padding]
deconv1 = []
deconv2 = []
deconv = [deconv1, deconv2]

    #The MLP hidden Layers : [[in_dim,hlayer1_dim], [hlayer1_dim,hlayer2_dim], ...] 
h_dims = []
z_dim = 0

#Hyper parameters : non-linearity? batchnorm? dropout?
nnLin = " " #relu or tanh or sig 
use_cuda = torch.cuda.is_available()
use_bn = False
dropout = False # False or a prob between 0 and 1
final_beta = 1
wu_time = 100
use_tensorboard = True
  
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
    
    for i, data in enumerate(dataloader) :
        optimizer.zero_grad()
        
        #1. get the inputs and wrap them in Variable
        raw_inputs, labels = data        
        x, labels = Variable(raw_inputs), Variable(labels)
        if use_cuda:
            x = x.cuda()
        
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
        for name, param in vae.named_parameters():
            vae.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch+1)

        vae.writer.add_scalars('avglosses', {'loss': epoch_loss/epoch_size,
                                           'Recon_loss': epoch_recon/epoch_size,
                                           'KL_loss': epoch_KL/epoch_size},
                                            epoch+1)
        
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
print("MERCI DE VOTRE PATIENCE MAITRE. \n J'AI FINI L'ENTRAINEMENT ET JE NE SUIS QU'UNE VULGAIRE ACHINE ENTIEREMENT SOUMISE.")