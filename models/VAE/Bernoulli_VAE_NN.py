#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:22:59 2018

@author: cyranaouameur
"""


import numpy as np
import torch
import torch.onnx
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as data_utils


import torch.optim as optim
#import matplotlib.pyplot as plt

def xavier_init(size):
    """Xavier init to initialize Tensor in Encoder/Decoder's nets"""
    in_dim = size[0]
    xavier_stddev = 1 / np.sqrt(in_dim / 2.)
    return torch.randn(*size) * xavier_stddev

def sample_z(mu, log_var, mb_size, Z_dim):
    # Using reparameterization trick to sample from a gaussian
    eps = Variable(torch.randn(mb_size, Z_dim))
    return mu + torch.exp(log_var / 2) * eps

def value_test(x):
    if (np.nan in x.data.numpy() or np.inf in abs(x.data.numpy())):
        return 'nan'
    else:
        return ''


class Bernoulli_VAE(nn.Module):
    
    def __init__(self, x_dim, h_dim, z_dim, mb_size = 100, use_cuda = False, use_tensorboard = False):
        super(Bernoulli_VAE, self).__init__()
        
        if use_tensorboard:

            import tensorboardX
            from tensorboardX import SummaryWriter
            
            self.tensorboard = True
            self.writer = SummaryWriter()
        
        #PARAMS
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.mb_size = mb_size
        
        #ENCODER LAYERS
        self.xh = nn.Linear(x_dim, h_dim)
        self.hz_mu = nn.Linear(h_dim, z_dim)
        self.hz_var = nn.Linear(h_dim, z_dim)
    
        #DECODER LAYERS
        self.zh = nn.Linear(z_dim, h_dim)
        self.hx = nn.Linear(h_dim, x_dim)


        
    def encode(self, x):
        h = self.xh(x)
        z_mu = self.hz_mu(h)
        z_var = F.sigmoid(self.hz_var(h))
        return z_mu, z_var
    
    
    def decode(self, z):
        h = F.relu(self.zh(z))
        x = F.sigmoid(self.hx(h))
        return x
    

    def B_loss(self, x, x_recon, z_mu, z_var):
        
        recon = F.binary_cross_entropy(x_recon, x, size_average=False)
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
        
        loss = recon + kl_loss
        return loss, recon, kl_loss
    
    
    def forward(self, x, sample = True):
        z_mu, z_var = self.encode(x)
        if sample:    
            z = sample_z(z_mu,z_var, self.mb_size, self.z_dim) 
        else :
            z = z_mu
        x_recon = self.decode(z)
        return x_recon
    
    def train(self, trainloader, n_epoch):
        
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        running_loss, recon_loss, KLloss = 0.0, 0.0, 0.0
        
#        dummy_input = Variable(torch.rand(13, 1, 28, 28))
#        self.writer.add_graph(self, dummy_input)
        
        for epoch in range(n_epoch):
            
            for i, data in enumerate(trainloader):
                
                iter = epoch*600 + i
                
                # get the inputs
                raw_inputs, labels = data
                
                inputs = raw_inputs.view((1,self.mb_size,self.x_dim))
        
                # wrap them in Variable
                inputs, labels = Variable(torch.FloatTensor(1*(inputs.numpy()>0.5))), Variable(labels)
        
                # zero the parameter gradients
                optimizer.zero_grad()
            
                x = inputs
                z_mu, z_var = self.encode(x)
                z = sample_z(z_mu,z_var, self.mb_size, self.z_dim) 
                x_recon = self.decode(z)

                loss = self.B_loss(x, x_recon, z_mu, z_var)
                
                if self.tensorboard:
                #TENSORBOARD VISUALIZATION
                    for name, param in self.named_parameters():
                        self.writer.add_histogram(name, param.clone().cpu().data.numpy(), iter)
                        
                    self.writer.add_scalars('losses', {'loss': loss[0].data[0],
                                                       'Recon_loss': loss[1].data[0],
                                                       'KL_loss': loss[2].data[0]}, iter)
                
                # BACKPROP
                loss[0].backward()
                optimizer.step()
                
                # print statistics
                
                running_loss += loss[0].data[0]
                recon_loss += loss[1].data[0]
                KLloss += loss[2].data[0]
                
                print('[%d, %5d] \n loss: %.3f \n recon_loss: %.3f \n KLloss: %.3f \n -----------------' %
                          (epoch + 1, i + 1, running_loss / 100, recon_loss/100, KLloss/100 ))
                running_loss, recon_loss, KLloss = 0.0, 0.0, 0.0
        if self.tensorboard:
            self.writer.close()
        print("Finished ")
        return True

