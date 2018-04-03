#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 15:22:02 2018

@author: cyranaouameur
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:54:13 2018

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

import tensorboardX
from tensorboardX import SummaryWriter

import torch.optim as optim
import matplotlib.pyplot as plt

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
    
    def __init__(self, inputSize, h_dim, z_dim, mb_size = 100):
        super(Bernoulli_VAE, self).__init__()
        
        self.writer = SummaryWriter()
        
        #PARAMS
        self.x_dim = inputSize
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.mb_size = mb_size
        
        #ENCODER LAYERS
        self.Wxh = nn.Parameter(xavier_init([inputSize,h_dim]))
        self.Bxh = nn.Parameter(xavier_init([h_dim]))
        
        self.Whz_mu = nn.Parameter(xavier_init([h_dim,z_dim]))
        self.Bhz_mu = nn.Parameter(xavier_init([z_dim]))
    
        self.Whz_var = nn.Parameter(xavier_init([h_dim,z_dim]))
        self.Bhz_var = nn.Parameter(xavier_init([z_dim]))
        
        #DECODER LAYERS
        self.Wzh = nn.Parameter(xavier_init([z_dim,h_dim]))
        self.Bzh = nn.Parameter(xavier_init([h_dim]))
        
        self.Whx = nn.Parameter(xavier_init([h_dim,inputSize]))
        self.Bhx = nn.Parameter(xavier_init([inputSize]))


        
    def encode(self, x):
        h = F.relu(x @ self.Wxh +  self.Bxh.repeat(x.size(0), 1))
        z_mu = h @ self.Whz_mu + self.Bhz_mu.repeat(h.size(0), 1)
        z_var = F.sigmoid(h @ self.Whz_var + self.Bhz_var.repeat(h.size(0), 1))
        return z_mu, z_var
    
    
    def decode(self, z):
        h = F.relu(z @ self.Wzh + self.Bzh.repeat(z.size(0), 1))
        x = F.sigmoid(h @ self.Whx + self.Bhx.repeat(h.size(0), 1))
        
        return x
    

    def B_loss(self, x, x_recon, z_mu, z_var):
#       
        recon = F.binary_cross_entropy(x_recon, x, size_average=False)
    
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1. - z_var)
        
        loss = recon + kl_loss
        return loss, recon, kl_loss
    
    
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
                          (epoch + 1, i + 1, running_loss / 10, recon_loss/10, KLloss/10 ))
#                if i % 10 == 9:    # print every 10 mini-batches
#                    print('[%d, %5d] loss: %.3f' %
#                          (epoch + 1, i + 1, running_loss / 10))
                running_loss, recon_loss, KLloss = 0.0, 0.0, 0.0
                
                if (loss[0].data[0] == np.nan or np.abs(loss[0].data[0]) == np.inf):
                    print('ce batch engendre un nan')
                    return raw_inputs
        
        self.writer.close()
        print("Finished HIHIHIHI")
        return True

