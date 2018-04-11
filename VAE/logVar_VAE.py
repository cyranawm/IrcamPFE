# -*- coding: utf-8 -*-

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


def sample_z(mu, logvar, mb_size, Z_dim, use_cuda):
    # Using reparameterization trick to sample from a gaussian
    
    if use_cuda:
        eps = Variable(torch.randn(mb_size, Z_dim))
        res = mu + torch.exp(0.5*logvar.cuda()) * eps.cuda()
    else:
        eps = Variable(torch.randn(mb_size, Z_dim))
        res = mu + torch.exp(0.5*logvar) * eps
    
    return res


class Vanilla_VAE(nn.Module):
    
    def __init__(self, 
                 x_dim, 
                 h_dim, 
                 z_dim, 
                 mb_size = 100, 
                 use_cuda = False, 
                 use_tensorboard = False):
        
        super(Vanilla_VAE, self).__init__()
        
        self.use_tensorboard = use_tensorboard
        
        if use_tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter()
        
        #PARAMS
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.mb_size = mb_size
        
        #ENCODER LAYERS
        self.xh = nn.Linear(x_dim, h_dim)
        self.hz_mu = nn.Linear(h_dim, z_dim)
        self.hz_logvar = nn.Linear(h_dim, z_dim)
    
        #DECODER LAYERS
        self.zh = nn.Linear(z_dim, h_dim)
        self.hx_mu = nn.Linear(h_dim, x_dim)
        self.hx_logvar = nn.Linear(h_dim, x_dim)
        
        self.use_cuda = use_cuda


        
    def encode(self, x):
        h = self.xh(x)
        z_mu = self.hz_mu(h)
        z_logvar = self.hz_logvar(h)
        return z_mu, z_logvar
    
    
    def decode(self, z):
        h = F.relu(self.zh(z))
        x_mu = F.sigmoid(self.hx_mu(h))
        x_logvar = self.hx_logvar(h)
        return x_mu, x_logvar
    

    def G_loss(self, x, x_recon_mu, x_recon_logvar, z_mu, z_logvar):
        
        #z_sigma = z_var.sqrt()+1e-8
        
        recon= x_recon_logvar.add(np.log(2 * np.pi))+  (x-x_recon_mu).pow(2).div(torch.exp(x_recon_logvar) + 1e-8)
        recon = 0.5 * torch.sum(recon,1)
        recon = torch.mean(recon)
        #recon /= (self.mb_size * self.x_dim)
    
        kl_loss = torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar
        kl_loss = 0.5 * torch.sum(torch.sum(kl_loss,1)) #no size average
        kl_loss = torch.mean(kl_loss)
        
        loss = recon + kl_loss
        return loss, recon, kl_loss
    
    
    def forward(self, x, sample = True):
        z_mu, z_logvar = self.encode(x)
        if sample:    
            z = sample_z(z_mu,z_logvar, self.mb_size, self.z_dim) 
        else :
            z = z_mu
        x_recon = self.decode(z)
        return x_recon[0]
    
    def train(self, trainloader, n_epoch):
        
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.1)
        
#        dummy_input = Variable(torch.rand(13, 1, 28, 28))
#        self.writer.add_graph(self, dummy_input)
        
        epoch_size = 600
        
        for epoch in range(n_epoch):
            
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_KL = 0.0
            
            for i, data in enumerate(trainloader):
                
                #iter = epoch*600 + i
                
                running_loss, recon_loss, KLloss = 0.0, 0.0, 0.0
                
                # get the inputs
                raw_inputs, labels = data
                
                inputs = raw_inputs.view(1,self.mb_size,self.x_dim)
        
                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)
        
                # zero the parameter gradients
                optimizer.zero_grad()
            
                x = inputs
                if self.use_cuda:
                    x = x.cuda()
                
                z_mu, z_logvar = self.encode(x)
                z = sample_z(z_mu,z_logvar, self.mb_size, self.z_dim, self.use_cuda) 
                x_recon_mu, x_recon_logvar = self.decode(z)

                loss = self.G_loss(x, x_recon_mu, x_recon_logvar, z_mu, z_logvar)
                
                if i == epoch_size-1 :
                    if self.use_tensorboard:
                        #TENSORBOARD VISUALIZATION
                        for name, param in self.named_parameters():
                            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch+1)
                            
                        self.writer.add_scalars('avglosses', {'loss': loss[0].data[0],
                                                           'Recon_loss': loss[1].data[0],
                                                           'KL_loss': loss[2].data[0]}, epoch+1)
                
                #Annealing
                
    
                # BACKPROP
                loss[0].backward()
                optimizer.step()
                
                # print statistics                
                running_loss += loss[0].data[0]
                recon_loss += loss[1].data[0]
                KLloss += loss[2].data[0]
                
                print('[%d, %5d] \n loss: %.3f \n recon_loss: %.3f \n KLloss: %.3f \n -----------------' %
                          (epoch + 1, 
                           i + 1, 
                           running_loss, 
                           recon_loss, 
                           KLloss ))
                
                #tensorboard plot
                if self.use_tensorboard:
                    epoch_loss += loss[0].data[0]
                    epoch_recon += loss[1].data[0]
                    epoch_KL += loss[2].data[0]
                    
                    if i == epoch_size-1 :
                        for name, param in self.named_parameters():
                            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch+1)
                            
                        self.writer.add_scalars('avglosses', {'loss': epoch_loss/epoch_size,
                                                           'Recon_loss': epoch_recon/epoch_size,
                                                           'KL_loss': epoch_KL/epoch_size},
                                                            epoch+1)
             
            scheduler.step()
                
        if self.use_tensorboard:
            self.writer.close()
        print("Finished")
        return True

