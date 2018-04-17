# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 18:22:59 2018

@author: cyranaouameur
"""

import numpy as np
import torch
#import torch.onnx
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
                 h1_dim,
                 h2_dim,
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
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.z_dim = z_dim
        self.mb_size = mb_size
        
        #ENCODER LAYERS
        self.xh1 = nn.Linear(x_dim, h1_dim)
        self.enc_norm1 = nn.BatchNorm1d(h1_dim)
        self.h1h2 = nn.Linear(h1_dim, h2_dim)
        self.enc_norm2 = nn.BatchNorm1d(h2_dim)
        self.hz_mu = nn.Linear(h2_dim, z_dim)
        self.hz_logvar = nn.Linear(h2_dim, z_dim)
    
        #DECODER LAYERS
        self.zh2 = nn.Linear(z_dim, h2_dim)
        self.dec_norm1 = nn.BatchNorm1d(h2_dim)
        self.h2h1 = nn.Linear(h2_dim, h1_dim)
        self.dec_norm2 = nn.BatchNorm1d(h1_dim)
        self.hx_mu = nn.Linear(h1_dim, x_dim)
        self.hx_logvar = nn.Linear(h1_dim, x_dim)
        
        self.use_cuda = use_cuda


        
    def encode(self, x):
        h1 = F.relu6(self.xh1(x))
        h1 = self.enc_norm1(h1)
        h2 = F.relu6(self.h1h2(h1))
        h2 = self.enc_norm2(h2)
        z_mu = self.hz_mu(h2)
        z_logvar = self.hz_logvar(h2)
        return z_mu, z_logvar
    
    
    def decode(self, z):
        h2 = F.relu6(self.zh2(z))
        h2 = self.dec_norm1(h2)
        h1 = self.h2h1(h2)
        h1 = self.dec_norm2(h1)
        x_mu = self.hx_mu(h1)
        x_logvar = self.hx_logvar(h1)
        return x_mu, x_logvar
    

    def G_loss(self, x, x_recon_mu, x_recon_logvar, z_mu, z_logvar):
        
        #z_sigma = z_var.sqrt()+1e-8
        
        recon= x_recon_logvar.add(np.log(2.0 * np.pi))+  (x-x_recon_mu).pow(2).div(torch.exp(x_recon_logvar) + 1e-8)
        recon = 0.5 * torch.sum(recon,1)
        recon = torch.mean(recon)
        #recon /= (self.mb_size * self.x_dim)
    
        kl_loss = torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar
        kl_loss = 0.5 * torch.sum(kl_loss,1) #no size average
        kl_loss = torch.mean(kl_loss)
        
        #loss = recon + kl_loss
        return recon, kl_loss
    
    
    def forward(self, x, sample = True):
        z_mu, z_logvar = self.encode(x)
        if sample:    
            z = sample_z(z_mu,z_logvar, self.mb_size, self.z_dim) 
        else :
            z = z_mu
        x_recon = self.decode(z)
        return x_recon[0]
    
    def do_train(self, trainloader, n_epoch, wu_time):
        
        self.train()
        
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
        
        epoch_size = 600
        
        for epoch in range(n_epoch):
            
            if epoch < wu_time:
                beta = epoch / wu_time
            else :
                beta = 1
            
            
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_KL = 0.0
            
            for i, data in enumerate(trainloader):
                
                #iter = epoch*600 + i
                
                # get the inputs
                raw_inputs, labels = data
                
                inputs = raw_inputs.view(self.mb_size,self.x_dim)
        
                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = (inputs*2)-1 #normal rescale
        
                # zero the parameter gradients
                optimizer.zero_grad()
            
                x = inputs
                if self.use_cuda:
                    x = x.cuda()
                
                z_mu, z_logvar = self.encode(x)
                z = sample_z(z_mu,z_logvar, self.mb_size, self.z_dim, self.use_cuda) 
                x_recon_mu, x_recon_logvar = self.decode(z)

                recon_loss, kl_loss = self.G_loss(x, x_recon_mu, x_recon_logvar, z_mu, z_logvar)
                
                loss = recon_loss + beta*kl_loss
                
                if i == epoch_size-1 :
                    if self.use_tensorboard:
                        #TENSORBOARD VISUALIZATION
                        for name, param in self.named_parameters():
                            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch+1)
                        
                        epoch_loss += loss.data[0]
                        epoch_recon += recon_loss.data[0]
                        epoch_KL += kl_loss.data[0]
                            
                        self.writer.add_scalars('avglosses', {'loss': epoch_loss/epoch_size,
                                                           'Recon_loss': epoch_recon/epoch_size,
                                                           'KL_loss': epoch_KL/epoch_size},
                                                            epoch+1)
                        if np.mod(epoch,50) == 0:
                            for j in range(2):
                                original = inputs[j]
                                original = original.view(28,28)
                                self.writer.add_image('Original_'+str(epoch)+str(j), original, epoch)
                                
                                img_rec = x_recon_mu[j]
                                img_rec = img_rec.view(28,28)
                                self.writer.add_image('Reconstructed_'+str(epoch)+str(j), img_rec, epoch)
                
                #Annealing
                
    
                # BACKPROP
                loss.backward()
                for layer in self.parameters():
                    layer.grad[layer.grad>1e3] = 1e3
                optimizer.step()
                
            #end of epoch 
            scheduler.step()
            print('[End of epoch %d] \n beta : %.3f \n loss: %.3f \n recon_loss: %.3f \n KLloss: %.3f \n -----------------' %
                          (epoch + 1,
                           beta,
                           epoch_loss, 
                           epoch_recon, 
                           epoch_KL ))
                
        if self.use_tensorboard:
            self.writer.close()
        print("Finished")
        return True

