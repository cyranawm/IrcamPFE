#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:49:31 2018

@author: cyranaouameur
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.Wavenet.EncoderDecoder import TemporalEncoder, TemporalDecoder

import numpy as np

class TemporalModel(nn.Module):
    
    def __init__(self,
                 nbLayers=10,
                 nbBlocks=3,
                 dilation_channels=32,
                 residual_channels=32,
                 zDim = 64,
                 nbClasses=256,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False,
                 use_cuda = False,
                 *args, **kwargs):
        
        super(TemporalModel,self).__init__()
        
        self.useCuda = use_cuda
        self.nbLayers = nbLayers
        self.nbBlocks = nbBlocks
        self.zDim = zDim
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.nbClasses = nbClasses
        self.kernel_size = kernel_size
        self.dtype = dtype
        
        if not hasattr(self, 'constructor'):
            self.constructor = {'nbLayers':nbLayers, 'nbBlocks':nbBlocks, 'zDim':zDim, 'dilation_channels':dilation_channels, 
                                'residual_channels':residual_channels, 'nbClasses':nbClasses, 'kernel_size': kernel_size, 
                                'dtype':dtype, 'bias':bias, 'args':args, 'kwargs':kwargs} # remind construction arguments for easy load/save
       
                
        #Input convolution to create channels
        self.inputConv = nn.Conv1d(in_channels=1,
                                   out_channels=residual_channels,
                                   kernel_size=1,
                                   bias=bias)
        
        self.encoder = TemporalEncoder(nbLayers,
                                     nbBlocks,
                                     dilation_channels,
                                     residual_channels,                 
                                     nbClasses,
                                     kernel_size,
                                     dtype,
                                     bias,
                                     use_cuda,
                                     *args, **kwargs)
        
        #Final encoder-side convolution
        self.enc_finalConv = nn.Conv1d(in_channels=residual_channels,
                                         out_channels=1,
                                         kernel_size=1,
                                         bias=True)
 
        self.enc_linear_mu = nn.Linear(2205,zDim) #TODO : complete
        self.enc_linear_logvar = nn.Linear(2205,zDim) #TODO : complete
        
        self.dec_linear = nn.Linear(zDim,2205) #TODO : complete
        
        self.dec_startConv = nn.Conv1d(in_channels=1,
                                         out_channels=residual_channels,
                                         kernel_size=1,
                                         bias=True)
        
        self.decoder = TemporalDecoder(nbLayers,
                                     nbBlocks,
                                     dilation_channels,
                                     residual_channels,                 
                                     nbClasses,
                                     kernel_size,
                                     dtype,
                                     bias,
                                     use_cuda,
                                     *args, **kwargs)
        
        #Input convolution to create channels
        self.outputConv = nn.Conv1d(in_channels=residual_channels,
                                   out_channels=nbClasses,
                                   kernel_size=1,
                                   bias=bias)
        
        
    def encode(self, x):
        h = self.inputConv(x)
        h = self.encoder(h)
        h = self.enc_finalConv(h)
        z_mu = self.enc_linear_mu(h)
        z_logvar = self.enc_linear_logvar(h)
        return z_mu, z_logvar
    
    
    def reparametrize(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()), requires_grad = False)
        if torch.cuda.is_available():
            eps = eps.cuda()
        return mu+torch.exp(logvar/2)*eps
    
    
    def decode(self, z):
        h = self.dec_linear(z)
        h = self.dec_startConv(h)
        h = self.decoder(h)
        x_rec = self.outputConv(h)
        return x_rec
    
    
    def forward(self, mb):
        z_mu, z_logvar = self.encode(mb)
        z = self.reparametrize(z_mu, z_logvar)
        x_rec= self.decode(z)
        x_rec = F.log_softmax(x_rec, dim = 1)
        return x_rec, z_mu, z_logvar
    
    
    def loss(self, x, x_rec, z_mu, z_logvar):
        
        list_in = torch.cat([i for i in x.squeeze(1)]).long()
        list_out = torch.cat([i for i in x_rec], dim = 1).t()
        recon = F.nll_loss(list_out, list_in, size_average=True)
        
        kl_loss = 0.5*(-z_logvar+torch.exp(z_logvar)+z_mu.pow(2)-1.) # prior is unit gaussian here
        kl_loss = torch.mean(torch.sum(kl_loss,1))
        
        return recon, kl_loss
    
    def save(self, filename, *args, **kwargs):
        if self.useCuda:
            state_dict = self.state_dict()
            for i, k in state_dict.items():
                state_dict[i] = k.cpu()
        else:
            state_dict = self.state_dict()
        constructor = dict(self.constructor)
        save = {'state_dict':state_dict, 'init_args':constructor, 'class':self.__class__}
        for k,v in kwargs.items():
            save[k] = v
        torch.save(save, filename)
        
            
    @classmethod
    def load(cls, pickle):
        init_args = pickle['init_args']
        for k,v in init_args['kwargs'].items():
            init_args[k] = v
        del init_args['kwargs']
        vae = cls(**pickle['init_args'])
        vae.load_state_dict(pickle['state_dict'])
        return vae
        
    def valid_loss(self, validset, beta, use_cuda, last_batch = False):
        
        valid_loss = 0.0
        
        self.eval()
        
        for i, data in enumerate(validset):
            #1. get the inputs and wrap them in Variable
            raw_in, labels = data
            raw_in = np.concatenate([i for i in raw_in])
            torch_in, labels = torch.from_numpy(raw_in).float(), torch.from_numpy(labels)
            if use_cuda:
                torch_in = torch_in.cuda()
            x = torch_in.unsqueeze(1)
            x, labels = Variable(x), Variable(labels)
            
            #2. Forward data
            x_rec, z_mu, z_logvar = self.forward(x)
    
            #3. Compute losses (+validation loss)
            recon_loss, kl_loss = self.loss(x, x_rec, z_mu, z_logvar)
            loss = recon_loss + beta*kl_loss
            
            valid_loss += loss.data[0]
            
        valid_loss /= i+1
        
        self.train()
        
        if last_batch:
            last_batch_in = raw_in
            last_batch_out = x_rec
            return valid_loss, last_batch_in, last_batch_out
        
        else :
            return valid_loss
        
        