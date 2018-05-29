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
        
                
        #Input convolution to create channels
        self.inputConv = nn.Conv1d(in_channels=nbClasses,
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
        x_rec = F.softmax(x_rec, dim = 1)
        return x_rec, z_mu, z_logvar
    
    
    def B_loss(self, x, x_rec, z_mu, z_logvar):
        
        recon = F.binary_cross_entropy(x_rec, x, size_average=True)
        
        kl_loss = 0.5*(-z_logvar+torch.exp(z_logvar)+z_mu.pow(2)-1.) # prior is unit gaussian here
        kl_loss = torch.mean(torch.sum(kl_loss,1))
        
        loss = recon + kl_loss
        return loss, recon, kl_loss
        
        