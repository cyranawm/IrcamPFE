#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 15:51:59 2018

@author: cyranaouameur




TODO : Non-Linearity
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from copy import deepcopy

def conv_loss(target, rec_mu, rec_logvar, z_mu, z_logvar):
    ## latent variables are 1D but data variables are 2D --> modif sum/mean
    recon_loss = torch.mean(torch.sum(torch.sum(0.5*(rec_logvar+(target-rec_mu).pow(2).div(torch.exp(rec_logvar).add(1e-7))+np.log(2*np.pi)),2),1))
    
    KL_loss = 0.5*(-z_logvar+torch.exp(z_logvar)+z_mu.pow(2)-1.) # prior is unit gaussian here
    KL_loss = torch.mean(torch.sum(KL_loss,1))
    
    return recon_loss, KL_loss


    

class Conv_VAE(nn.Module):
    def __init__(self, conv_list, h_dims, z_dim, deconv_list, nnLin, use_bn, dropout ):
        super(Conv_VAE, self).__init__()
        
        activation = {
                'relu' : nn.ReLU(),
                'tanh' : nn.Tanh(),
                'elu' : nn.ELU(),
                'none' : nn.Sequential()
                }
        
        h_nnLin = activation[nnLin[0]]
        out_nnLin = activation[nnLin[1]]
        
        
        #ENCODER LAYERS
        #CONV
        self.enc_conv = nn.Sequential()
        for n, conv in enumerate(conv_list):
            name = "enc_conv"+str(n)
            layer = nn.Conv2d(conv[0],conv[1],conv[2],stride=conv[3],padding=conv[4]) #[in_channels, out_channels, kernel_size, stride, padding]
            self.enc_conv.add_module(name, layer)
            if use_bn:
                name = "enc_conv" + str(n) +"_norm"
                self.enc_conv.add_module(name,nn.BatchNorm2d(conv[1]))
            self.enc_conv.add_module("enc_conv" + str(n) +"_act",h_nnLin)
            if dropout:
                name = "enc_conv" + str(n) +"_drop"
                self.enc_conv.add_module(name, nn.Dropout2d(p=dropout))
            
                
        #MLP
        self.enc_MLP = nn.Sequential()
        for n, h_dim in enumerate(h_dims):
            name = "enc_lin"+str(n)
            layer = nn.Linear(h_dim[0], h_dim[1])
            self.enc_MLP.add_module(name, layer)
            if use_bn:
                name = "enc_lin" + str(n) +"_norm"
                self.enc_MLP.add_module(name,nn.BatchNorm1d(h_dim[1]))
            self.enc_MLP.add_module("enc_lin" + str(n) +"_act",h_nnLin)
            if dropout:
                name = "enc_lin" + str(n) +"_drop"
                self.enc_MLP.add_module(name, nn.Dropout(p=dropout))
        
        #Latent layers
        last_h_dim = h_dim[1]
        self.hz_mu = nn.Linear(last_h_dim, z_dim)
        self.hz_logvar = nn.Linear(last_h_dim, z_dim)
        #ACTIVATIONS VERS Z ?? out_nnLin
        
        #DECODER LAYERS
        #MLP
        self.dec_MLP = nn.Sequential()
        rev_h_dims = [list(reversed(i)) for i in list(reversed(h_dims))]
        self.dec_MLP.add_module('z_dec_lin0', nn.Linear(z_dim, last_h_dim))
        for n, h_dim in enumerate(rev_h_dims):
            name = "dec_lin"+str(n)
            layer = nn.Linear(h_dim[0], h_dim[1])
            self.dec_MLP.add_module(name, layer)
            if use_bn:
                name = "dec_lin" + str(n) +"_norm"
                self.dec_MLP.add_module(name,nn.BatchNorm1d(h_dim[1]))
            self.dec_MLP.add_module("dec_lin" + str(n) +"_act",h_nnLin)
            if dropout:
                name = "dec_lin" + str(n) +"_drop"
                self.dec_MLP.add_module(name, nn.Dropout(p=dropout))   
                
        #DECONV
        self.dec_conv = nn.Sequential()
        for n, deconv in enumerate(deconv_list[:-1]):
            name = "dec_conv"+str(n)
            layer = nn.ConvTranspose2d(deconv[0],deconv[1],deconv[2],stride=deconv[3],padding=deconv[4],output_padding = deconv[5]) #[in_channels, out_channels, kernel_size, stride, padding, output_padding]
            self.dec_conv.add_module(name, layer)
            if use_bn:
                name = "dec_conv" + str(n) +"_norm"
                self.dec_conv.add_module(name,nn.BatchNorm2d(deconv[1]))
            self.dec_conv.add_module("dec_conv" + str(n) +"_act",h_nnLin)
            if dropout:
                name = "dec_conv" + str(n) +"_drop"
                self.dec_conv.add_module(name, nn.Dropout2d(p=dropout))   
        
        deconv = deconv_list[-1]
        self.hrec_mu = nn.ConvTranspose2d(deconv[0],deconv[1],deconv[2],stride=deconv[3],padding=deconv[4],output_padding = deconv[5])
        self.hrec_logvar = nn.ConvTranspose2d(deconv[0],deconv[1],deconv[2],stride=deconv[3],padding=deconv[4],output_padding = deconv[5])
        
        #INIT XAVIER (weights) AND ZERO (biases)
        for name, param in self.named_parameters():
            if ('weight' in name) and (not 'norm' in name):
                nn.init.xavier_normal(param)
            elif ('bias' in name) and (not 'norm' in name):
                nn.init.uniform(param,0,0)
        
        
    def encode(self, x):
        out_conv = self.enc_conv(x)
        self.unflat_size = out_conv.size()
        flat_dim = out_conv.size()[-1] * out_conv.size()[-2] * out_conv.size()[-3]
        in_MLP = out_conv.view(-1, flat_dim)
        h = self.enc_MLP(in_MLP)
        z_mu, z_logvar = self.hz_mu(h), self.hz_logvar(h)
        return z_mu, z_logvar
    
    
    def reparametrize(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()), requires_grad = False)
        if torch.cuda.is_available():
            eps = eps.cuda()
        return mu+torch.exp(logvar/2)*eps
    
    
    def decode(self, z):
        out_MLP = self.dec_MLP(z)
        in_deconv = out_MLP.view(self.unflat_size)
        h = self.dec_conv(in_deconv)
        rec_mu, rec_logvar = self.hrec_mu(h), self.hrec_logvar(h)
        return rec_mu, rec_logvar
    
    
    def forward(self, mb):
        z_mu, z_logvar = self.encode(mb)
        z = self.reparametrize(z_mu, z_logvar)
        rec_mu, rec_logvar = self.decode(z)
        return rec_mu, rec_logvar, z_mu, z_logvar
    
    def save(self, name, use_cuda):
        copy = deepcopy(self)
        if use_cuda:
            copy.cpu()
        savepath = 'results/'+name
        torch.save(copy.state_dict(), savepath)
        
#    def load(self, name, directory):