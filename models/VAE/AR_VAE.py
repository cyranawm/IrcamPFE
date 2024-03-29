#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 16:05:14 2018

@author: cyranaouameur
"""

import torch
import pyro.nn as pynn
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
from copy import deepcopy

class AR_VAE(nn.Module):
    def __init__(self, AR_dims, h_dims, z_dim, nnLin, use_bn, dropout, use_cuda = False, *args, **kwargs):
        super(AR_VAE, self).__init__()
        
        self.useCuda = use_cuda
        
        if not hasattr(self, 'constructor'):
            self.constructor = {'AR_dims':AR_dims, 'h_dims':h_dims, 'z_dim':z_dim, 'nnLin':nnLin,
                                'use_bn':use_bn, 'dropout': dropout, 'args':args, 'kwargs':kwargs} # remind construction arguments for easy load/save
        
        activation = {
                'relu' : nn.ReLU(),
                'tanh' : nn.Tanh(),
                'elu' : nn.ELU(),
                'none' : nn.Sequential()
                }
        
        h_nnLin = activation[nnLin[0]]
        out_nnLin = activation[nnLin[1]]
        
        
        #ENCODER LAYERS
        #AR
        self.enc_AR = nn.Sequential()
        ARmodule = pynn.AutoRegressiveNN(AR_dims[0], AR_dims[1])
        self.enc_AR.add_module('enc_AR', ARmodule)
        if use_bn:
            name = "enc_AR_norm"
            self.enc_AR.add_module(name,nn.BatchNorm2d(deconv[1]))
        self.enc_AR.add_module("enc_AR_act",h_nnLin)
        if dropout:
            name = "enc_AR_drop"
            self.enc_AR.add_module(name, nn.Dropout2d(p=dropout))   
    
        
                
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
        out_AR = self.enc_AR(x)
        self.unflat_size = out_AR.size()
        flat_dim = out_AR.size()[-1] * out_AR.size()[-2] * out_AR.size()[-3]
        in_MLP = out_AR.view(-1, flat_dim)
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
        in_AR = out_MLP.view(self.unflat_size)
        h = self.dec_AR(in_AR)
        rec_mu, rec_logvar = self.hrec_mu(h), self.hrec_logvar(h)
        return rec_mu, rec_logvar
    
    
    def forward(self, mb):
        z_mu, z_logvar = self.encode(mb)
        z = self.reparametrize(z_mu, z_logvar)
        rec_mu, rec_logvar = self.decode(z)
        return rec_mu, rec_logvar, z_mu, z_logvar
    
#    def save(self, name, use_cuda):
#        copy = deepcopy(self.state_dict())
#        if use_cuda:
#            for i, k in copy.items():
#                copy[i] = k.cpu()
#        savepath = 'results/'+name
#        torch.save(copy, savepath)
        
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
        
        for i, data in enumerate(validset):
            #1. get the inputs and wrap them in Variable
            inputs, labels = data
            inputs, labels = torch.from_numpy(inputs).float(), torch.from_numpy(labels)
            if use_cuda:
                inputs = inputs.cuda()
            inputs = inputs.unsqueeze(1)
            x, labels = Variable(inputs), Variable(labels)
            
            #2. Forward data
            rec_mu, rec_logvar, z_mu, z_logvar = self.forward(x)
    
            #3. Compute losses (+validation loss)
            recon_loss, kl_loss = conv_loss(x, rec_mu, rec_logvar, z_mu, z_logvar)
            loss = recon_loss + beta*kl_loss
            
            valid_loss += loss.data[0]
            
        valid_loss /= i+1
        
        if last_batch:
            last_batch_in = inputs
            last_batch_out = rec_mu
            return valid_loss, last_batch_in, last_batch_out
        
        else :
            return valid_loss