#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 17:47:42 2018

@author: cyranaouameur
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class WavenetLayer(nn.Module):
    def __init__(self, 
                 dilation, 
                 kernel_size = 2, 
                 dilation_channels=32,
                 residual_channels=32, 
                 bias = False):
        
        super(WavenetLayer, self).__init__()
        
        
        self.dilation = dilation
        self.dilatedConv = nn.Conv1d(in_channels=residual_channels, 
                                     out_channels=dilation_channels, 
                                     kernel_size=kernel_size, 
                                     stride=1, 
                                     dilation=self.dilation, 
                                     bias=bias)
        
        self.resConv = nn.Conv1d(in_channels=dilation_channels,
                                 out_channels=residual_channels,
                                 kernel_size=1,
                                 bias=bias)
        
    def forward(self, unit_input):
        
        output = F.relu(unit_input)
        output = self.dilatedConv(output)
        output = F.relu(output)
        output = self.resConv()
        output += unit_input
        
        return output
    

class WavenetBlock(nn.Module):
    def __init__(self, 
                 nbLayers, 
                 kernel_size = 2, 
                 dilation_channels=32,
                 residual_channels=32, 
                 bias = False):
        
        super(WavenetBlock, self).__init__()
        
        self.layers = nn.Sequential()
        for i in range(nbLayers):
            dilation = 2**i
            self.layers.add_module(WavenetLayer(dilation, kernel_size, 
                                                dilation_channels, residual_channels))
            
    def forward(self, block_input):
        output = self.layers(block_input)
        return output

#####################################
###  DEFINITION OF THE VAE MODEL  ###
#####################################

        
class TemporalEncoder(nn.Module):
        
    def __init__(self,
                 nbLayers=10,
                 nbBlocks=3,
                 dilation_channels=32,
                 residual_channels=32,                 
                 nbClasses=256,
                 output_length=32,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False,
                 use_cuda = False,
                 *args, **kwargs):
        
        super(TemporalEncoder, self).__init__()
                
        self.useCuda = use_cuda
        self.nbLayers = nbLayers
        self.nbBlocks = nbBlocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.nbClasses = nbClasses
        self.kernel_size = kernel_size
        self.dtype = dtype

        
        #Input convolution to create channels
        self.inputConv = nn.Conv1d(in_channels=self.nbClasses,
                                   out_channels=self.residual_channels,
                                   kernel_size=1,
                                   bias=bias)
        
        #Wavenet blocks
        self.wavenet_blocks = nn.Sequential()
        for n in range(nbBlocks):
            name = "wavenet_block"+str(n)
            block = WavenetBlock(self, nbLayers, kernel_size = 2, dilation_channels=32,
                                 residual_channels=32, bias = False)
            self.wavenet_blocks.add_module(name, block)
            
                
        #Final convolution
        self.outputConv = nn.Conv1d(in_channels=residual_channels,
                                         out_channels=1,
                                         kernel_size=1,
                                         bias=True)
        
        #Average pooling
        #TODO : adjust stride to get the correct z size
        self.avgPooling = nn.AvgPool1d(kernel_size, stride=None, padding=0, 
                                        ceil_mode=False, count_include_pad=True)
        
        #Init params with XAVIER (weights) or ZERO (biases)
        for name, param in self.named_parameters():
            if ('weight' in name) and (not 'norm' in name):
                nn.init.xavier_normal(param)
            elif ('bias' in name) and (not 'norm' in name):
                nn.init.uniform(param,0,0)
        
        
    def forward(self, waveform_in):
        channeled_in = self.inputConv(waveform_in)
        block_output = self.wavenet_blocks(channeled_in)
        flattened = self.outputConv(block_output)
        latent = self.avgPooling(flattened)
        return latent
    
