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

#####################################
###   DEFINITION OF THE ENCODER   ###
#####################################


class EncoderLayer(nn.Module):
    def __init__(self, 
                 dilation, 
                 kernel_size = 2, 
                 dilation_channels=32,
                 residual_channels=32, 
                 bias = False):
        
        super(EncoderLayer, self).__init__()
        
        
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
        if self.dilation == 1:
            pad = (0,1)
        else:
            pad = (self.dilation//2, self.dilation//2)
        padded_input = F.pad(unit_input, pad)
        output = F.relu(padded_input)
        output = self.dilatedConv(output)
        output = F.relu(output)
        output = self.resConv(output)
        output += unit_input
        
        return output
    

class EncoderBlock(nn.Module):
    def __init__(self, 
                 nbLayers, 
                 kernel_size = 2, 
                 dilation_channels=32,
                 residual_channels=32, 
                 bias = False):
        
        super(EncoderBlock, self).__init__()
        
        self.layers = nn.Sequential()
        for i in range(nbLayers):
            dilation = 2**i
            name = 'dilation' + str(dilation)
            self.layers.add_module(name, EncoderLayer(dilation, kernel_size, 
                                                dilation_channels, residual_channels))
            
    def forward(self, block_input):
        output = self.layers(block_input)
        return output



        
class TemporalEncoder(nn.Module):
        
    def __init__(self,
                 nbLayers=10,
                 nbBlocks=3,
                 dilation_channels=32,
                 residual_channels=32,                 
                 nbClasses=256,
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


        
        #Wavenet blocks
        self.encoder_blocks = nn.Sequential()
        for n in range(nbBlocks):
            name = "encoder_block"+str(n)
            block = EncoderBlock(nbLayers, kernel_size, dilation_channels,
                                 residual_channels, bias)
            self.encoder_blocks.add_module(name, block)
            
        
        #Init params with XAVIER (weights) or ZERO (biases)
        for name, param in self.named_parameters():
            if ('weight' in name) and (not 'norm' in name):
                nn.init.xavier_normal_(param)
            elif ('bias' in name) and (not 'norm' in name):
                nn.init.constant_(param,0)
        
        
    def forward(self, channeled_in):
        block_output = self.encoder_blocks(channeled_in)
        return block_output
    
#####################################
###   DEFINITION OF THE DECODER   ###
#####################################



class DecoderLayer(nn.Module):
    def __init__(self, 
                 dilation, 
                 kernel_size = 2, 
                 dilation_channels=32,
                 residual_channels=32, 
                 bias = False):
        
        super(DecoderLayer, self).__init__()
        
        
        self.dilation = dilation
        self.dilatedDeconv = nn.ConvTranspose1d(in_channels=residual_channels, 
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
        output = self.dilatedDeconv(output)
        output = output[:,:,:-self.dilation]
        output = F.relu(output)
        output = self.resConv(output)
        output += unit_input
        
        return output
    

class DecoderBlock(nn.Module):
    def __init__(self, 
                 nbLayers, 
                 kernel_size = 2, 
                 dilation_channels=32,
                 residual_channels=32, 
                 bias = False):
        
        super(DecoderBlock, self).__init__()
        
        self.layers = nn.Sequential()
        for i in range(nbLayers):
            dilation = 2**(nbLayers-i-1)
            name = 'dilation' + str(dilation)
            self.layers.add_module(name, DecoderLayer(dilation, kernel_size, 
                                                dilation_channels, residual_channels))
            
    def forward(self, block_input):
        output = self.layers(block_input)
        return output


        
class TemporalDecoder(nn.Module):
        
    def __init__(self,
                 nbLayers=10,
                 nbBlocks=3,
                 dilation_channels=32,
                 residual_channels=32,                 
                 nbClasses=256,
                 kernel_size=2,
                 dtype=torch.FloatTensor,
                 bias=False,
                 use_cuda = False,
                 *args, **kwargs):
        
        super(TemporalDecoder, self).__init__()
                
        self.useCuda = use_cuda
        self.nbLayers = nbLayers
        self.nbBlocks = nbBlocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.nbClasses = nbClasses
        self.kernel_size = kernel_size
        self.dtype = dtype

        
        #decoder blocks
        self.decoder_blocks = nn.Sequential()
        for n in range(nbBlocks):
            name = "deocder_block"+str(n)
            block = DecoderBlock(nbLayers, kernel_size, dilation_channels,
                                 residual_channels, bias)
            self.decoder_blocks.add_module(name, block)

        
        #Init params with XAVIER (weights) or ZERO (biases)
        for name, param in self.named_parameters():
            if ('weight' in name) and (not 'norm' in name):
                nn.init.xavier_normal_(param)
            elif ('bias' in name) and (not 'norm' in name):
                nn.init.constant_(param,0)
        
        
    def forward(self, channeled_in):
        block_output = self.decoder_blocks(channeled_in)
        return block_output
    
