#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 10:04:02 2018

@author: cyranaouameur
"""

import torch
import numpy as np


class MuLaw(object):
    """
    Encode signal based on mu-law companding.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_
    This algorithm assumes the signal has been scaled to between -1 and 1 and
    returns a signal encoded with values from 0 to quantization_channels - 1
    Args:
        quantization_channels (int): Number of channels. default: 256
    """

    def __init__(self, quantization_channels=256):
        self.qc = quantization_channels

    def __call__(self, x):
       return self.encode(x)
    
    def encode(self, x):
        """
        Args:
            x (FloatTensor/LongTensor or ndarray)
        Returns:
            x_mu (LongTensor or ndarray)
        """
        mu = self.qc - 1.
        if isinstance(x, np.ndarray):
            x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).astype(int)
        elif isinstance(x, (torch.Tensor, torch.LongTensor)):
            if isinstance(x, torch.LongTensor):
                x = x.float()
            mu = torch.FloatTensor([mu])
            x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
            x_mu = ((x_mu + 1) / 2 * mu + 0.5).long()
        return x_mu
    
    def categorical(self, x):
        if isinstance(x, np.ndarray):
            output = np.zeros((self.qc, len(x)))
            for i, val in enumerate(x):
                output[int(val), i] = 1
                
        elif isinstance(x, (torch.Tensor, torch.LongTensor)):
            if isinstance(x, torch.LongTensor):
                x = x.float()
            output = torch.zeros((self.qc, len(x)))
            for i, val in enumerate(x):
                output[int(val), i] = 1
        
        return output
    
    def to_int(self, x_mu):
        if isinstance(x, np.ndarray):
            output = np.zeros((1,self.qc))
            for i in range(x_mu.shape[1]):
                idx = np.argmax(x_mu[:,i])
                output[i] = idx
        
        elif isinstance(x, (torch.Tensor, torch.LongTensor)):
            if isinstance(x, torch.LongTensor):
                x = x.float()
            output = torch.zeros((1, self.qc))
            for i in range(x_mu.size()[1]):
                idx = torch.argmax(x_mu[:,i])
                output[i] = idx
            
        return output
            
        
    def decode(self, x_mu)
        """
        Args:
            x_mu (FloatTensor/LongTensor or ndarray)
        Returns:
            x (FloatTensor or ndarray)
        """
        mu = self.qc - 1.
        if isinstance(x_mu, np.ndarray):
            x = ((x_mu) / mu) * 2 - 1.
            x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
        elif isinstance(x_mu, (torch.Tensor, torch.LongTensor)):
            if isinstance(x_mu, torch.LongTensor):
                x_mu = x_mu.float()
            mu = torch.FloatTensor([mu])
            x = ((x_mu) / mu) * 2 - 1.
            x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
        return x
    
    
class MuLawExpanding(object):
    """
    Decode mu-law encoded signal.  For more info see the
    `Wikipedia Entry <https://en.wikipedia.org/wiki/%CE%9C-law_algorithm>`_
    This expects an input with values between 0 and quantization_channels - 1
    and returns a signal scaled between -1 and 1.
    Args:
        quantization_channels (int): Number of channels. default: 256
    """

    def __init__(self, quantization_channels=256):
        self.qc = quantization_channels

    def __call__(self, x_mu):
        """
        Args:
            x_mu (FloatTensor/LongTensor or ndarray)
        Returns:
            x (FloatTensor or ndarray)
        """
        mu = self.qc - 1.
        if isinstance(x_mu, np.ndarray):
            x = ((x_mu) / mu) * 2 - 1.
            x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
        elif isinstance(x_mu, (torch.Tensor, torch.LongTensor)):
            if isinstance(x_mu, torch.LongTensor):
                x_mu = x_mu.float()
            mu = torch.FloatTensor([mu])
            x = ((x_mu) / mu) * 2 - 1.
            x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.) / mu
        return x
