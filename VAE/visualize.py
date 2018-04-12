#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:20:47 2018

@author: cyranaouameur
"""

import matplotlib.pyplot as plt
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch







def plotInOut(test_set, vae):
    for i, data in enumerate(test_set):
        if i > 3:
            break
        
        idx = i+1
        
        raw_inputs, labels = data
        inputs = raw_inputs.view((1,1,28*28))
        
        #inputs, labels = Variable(torch.FloatTensor(1*(inputs.numpy()>0.5))), Variable(labels)
        inputs, labels = Variable(inputs), Variable(labels)
        
        
        z = vae.encode(inputs)[0]
        recon = vae.decode(z)[0]
        
        raw_inputs = inputs.data.view(28,28)
        recon = recon.view(28,28)
        
        #print(recon)
        plt.subplot(2,4,idx)
        plt.imshow(raw_inputs)
        plt.subplot(2,4,4+idx)
        plt.imshow(recon.data.numpy())
        
def plotInOut_Conv(test_set, vae):
    for i, data in enumerate(test_set):
        if i > 3:
            break
        
        idx = i+1
        
        inputs, labels = data
        
        
        #inputs, labels = Variable(torch.FloatTensor(1*(inputs.numpy()>0.5))), Variable(labels)
        inputs, labels = Variable(inputs), Variable(labels)
        
        
        z = vae.encode(inputs)[0]
        recon = vae.decode(z)[0]
        
        raw_inputs = inputs.data.view(28,28)
        recon = recon.view(28,28)
        
        #print(recon)
        plt.subplot(2,4,idx)
        plt.imshow(raw_inputs)
        plt.subplot(2,4,4+idx)
        plt.imshow(recon.data.numpy())
        
    