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







def visu_recon(nb, test_set, vae):
    for i, data in enumerate(test_set):
        if i > nb:
            break
        raw_inputs, labels = data
        inputs = raw_inputs.view((1,1,28*28))
        
        #inputs, labels = Variable(torch.FloatTensor(1*(inputs.numpy()>0.5))), Variable(labels)
        inputs, labels = Variable(inputs), Variable(labels)
        
        
        z = vae.encode(inputs)[0]
        print(z.data)
        recon = vae.decode(z)[0]
        
        raw_inputs = inputs.data.view(28,28)
        recon = recon.view(28,28)
        
        #print(recon)
        plt.subplot(211)
        plt.imshow(raw_inputs)
        plt.subplot(212)
        plt.imshow(recon.data.numpy())
    