#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 14:20:47 2018

@author: cyranaouameur
"""


try:
    import matplotlib
    from matplotlib import pyplot as plt
except:
    import sys
    sys.path.append("/usr/local/lib/python3.6/site-packages/")
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt



from torch.autograd import Variable
import numpy as np



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
        
        
        
        
        
def saveInOut(test_set, vae, name):
    
    fig = plt.figure()
    
    for i, data in enumerate(test_set):
        if i > 10:
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
    
    fig.savefig('./results/images/' + name)
        
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
        
    