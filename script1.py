#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:47:18 2018

@author: cyranaouameur
"""

use_cuda = True


from VAE.Vanilla_VAE_NN import Vanilla_VAE
#from Bernoulli_VAE import Bernoulli_VAE
#from VAE.Bernoulli_VAE_NN import Bernoulli_VAE
from tensorboardX import SummaryWriter

from scipy.io import loadmat

from visualize import visu_recon


import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

def load_MNIST(mb_size):
    transform = transforms.ToTensor()
    MNIST = torchvision.datasets.MNIST("./datasets/MNIST/", train=True, transform=transform, target_transform=None, download=True)
    
    data_loader = torch.utils.data.DataLoader(MNIST,
                                          batch_size=mb_size,
                                          shuffle=True)
    return data_loader


def test_MNIST():
    transform = transforms.ToTensor()
    MNIST = torchvision.datasets.MNIST("./datasets/MNIST/", train=False, transform=transform, target_transform=None, download=True)
    
    test_loader = torch.utils.data.DataLoader(MNIST,
                                          batch_size=1,
                                          shuffle=True)
    return test_loader

#
#def load_ff(mb_size):
#    img_rows, img_cols = 28, 20
#    path = '/Users/cyranaouameur/Desktop/Stage Ircam/Code/Code Cyran/datasets/frey_rawface.mat'
#    ff = loadmat(path, squeeze_me=True, struct_as_record=False)
#    ff = ff["ff"].T.reshape((-1, img_rows, img_cols))
#    
#    n_pixels = img_rows * img_cols
#    X_train = ff[:1800]
#    X_val = ff[1800:1900]
#    
#    X_train = X_train.astype('float32') / 255.
#    X_val = X_val.astype('float32') / 255.
#    X_train = X_train.reshape((len(X_train), n_pixels))
#    X_val = X_val.reshape((len(X_val), n_pixels))
#    
#    return X_train, X_val

x_dim = 28*28
h_dim = 500
z_dim = 10

vae1 = Vanilla_VAE(x_dim,h_dim,z_dim, use_cuda)
if use_cuda:
    vae1.cuda()


trainloader = load_MNIST(vae1.mb_size)
testloader = test_MNIST()

vae1.train(trainloader, 2)

visu_recon(1,testloader,vae1)

#for param in list(vae1.named_parameters()):
#    if (param[1].data.numpy().any() == np.nan or  abs(param[1].data.numpy()).any() == np.inf):
#        print(param[0])