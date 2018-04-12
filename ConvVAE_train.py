#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:47:18 2018

@author: cyranaouameur
"""

#use_cuda = False
#use_tensorboard = False

use_tensorboard = True

#from VAE.Vanilla_VAE_NN import Vanilla_VAE
from VAE.Conv_VAE import Conv_VAE
#from Bernoulli_VAE import Bernoulli_VAE
#from VAE.Bernoulli_VAE_NN import Bernoulli_VAE


#from visualize import visu_recon


import numpy as np

import torch
use_cuda = torch.cuda.is_available()


import torchvision
import torchvision.transforms as transforms

from datasets.MNIST import load_MNIST

  
x_dim = 28*28
h_dim = 500
z_dim = 10

mb_size = 100

convvae = Conv_VAE(h_dim,z_dim, mb_size, use_cuda, use_tensorboard)

if use_cuda :
    torch.cuda.set_device(1)
    print("**************************** USING CUDA ****************************")
    convvae.cuda()



trainloader = load_MNIST(convvae.mb_size)


convvae.train(trainloader, 100)

if use_cuda:
    convvae.cpu()

name = 'conv1'
savepath = 'results/'+name
torch.save(convvae.state_dict(), savepath)


