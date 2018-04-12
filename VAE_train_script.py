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
from VAE.logVar_VAE import Vanilla_VAE
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

vae1 = Vanilla_VAE(x_dim,h_dim,z_dim, mb_size, use_cuda, use_tensorboard)

if use_cuda :
    torch.cuda.set_device(1)
    print("**************************** USING CUDA ****************************")
    vae1.cuda()



trainloader = load_MNIST(vae1.mb_size)


vae1.train(trainloader, 1000)

if use_cuda:
    vae1.cpu()


name = 'test1'
savepath = 'results/'+name
torch.save(vae1.state_dict(), savepath)


