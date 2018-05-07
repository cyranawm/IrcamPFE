#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 18:02:24 2018

@author: cyranaouameur
"""

import torch
from torch.autograd import Variable
from models.VAE.Conv_VAE import Conv_VAE, conv_loss, layers_config
from skimage.transform import resize
import numpy as np


#SET THE CONFIGURATION TO TEST
config = 1




#%% Define the parameters of the model (configs are in models/VAE/Conv_VAE.py) :
conv, h_dims, deconv = layers_config(config)
z_dim = 32
nnLin = ['relu','none'] #[h_act, out_act] with 'relu' or 'tanh' or 'elu' or 'none'
use_bn = True
dropout = False # False or a prob between 0 and 1


#initialize the model and use cuda if available
vae = Conv_VAE(conv, h_dims, z_dim, deconv, nnLin, use_bn, dropout)
#%%
data = np.ones((10,362,410))
nbFrames, nbBins = 362, 410
downFactor = 2
down_data = []
for img in data:
    down_data.append(resize(img, (int(nbFrames / downFactor), nbBins), mode='constant'))

ten = torch.from_numpy(np.array(down_data))
ten = ten/torch.max(torch.abs(ten))
ten = ten.unsqueeze(1).float()


ten = Variable(ten)

#%%

res = vae.forward(ten)[0]
print(ten.size())
print(res.size())
