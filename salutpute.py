#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:46:56 2018

@author: cyranaouameur
"""
#%%
import torch
from torch.autograd import Variable
import torch.nn as nn

test = torch.randn(1,1,313,410)
test = Variable(test)


conv1 = nn.Conv2d(1, 64, (20,10), (10,5), (2,2))
conv2 = nn.Conv2d(64, 64, (10,5), (4,4), (0,2))

deconv1 = nn.ConvTranspose2d(16, 8, (10,5), (4,4), (0,2), (1,1))
deconv2 = nn.ConvTranspose2d(8, 1, (13,9), (10,5), (0,2))

test1 = conv1(test)
test2 = conv2(test1)

test3 = deconv1(test2)
test4 = deconv2(test3)


print(test2.size(), test3.size(), test4.size())

















#%%
import sys
sys.path.append('aciditools/')
from drumLearning import importDataset
