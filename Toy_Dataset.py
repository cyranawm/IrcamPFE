#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:18:11 2018

@author: cyranaouameur
"""

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
import torch

class Toy_Dataset(Dataset):
    
    def __init__(self,path):
        npz = np.load(path)
        names = npz.files
        data = npz[names[0]]
        lbl = npz[names[1]]
        
        assert len(data) == len(lbl)
        

        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(lbl)
        
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.size(0)
    
    def load(self, mb_size, shuffle):
        data_loader = torch.utils.data.DataLoader(self,
                                          batch_size=mb_size,
                                          shuffle=True)
        return data_loader

#%% KIFF
data_folder = 'datasets/Toy1/'
dataset = 'toy-spectral-richness-v2-db-pos.npz'

path = data_folder+dataset

toy = Toy_Dataset(path)
toy_loader = toy.load(100, shuffle = False)

for i, data in enumerate(toy_loader):
    if i%21 == 0:
        plt.figure()
        raw_inputs, labels = data
        plt.stem(raw_inputs[1].numpy())


