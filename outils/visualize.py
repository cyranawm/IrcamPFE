#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 13:13:39 2018

@author: cyranaouameur
"""
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from torch.autograd import Variable
import torch


if torch.cuda.is_available():   
    try:
        import matplotlib
        matplotlib.use('agg')
    except:
        import sys
        sys.path.append("/usr/local/lib/python3.6/site-packages/")
        import matplotlib
        matplotlib.use('agg')
else : 
    import matplotlib
    
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# visualize with PCA
def PlotPCA(VAE, dataloader, path, PCAdim = 2):
    
    z_array = []
    lbl_array = []
    
    print('Starting forward pass')
    
    for i, data in enumerate(dataloader):
        raw_input, label = data
        pre_process = torch.from_numpy(raw_input).float()
        if torch.cuda.is_available():
            pre_process = pre_process.cuda()
        pre_process = pre_process.unsqueeze(1)
        x = Variable(pre_process)
        
        #2. Forward data
        rec_mu, rec_logvar, z_mu, z_logvar = VAE.forward(x)
        z_array.append(z_mu.data.cpu().numpy())
        
        lbl_array.append(label[0])
    
    print('Ended forward pass')
    
    z_array = np.array(z_array)[:,0,:]
    print(lbl_array)
    sklearn_pca = sklearnPCA(PCAdim)
    print('Starting PCA')
    PCA_proj = sklearn_pca.fit_transform(z_array)
    
    colors = ["red", "green", "blue"]
    col_array = [colors[i] for i in lbl_array]
    
    labels = ['Kicks', 'Snares', 'Claps']
    lbl_array = [labels[i] for i in lbl_array]
    
    print('Starting plots')
    fig, ax = plt.subplots()
    for data, color, label in zip(PCA_proj, col_array, lbl_array):
        x, y = data[0], data[1]
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=label)
    
    handles = []
    for col, lbl in zip(colors, labels):
        patch = mpatches.Circle(xy = (0,0), radius = 1, color=col, label=lbl)
        handles.append(patch)

    
    ax.legend(handles = handles)
    ax.grid(True)
    fig.savefig(path)
    
    return PCA_proj