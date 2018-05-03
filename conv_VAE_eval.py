#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:06:43 2018

@author: cyranaouameur
"""

#%%Parse arguments
import argparse

parser = argparse.ArgumentParser(description='model to evaluate')


# VAE dimensions
parser.add_argument('model', type=str,
                    help='<Required> Choice of the model to evaluate')

parser.add_argument('--gpu', type=int, default=1, metavar='N',
                    help='The ID of the GPU to use')

parser.add_argument('--pca', action='store_true',
                    help='compute PCA')

parser.add_argument('--sound', action='store_true',
                    help='compute sounds')

args = parser.parse_args()





#%%imports
print('BEGIN IMPORTS')

import numpy as np
import sys
from skimage.transform import resize


import torch

from outils.scaling import scale_array
from outils.visualize import PlotPCA2D, PlotPCA3D, npy2scatter
from models.VAE.Conv_VAE import Conv_VAE, conv_loss, layers_config
from outils.sound import regenerate

sys.path.append('./aciditools/')
try:
    from aciditools.utils.dataloader import DataLoader
    from aciditools.drumLearning import importDataset #Should work now
except:
    sys.path.append('/Users/cyranaouameur/anaconda2/envs/py35/lib/python3.5/site-packages/nsgt')
    from aciditools.utils.dataloader import DataLoader
    from aciditools.drumLearning import importDataset  


#%% Compute transforms and load data
log_scaling = True
normalize = 'gaussian'

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    print('USING CUDA ON GPU'+str(torch.cuda.current_device()))
    path = '/fast-1/DrumsDataset'
else:
    path = '/Users/cyranaouameur/Desktop/StageIrcam/Code/CodeCyran/datasets/DummyDrumsCropped'

print('IMPORT DATA')
dataset = importDataset(base_path = path, targetDur = 0.74304)

dataset.metadata['instrument'] = np.array(dataset.metadata['instrument']) #to array
dataset.data = np.abs(dataset.data) # to real positive array


#downsample by a given factor
nbFrames, nbBins = dataset.get(0).shape
downFactor = 2
downsampled = []
for img in dataset.data:
    downsampled.append(resize(img, (int(nbFrames / downFactor), nbBins), mode='constant'))
in_shape = (int(nbFrames / downFactor), nbBins)

#Scale data
dataset.data, norm_const = scale_array(downsampled, log_scaling, normalize) 

#Create the Loaders
evalloader = DataLoader(dataset, 1, task = 'instrument') 

#%%Load the model

print('LOAD MODEL')

model = args.model
dico = torch.load(model)
vae = dico["class"].load(dico)
if torch.cuda.is_available():
    vae.cuda()
vae.eval()


#%%

if args.pca: 
    PlotPCA2D(vae, evalloader, './results/images/PCA/PCA2d_' + args.model.split('/')[-1] +'.png')
    pca3d, colors = PlotPCA3D(vae, evalloader, './results/images/PCA/PCA3d_' + args.model.split('/')[-1] +'.png')
    
    np.save('./results/images/PCA/pca3D_'+ args.model.split('/')[-1] + '_data', pca3d)
    np.save('./results/images/PCA/pca3D_' + args.model.split('/')[-1] + 'colors', colors)


#%%

if args.sound:
    soundPath = './results/sounds/'
    regenerate(vae, evalloader, norm_const, normalize, log_scaling, downFactor, soundPath)

#%%
#
#data = np.load('/Users/cyranaouameur/Desktop/StageIrcam/Code/CodeCyran/results/images/PCA/pca3D_conv_config1_final_data.npy')
#col = np.load('/Users/cyranaouameur/Desktop/StageIrcam/Code/CodeCyran/results/images/PCA/pca3D_conv_config1_final_colors.npy')
#
#npy2scatter(data, col)
#
#









