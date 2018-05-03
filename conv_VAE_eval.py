#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:06:43 2018

@author: cyranaouameur
"""

import numpy as np
import sys
from skimage.transform import resize


import torch

from outils.scaling import scale_array
from outils.visualize import PlotPCA
from models.VAE.Conv_VAE import Conv_VAE, conv_loss, layers_config

sys.path.append('./aciditools/')
try:
    from aciditools.utils.dataloader import DataLoader
    from aciditools.drumLearning import importDataset #Should work now
except:
    sys.path.append('/Users/cyranaouameur/anaconda2/envs/py35/lib/python3.5/site-packages/nsgt')
    from aciditools.utils.dataloader import DataLoader
    from aciditools.drumLearning import importDataset  
    
#if torch.cuda.is_available():   
#    try:
#        import matplotlib
#        matplotlib.use('agg')
#    except:
#        import sys
#        sys.path.append("/usr/local/lib/python3.6/site-packages/")
#        import matplotlib
#        matplotlib.use('agg')
#else : 
#    import matplotlib
#    
#import matplotlib.pyplot as plt

import argparse
#%%Parse arguments

parser = argparse.ArgumentParser(description='model to evaluate')


# VAE dimensions
parser.add_argument('model', type=str,
                    help='<Required> Choice of the model to evaluate')

args = parser.parse_args()
    
#%% Compute transforms and load data
log_scaling = True
normalize = 'gaussian'

if torch.cuda.is_available():
    path = '/fast-1/DrumsDataset'
else:
    path = '/Users/cyranaouameur/Desktop/StageIrcam/Code/CodeCyran/datasets/DummyDrumsCropped'

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

model = args.model
dico = torch.load(model)
vae = dico["class"].load(dico)
if torch.cuda.is_available():
    vae.cuda()
vae.eval()


#%%


PlotPCA(vae, evalloader, './results/images/PCA' + args.model.split('/')[-1] +'.png')
















