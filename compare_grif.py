#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 17:55:53 2018

@author: cyranaouameur
"""

#%%Parse arguments
import argparse

parser = argparse.ArgumentParser(description='model to evaluate')


# VAE dimensions
parser.add_argument('model', type=str,
                    help='<Required> Choice of the model to evaluate')

parser.add_argument('resfold', type=str,
                    help='<Required> Folder where the results will be saved (ends with /)')

parser.add_argument('--gpu', type=int, default=1, metavar='N',
                    help='The ID of the GPU to use')

parser.add_argument('--pca', action='store_true',
                    help='compute PCA')

parser.add_argument('--soundrec', type=int, metavar = 'nb_reconstructions', 
                    help='Compute reconstructions ')

parser.add_argument('--soundlines', type=int, metavar = 'nb_lines', 
                    help='Compute sound lines ')
args = parser.parse_args()





#%%imports
print('BEGIN IMPORTS')

import numpy as np
import sys
from skimage.transform import resize

import os

import torch
from torch.autograd import Variable

from outils.scaling import scale_array, unscale_array
from outils.visualize import PlotPCA2D, PlotPCA3D, npy2scatter
from models.VAE.Conv_VAE import Conv_VAE, conv_loss, layers_config
from outils.sound import regenerate, create_line, get_nn, get_phase
from outils.nsgt_inversion import regenerateAudio


sys.path.append('./aciditools/')
try:
    from aciditools.utils.dataloader import DataLoader
    from aciditools.drumLearning import importDataset #Should work now
except:
    sys.path.append('/Users/cyranaouameur/anaconda2/envs/py35/lib/python3.5/site-packages/nsgt')
    from aciditools.utils.dataloader import DataLoader
    from aciditools.drumLearning import importDataset  


#%% Compute transforms and load data

if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu)
    print('USING CUDA ON GPU'+str(torch.cuda.current_device()))
    path = '/fast-1/DrumsDataset'
else:
    path = '/Users/cyranaouameur/Desktop/StageIrcam/Code/CodeCyran/datasets/DummyDrumsCropped'

print('IMPORT DATA')
dataset = importDataset(base_path = path, targetDur = 1.15583)

dataset.metadata['instrument'] = np.array(dataset.metadata['instrument']) #to array
#phases = np.angle(dataset.data)
dataset.data = np.abs(dataset.data) # to real positive array
#%%

#downsample by a given factor
nbFrames, nbBins = dataset.get(0).shape
downFactor = 2
downsampled = []
for img in dataset.data:
    downsampled.append(resize(img, (int(nbFrames / downFactor), nbBins), mode='constant'))
in_shape = (int(nbFrames / downFactor), nbBins)

#Scale data
log_scaling = True
normalize = 'gaussian'
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

results_folder = args.resfold
subfolders = ['images/PCA', 'sounds', 'sounds/line']

for folder in subfolders:
    if not os.path.isdir(results_folder + folder):
        os.makedirs(results_folder + folder)
    

#%%

if args.soundrec:
    
    nb_rec = args.soundrec
    soundPath = results_folder + 'sounds/'
    for it in [100,200,300]:
        for ph in [False, True]:
            ext = '_'+str(it)+str(ph)
            print(ph)
            regenerate(vae, dataset, nb_rec, it, norm_const, normalize, log_scaling, downFactor, soundPath, initPhase = ph, nameExtension = ext)

if args.soundlines:
    
    nb_lines = args.soundlines
    #get latent coords for each entry ?
    
    latentCoords = [] 
    nb_samples = 10   
    soundPath = results_folder + 'sounds/line/'
    targetLen = int(1.15583*22050)
    
    for i, raw_input in enumerate(dataset.data):
        
        pre_process = torch.from_numpy(raw_input).float()
        if torch.cuda.is_available():
            pre_process = pre_process.cuda()
        pre_process = pre_process.unsqueeze(0)
        pre_process = pre_process.unsqueeze(0)#add 2 dimensions to forward into vae
        x = Variable(pre_process)
        
        rec_mu, rec_logvar, z_mu, z_logvar = vae.forward(x)
        latentCoords.append(z_mu.data.cpu().numpy())
        
    for n in range(nb_lines):
        #take 2 coord set and draw a line
        i, j = np.random.randint(len(latentCoords)), np.random.randint(len(latentCoords))
        line_coords = create_line(latentCoords[i], latentCoords[j], nb_samples)
        
        #decode for each
        line = torch.from_numpy(line_coords).float()
        if torch.cuda.is_available():
            line = line.cuda()
        line = Variable(line)
        x_rec = vae.decode(line)[0]
        #regenerate
        for i, nsgt in enumerate(x_rec.data.cpu().numpy()):
            nnIndex = get_nn(latentCoords, line_coords[i])
            nn = dataset.files[nnIndex]
            nnPhase = get_phase(nn, targetLen)
            
            #suppress dumb sizes and transpose to regenerate
            nsgt = nsgt[0].T
            
            #compute the resize needed
            nbFreq, nbFrames = regenerateAudio(np.zeros((1, 1)), testSize = True, targetLen = targetLen)
    
            # RE-UPSAMPLE the distribution
            factor = np.max(np.abs(nsgt))        
            nsgt = resize(nsgt/factor, (nbFreq, nbFrames), mode='constant')
            nsgt *= factor
            
            #rescale
            nsgt = unscale_array(nsgt, norm_const, normalize, log_scaling)
            
            for it in [100,200,300]:
                for ph in [False, True]:
                    if ph == True:
                        phase = nnPhase
                    else:
                        phase = False
                    
                    regenerateAudio(nsgt, sr=22050, targetLen = int(1.15583*22050), iterations=it, initPhase = phase, curName=soundPath + str(n) + '_' + str(i) + '_' + str(it) + '_ph' + str(ph))
    
#%%
#
#data = np.load('/Users/cyranaouameur/Desktop/StageIrcam/Code/CodeCyran/results/images/PCA/pca3D_conv_config1_final_data.npy')
#col = np.load('/Users/cyranaouameur/Desktop/StageIrcam/Code/CodeCyran/results/images/PCA/pca3D_conv_config1_final_colors.npy')
#
#npy2scatter(data, col)
#
#
