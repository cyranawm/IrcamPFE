#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 12:20:32 2018

@author: cyranaouameur
"""

import argparse
#%%Parse arguments

parser = argparse.ArgumentParser(description='Conv_VAE training and saving')


# VAE dimensions


parser.add_argument('z_dim', type=int,
                    help='<Required> Dimension of the latent space')
parser.add_argument('--layers', type=int, default=7, metavar='NbLayers',
                    help='number of layers per block (default: 7)')
parser.add_argument('--blocks', type=int, default=2, metavar='NbBlocks',
                    help='number of encoding/decoding lbocks (default: 2)')


#data settings
#parser.add_argument('--task', type=str, default='full', metavar='class', choices=['kicks', 'full'],
#                    help='Define the class of instruments to train the model on (kicks or full)')
#parser.add_argument('--downsample', type=int, default=1, metavar='N',
#                    help='Define the downsampling factor (default: 1 -> no downsample)')

# training settings
parser.add_argument('--mb_size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='number of epochs to train (default: 5000)')
parser.add_argument('--beta', type=int, default=1, metavar='N',
                    help='beta coefficient for regularization (default: 1)')
parser.add_argument('--Nwu', type=int, default=100, metavar='N',
                    help='epochs number for warm-up (default: 100)')

parser.add_argument('--gpu', type=int, default= -1, metavar='N',
                    help='The ID of the GPU to use')

parser.add_argument('--checkpoints', action='store_true',
                    help='save checkpoints each 200 epochs')


args = parser.parse_args()

#%%Imports
try:
    import matplotlib
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
except:
    import sys
    sys.path.append("/usr/local/lib/python3.6/site-packages/")
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

import numpy as np
import sys

import os

import gc

import torch
import torch.optim as optim
from torch.autograd import Variable

from outils.scaling import scale_array
from models.Wavenet.TemporalModel import TemporalModel
from outils.mulaw import MuLaw 


sys.path.append('./aciditools/')
try:
    from aciditools.utils.dataloader import DataLoader
    from aciditools.drumLearning import importDataset #Should work now
except:
    sys.path.append('/Users/cyranaouameur/anaconda2/envs/py35/lib/python3.5/site-packages/nsgt')
    from aciditools.utils.dataloader import DataLoader
    from aciditools.drumLearning import importDataset 


#%% CUDA

use_cuda = torch.cuda.is_available()
if use_cuda and args.gpu >= 0:
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    print('USING CUDA ON GPU' + str(torch.cuda.current_device()))
    
#%% Compute transforms and load data
#log_scaling = True

#task = args.task

#import raw data
    
print('LOADING DATA')

dataset = importDataset(transform = 'raw')

dataset.metadata['instrument'] = np.array(dataset.metadata['instrument']) #to array
dataset.data = np.array(dataset.data) # to array

#scale from -1 to 1
for s in dataset.data:
    s /= np.max(np.abs(s))

#compute mu-law
mulaw = MuLaw(256)
final_data = []

print('MULAW ENCODING')

for i in range(len(dataset.data)):
    final_data.append(mulaw(dataset.data[i]))
    
dataset.data = np.array(final_data)

#Constrcut partitions (train and validation sets)
print('CREATING LOADERS')
dataset.constructPartition('instrument', ['train','valid'], [0.8, 0.2])

#Compute the best mb_size for valid_set
mb_size = args.mb_size
len_val = len(dataset.partitions['valid'])
valid_mb = [x for x in range(len_val+1) if x != 0 and len_val%x == 0 and x<mb_size][-1]
if valid_mb == 1:
    valid_mb = mb_size


#mb_size = 3

#Create the Loaders
trainloader = DataLoader(dataset, mb_size, 'instrument', partition = 'train') 
testloader = DataLoader(dataset, valid_mb, 'instrument', partition = 'valid')


#%% Define the parameters of the model (configs are in models/VAE/Conv_VAE.py) :

z_dim = args.z_dim
nbLayers = args.layers
nbBlocks = args.blocks
dilation_channels = 64
residual_channels = 64
nbClasses = 256
kernel_size = 2

duration = 0.1*22050

in_shape = (nbClasses, duration)

#Hyper parameters : non-linearity? batchnorm? dropout?

#use_bn = True
#dropout = False # False or a prob between 0 and 1

final_beta = args.beta
wu_time = args.Nwu
use_tensorboard = True #True, False or 'Full' (histograms)




#initialize the model and use cuda if available
vae = TemporalModel()
if use_cuda :
    vae.cuda()
if use_tensorboard:
    from tensorboardX import SummaryWriter
    vae.writer = SummaryWriter()
    
    
#%%Training routine
    
model_name = 'TemporalModel_' +str(nbBlocks) +'_' + str(nbLayers) 
results_folder = './results/'+model_name
if not os.path.isdir(results_folder):
    os.makedirs(results_folder)
    os.makedirs(results_folder + '/images/reconstructions')
    os.makedirs(results_folder + '/checkpoints')

nb_epochs = args.epochs
#%%
vae.train()
optimizer = optim.Adam(vae.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=100, min_lr = 5e-06)

best_valid = 1e8

print('***Training Started***')

for epoch in range(nb_epochs):

#BETA WU
    beta = final_beta * min(1,epoch/wu_time)
########

    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_KL = 0.0
    
    #shuffle dataloader
    trainloader.shuffle()
    
    for i, data in enumerate(trainloader) : 
        optimizer.zero_grad()
        
        #1. get the inputs and wrap them in Variable
        raw_inputs, labels = data
        raw_inputs = np.concatenate([i for i in raw_inputs])
        raw_inputs = np.random.permutation(raw_inputs)[:10*mb_size]
        pre_process, labels = torch.from_numpy(raw_inputs).float(), torch.from_numpy(labels)
        if use_cuda:
            pre_process = pre_process.cuda()
        x = pre_process.unsqueeze(1)
        x, labels = Variable(x), Variable(labels)
        
        
        #2. Forward data
        x_rec, z_mu, z_logvar = vae.forward(x)

        #3. Compute losses (+validation loss)
        #print(x.size(), x_rec.size(), z_mu.size(), z_logvar.size())
        recon_loss, kl_loss = vae.loss(x, x_rec, z_mu, z_logvar)
        loss = recon_loss + beta*kl_loss
        
        epoch_loss += loss.data[0]
        epoch_recon += recon_loss.data[0]
        epoch_KL += kl_loss.data[0]
        
        #4. Backpropagate
        loss.backward()
        optimizer.step()

#4. EPOCH FINISHED :
    
    epoch_size = i+1
    
    
    # Saving sounds/waveforms
    if np.mod(epoch,50) == 0: 
            #from training set
            fig = plt.figure(figsize = (12,8))
            for idx in range(1,5):
                plt.subplot(4,2,2*idx-1)
                inputs = mulaw.decode(raw_inputs[idx])
                plt.plot(inputs)
                plt.subplot(4,2,2*idx)
                output = mulaw.to_int(x_rec[idx])
                output = mulaw.decode(output)
                plt.plot(output.clone().cpu().numpy()) #still a variable
            fig.savefig(results_folder + '/images/reconstructions/train_epoch'+str(epoch)+'.png', bbox_inches = 'tight')
            
    raw_inputs, pre_process, x, x_rec = None,None,None,None
    gc.collect()


#Compute validation loss and scheduler.step()
  
    valid_loss, valid_in, valid_out = vae.valid_loss(testloader, beta, use_cuda, last_batch = True)  
    scheduler.step(valid_loss)

#Tensorboard log
    if use_tensorboard:
        vae.writer.add_scalars('data/AvgLosses', {'Loss': epoch_loss/epoch_size,
                                             'Validation': valid_loss,
                                           'Reconstruction': epoch_recon/epoch_size,
                                           'KL Loss': epoch_KL/epoch_size},
                                            epoch+1)
        vae.writer.add_scalar('data/LearningRate', optimizer.param_groups[0]['lr'], epoch+1)
        
        if use_tensorboard == 'Full':
            for name, param in vae.named_parameters():
                vae.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch+1)
  
# Saving sounds/waveforms
    if np.mod(epoch,50) == 0: 
            
            #from validset
            fig = plt.figure(figsize = (12,8))
            for idx in range(1,5):
                plt.subplot(4,2,2*idx-1)
                inputs = mulaw.decode(valid_in[idx])
                plt.plot(inputs)
                plt.subplot(4,2,2*idx)
                output = mulaw.to_int(valid_out[idx])
                output = mulaw.decode(output)
                plt.plot(output.clone().cpu().numpy()) #still a variable
            fig.savefig(results_folder + '/images/reconstructions/valid_epoch'+str(epoch)+'.png', bbox_inches = 'tight' )
           
            
#saving models
    #checkpoints
    if np.mod(epoch,200) == 0: 
        if args.checkpoints:
            name = results_folder + '/checkpoints/temporal_' +str(nbBlocks) +'_' + str(nbLayers) + '_ep' + str(epoch) 
            vae.save(name, use_cuda)
    #bestmodel
    if valid_loss < best_valid and epoch>300:
        best_valid = valid_loss
        name = results_folder + '/temporal_' +str(nbBlocks) +'_' + str(nbLayers) + '_BEST'
        vae.save(name, use_cuda)        
#Print stats
    print('[End of epoch %d] \n recon_loss: %.3f \n KLloss: %.3f \n beta : %.3f \n loss: %.3f \n valid_loss: %.3f \n -----------------' %
              (epoch + 1,
               epoch_recon/epoch_size, 
               epoch_KL/epoch_size,
               beta,
               epoch_loss/epoch_size, 
               valid_loss))
    
    valid_in, valid_out = None,None
    gc.collect()

#5. TRAINING FINISHED
    
name = results_folder + '/temporal_' +str(nbBlocks) +'_' + str(nbLayers) + '_final'
vae.save(name)
print("MERCI DE VOTRE PATIENCE MAITRE. \n J'AI FINI L'ENTRAINEMENT ET JE NE SUIS QU'UNE VULGAIRE MACHINE ENTIEREMENT SOUMISE.")