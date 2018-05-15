#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:30:07 2018

@author: cyranaouameur
"""
import argparse
#%%Parse arguments

parser = argparse.ArgumentParser(description='Conv_VAE training and saving')


# VAE dimensions
parser.add_argument('config', type=int,
                    help='<Required> Choice of the net configuration')

parser.add_argument('z_dim', type=int,
                    help='<Required> Dimension of the latent space')

#data settings
parser.add_argument('--task', type=str, default='full', metavar='class', choices=['kicks', 'full'],
                    help='Define the class of instruments to train the model on (kicks or full)')
parser.add_argument('--downsample', type=int, default=1, metavar='N',
                    help='Define the downsampling factor (default: 1 -> no downsample)')

# training settings
parser.add_argument('--mb_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='number of epochs to train (default: 5000)')
parser.add_argument('--beta', type=int, default=1, metavar='N',
                    help='beta coefficient for regularization (default: 1)')
parser.add_argument('--Nwu', type=int, default=100, metavar='N',
                    help='epochs number for warm-up (default: 100)')

parser.add_argument('--gpu', type=int, default=1, metavar='N',
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
from skimage.transform import resize

import os

import torch
import torch.optim as optim
from torch.autograd import Variable

from outils.scaling import scale_array
from models.VAE.Conv_VAE import Conv_VAE, conv_loss, layers_config

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
if use_cuda:
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    print('USING CUDA ON GPU' + str(torch.cuda.current_device()))
    
#%% Compute transforms and load data
log_scaling = True
normalize = 'gaussian'

task = args.task


dataset = importDataset(targetDur = 1.15583)

dataset.metadata['instrument'] = np.array(dataset.metadata['instrument']) #to array
dataset.data = np.abs(dataset.data) # to real positive array

if task == 'kicks':
    print('TRAINING ONLY ON KICKS')
    dataset.data = dataset.data[dataset.metadata['instrument']==0]    
    dataset.metadata['instrument'] = dataset.metadata['instrument'][dataset.metadata['instrument']==0]    



#downsample by a given factor
nbFrames, nbBins = dataset.get(0).shape
downFactor = args.downsample
downsampled = []
for img in dataset.data:
    downsampled.append(resize(img, (int(nbFrames / downFactor), nbBins), mode='constant'))
in_shape = (int(nbFrames / downFactor), nbBins)

#Scale data
dataset.data, norm_const = scale_array(downsampled, log_scaling, normalize) 

#Constrcut partitions (train and validation sets)
dataset.constructPartition('instrument', ['train','valid'], [0.8, 0.2])

#Compute the best mb_size for valid_set
len_val = len(dataset.partitions['valid'])
valid_mb = [x for x in range(len_val+1) if x != 0 and len_val%x == 0 and x<150][-1]

mb_size = args.mb_size

#Create the Loaders
trainloader = DataLoader(dataset, mb_size, 'instrument', partition = 'train') 
testloader = DataLoader(dataset, valid_mb, 'instrument', partition = 'valid')

#TODO : to torch tensor?


#%% Define the parameters of the model (configs are in models/VAE/Conv_VAE.py) :
conv, h_dims, deconv = layers_config(args.config)
z_dim = args.z_dim



#Hyper parameters : non-linearity? batchnorm? dropout?
nnLin = ['relu','none'] #[h_act, out_act] with 'relu' or 'tanh' or 'elu' or 'none'
use_bn = True
dropout = False # False or a prob between 0 and 1
final_beta = args.beta
wu_time = args.Nwu
use_tensorboard = True #True, False or 'Full' (histograms)




#initialize the model and use cuda if available
vae = Conv_VAE(conv, h_dims, z_dim, deconv, nnLin, use_bn, dropout, use_cuda)
if use_cuda :
    vae.cuda()
if use_tensorboard:
    from tensorboardX import SummaryWriter
    vae.writer = SummaryWriter()
    
    
#%%Training routine
    
model_name = 'conv_config'+str(args.config)
results_folder = './results/'+model_name
if not os.path.isdir(results_folder):
    os.makedirs(results_folder)
    os.makedirs(results_folder + '/images/reconstructions')
    os.makedirs(results_folder + '/checkpoints')

nb_epochs = args.epochs
vae.train()
optimizer = optim.Adam(vae.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=100, min_lr = 5e-06)

best_valid = 1e8

for epoch in range(nb_epochs):

#BETA WU
    beta = final_beta * min(1,epoch/wu_time)
########

    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_KL = 0.0
    
    for i, data in enumerate(trainloader) : 
        optimizer.zero_grad()
        
        #1. get the inputs and wrap them in Variable
        raw_inputs, labels = data
        pre_process, labels = torch.from_numpy(raw_inputs).float(), torch.from_numpy(labels)
        if use_cuda:
            pre_process = pre_process.cuda()
        pre_process = pre_process.unsqueeze(1)
        x, labels = Variable(pre_process), Variable(labels)
        
        
        #2. Forward data
        rec_mu, rec_logvar, z_mu, z_logvar = vae.forward(x)

        #3. Compute losses (+validation loss)
        recon_loss, kl_loss = conv_loss(x, rec_mu, rec_logvar, z_mu, z_logvar)
        loss = recon_loss + beta*kl_loss
        
        epoch_loss += loss.data[0]
        epoch_recon += recon_loss.data[0]
        epoch_KL += kl_loss.data[0]
        
        #4. Backpropagate
        loss.backward()
        optimizer.step()

#4. EPOCH FINISHED :
    
    epoch_size = i+1
    
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
  
#Saving images
    if np.mod(epoch,50) == 0: 
        #from training set
        fig = plt.figure(figsize = (12,8))
        for idx in range(1,6):
            plt.subplot(2,5,idx)
            inputs = pre_process[idx].view(in_shape[0], in_shape[1])
            plt.imshow(inputs.clone().cpu().t(), aspect = 'auto')
            plt.subplot(2,5,5+idx)
            output = rec_mu[idx].view(in_shape[0], in_shape[1])
            plt.imshow(output.clone().cpu().data.t(), aspect = 'auto') #still a variable
        fig.savefig(results_folder + '/images/reconstructions/train_epoch'+str(epoch)+'.png', bbox_inches = 'tight')
        
        #from validset
        fig = plt.figure(figsize = (12,8))
        for idx in range(1,6):
            plt.subplot(2,5,idx)
            inputs = valid_in[idx].view(in_shape[0], in_shape[1])
            plt.imshow(inputs.clone().cpu().t(), aspect = 'auto')
            plt.subplot(2,5,5+idx)
            output = valid_out[idx].view(in_shape[0], in_shape[1])
            plt.imshow(output.clone().cpu().data.t(), aspect = 'auto') #still a variable
        fig.savefig(results_folder + '/images/reconstructions/valid_epoch'+str(epoch)+'.png', bbox_inches = 'tight' )
        
#saving models
    #checkpoints
    if np.mod(epoch,200) == 0: 
        if args.checkpoints:
            name = results_folder + '/checkpoints/conv_config'+str(args.config) + '_ep' + str(epoch)
            vae.save(name, use_cuda)
    #bestmodel
    if valid_loss < best_valid and epoch>300:
        best_valid = valid_loss
        name = results_folder + '/checkpoints/conv_config'+str(args.config) + '_ep' + str(epoch) + '_BEST'
        vae.save(name, use_cuda)        
#Print stats
    print('[End of epoch %d] \n recon_loss: %.3f \n KLloss: %.3f \n beta : %.3f \n loss: %.3f \n valid_loss: %.3f \n -----------------' %
              (epoch + 1,
               epoch_recon/epoch_size, 
               epoch_KL/epoch_size,
               beta,
               epoch_loss/epoch_size, 
               valid_loss))

#5. TRAINING FINISHED
    
name = results_folder + '/conv_config'+str(args.config) + '_final'
vae.save(name)
print("MERCI DE VOTRE PATIENCE MAITRE. \n J'AI FINI L'ENTRAINEMENT ET JE NE SUIS QU'UNE VULGAIRE MACHINE ENTIEREMENT SOUMISE.")