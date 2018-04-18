#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 12:47:18 2018

@author: cyranaouameur
"""


use_tensorboard = True
use_bn = True

from VAE.logVar_VAE import Vanilla_VAE
from VAE.logVar_VAE import sample_z
#from VAE.visualize import saveInOut

import numpy as np
from datasets.MNIST import load_MNIST, test_MNIST
import torch
import torch.optim as optim
from torch.autograd import Variable

try:
    import matplotlib
    from matplotlib import pyplot as plt
except:
    import sys
    sys.path.append("/usr/local/lib/python3.6/site-packages/")
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt



use_cuda = torch.cuda.is_available()




x_dim = 28*28
h1_dim = 500
h2_dim = 200
z_dim = 10
mb_size = 100

n_epoch = 1000
wu_time = 100



#%%INIT MODEL
vae1 = Vanilla_VAE(x_dim, h1_dim, h2_dim, z_dim, mb_size, use_cuda, use_bn, use_tensorboard)
print(vae1)

if use_cuda :
    torch.cuda.set_device(1)
    print("**************************** USING CUDA ****************************")
    vae1.cuda()
    
trainloader = load_MNIST(vae1.mb_size)
testloader = test_MNIST()
#%%TRAINING
vae1.train()
        
optimizer = optim.Adam(vae1.parameters(), lr=0.0001)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)

epoch_size = 600

for epoch in range(n_epoch):
    
    #BETA WU
    if epoch < wu_time:
        beta = epoch / wu_time
    else :
        beta = 1
    ########
    
    
    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_KL = 0.0
    
    for i, data in enumerate(trainloader):
                
        # get the inputs
        raw_inputs, labels = data        
        inputs = raw_inputs.view(vae1.mb_size,vae1.x_dim)

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        inputs = (inputs*2)-1 #normal rescale

        optimizer.zero_grad()
    
        x = inputs
        if vae1.use_cuda:
            x = x.cuda()
        
        #forward
        z_mu, z_logvar = vae1.encode(x)
        z = sample_z(z_mu,z_logvar, vae1.mb_size, vae1.z_dim, vae1.use_cuda) 
        x_recon_mu, x_recon_logvar = vae1.decode(z)

        #compute losses
        recon_loss, kl_loss = vae1.G_loss(x, x_recon_mu, x_recon_logvar, z_mu, z_logvar)
        loss = recon_loss + beta*kl_loss
        
        epoch_loss += loss.data[0]
        epoch_recon += recon_loss.data[0]
        epoch_KL += kl_loss.data[0]
        
        # BACKPROP
        loss.backward()
        optimizer.step()
        
#TO DO AT THE END OF AN EPOCH
    #scheduler.step()
    if np.mod(epoch,50) == 0:
        
        raw = raw_inputs.view(vae1.mb_size, 28, 28)
        print(raw.size())
        x_recon = x_recon_mu.view(vae1.mb_size,28,28)
        
        fig = plt.figure()
        for idx in range(1,6):
        #print(recon)
            plt.subplot(2,5,idx)
            print(raw_inputs[idx].clone().cpu().size())
            plt.imshow(raw_inputs[idx].clone().cpu())
            plt.subplot(2,5,5+idx)
            plt.imshow(x_recon[idx].clone().cpu().data)
        fig.savefig('./results/images/check_epoch'+str(epoch)+'.png' )
    
    ###################   TENSORBOARD VISUALIZATION   ##############
    if vae1.use_tensorboard:
        for name, param in vae1.named_parameters():
            vae1.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch+1)

        vae1.writer.add_scalars('avglosses', {'loss': epoch_loss/epoch_size,
                                           'Recon_loss': epoch_recon/epoch_size,
                                           'KL_loss': epoch_KL/epoch_size},
                                            epoch+1)
#        if np.mod(epoch,50) == 0:
#            for j in range(2):
#                original = inputs[j]
#                original = original.view(28,28)
#                vae1.writer.add_image('Original_'+str(epoch)+str(j), original, epoch)
#                
#                img_rec = x_recon_mu[j]
#                img_rec = img_rec.view(28,28)
#                vae1.writer.add_image('Reconstructed_'+str(epoch)+str(j), img_rec, epoch)
    ##############################################################
    
    print('[End of epoch %d] \n beta : %.3f \n loss: %.3f \n recon_loss: %.3f \n KLloss: %.3f \n -----------------' %
                  (epoch + 1,
                   beta,
                   epoch_loss/epoch_size, 
                   epoch_recon/epoch_size, 
                   epoch_KL/epoch_size ))
#TRAINING ENDED        
if vae1.use_tensorboard:
    vae1.writer.close()
print("Finished")

#%%TRAINING ENDED

if use_cuda:
    vae1.cpu()


name = 'test1'
savepath = 'results/'+name
torch.save(vae1.state_dict(), savepath)


