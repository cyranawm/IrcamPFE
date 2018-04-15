# -*- coding: utf-8 -*-

from VAE.logVar_VAE import Vanilla_VAE
from VAE.Conv_VAE import Conv_VAE
from VAE.visualize import plotInOut, plotInOut_Conv

import torch
import torchvision
import torchvision.transforms as transforms


from datasets.MNIST import load_MNIST, test_MNIST


testloader = test_MNIST()
trainloader = load_MNIST(1)


#%%   NORMAL VAE

name = 'test1'
savepath = 'results/'+name

state_dict = torch.load(savepath)

x_dim = 28*28
h_dim = 500
z_dim = 10

vae1 = Vanilla_VAE(x_dim,h_dim,z_dim, use_cuda = False)
vae1.load_state_dict(state_dict)
vae1.eval()

#%%
plotInOut(testloader,vae1)



#%%   CONV VAE

name = 'conv1'
savepath = 'results/'+name

state_dict = torch.load(savepath)

h_dim = 500
z_dim = 10

vae1 = Conv_VAE(h_dim,z_dim, use_cuda = False)
vae1.load_state_dict(state_dict)

#%%

plotInOut_Conv(testloader,vae1)
