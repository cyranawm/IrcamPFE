# -*- coding: utf-8 -*-

from VAE.Vanilla_VAE_NN import Vanilla_VAE
from VAE.visualize import plotInOut

import torch
import torchvision
import torchvision.transforms as transforms


from datasets.MNIST import load_MNIST, test_MNIST



name = 'test1'
savepath = 'results/'+name

state_dict = torch.load(savepath)

x_dim = 28*28
h_dim = 500
z_dim = 10

vae1 = Vanilla_VAE(x_dim,h_dim,z_dim, use_cuda = False)
vae1.load_state_dict(state_dict)




testloader = test_MNIST()
#trainloader = load_MNIST(1)






#%%
plotInOut(testloader,vae1)



