# -*- coding: utf-8 -*-

from VAE.Vanilla_VAE_NN import Vanilla_VAE
from VAE.visualize import visu_recon

import torch
import torchvision
import torchvision.transforms as transforms



def test_MNIST():
    transform = transforms.ToTensor()
    MNIST = torchvision.datasets.MNIST("./datasets/MNIST/", train=False, transform=transform, target_transform=None, download=True)
    
    test_loader = torch.utils.data.DataLoader(MNIST,
                                          batch_size=1,
                                          shuffle=True)
    return test_loader

def load_MNIST(mb_size):
    transform = transforms.ToTensor()
    MNIST = torchvision.datasets.MNIST("./datasets/MNIST/", train=True, transform=transform, target_transform=None, download=True)
    
    data_loader = torch.utils.data.DataLoader(MNIST,
                                          batch_size=mb_size,
                                          shuffle=True)
    return data_loader


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
visu_recon(1,testloader,vae1)


