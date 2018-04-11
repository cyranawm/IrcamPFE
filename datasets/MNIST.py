# -*- coding: utf-8 -*-

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