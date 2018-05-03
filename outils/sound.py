#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 17:14:29 2018

@author: cyranaouameur
"""
import numpy as np
import torch
from torch.autograd import Variable
from outils.nsgt_inversion import regenerateAudio
from skimage.transform import resize
from outils.scaling import unscale_array





def regenerate(VAE, dataloader, scale_param, scaling, log_scaling, downFactor, soundPath):
    
    targetLen = 16384
    it = 1000
    
    for i, data in enumerate(dataloader):
        if i>10:
            break
        
        raw_input, label = data
        pre_process = torch.from_numpy(raw_input).float()
        if torch.cuda.is_available():
            pre_process = pre_process.cuda()
        pre_process = pre_process.unsqueeze(1)
        x = Variable(pre_process)
        
        #2. Forward data
        rec_mu, rec_logvar, z_mu, z_logvar = VAE.forward(x)
        
        #suppress dumb sizes and transpose to regenerate
        originalNSGT = pre_process.data.cpu()[0,0,:,:].numpy().T
        recNSGT = rec_mu.data.cpu()[0,0,:,:].numpy().T
        
        #compute the resize needed
        nbFreq, nbFrames = regenerateAudio(np.zeros((1, 1)), testSize = True, targetLen = targetLen)

        # RE-UPSAMPLE the distribution
        oriFactor = np.max(np.abs(originalNSGT))
        recFactor = np.max(np.abs(recNSGT))
        
        originalNSGT = resize(originalNSGT/oriFactor, (nbFreq, nbFrames), mode='constant')
        recNSGT = resize(recNSGT/recFactor, (nbFreq, nbFrames), mode='constant')
        
        originalNSGT *= oriFactor
        recNSGT *= recFactor
        
        #rescale
        originalNSGT = unscale_array(originalNSGT, scale_param, scaling, log_scaling)
        recNSGT =unscale_array(recNSGT, scale_param, scaling, log_scaling)
        
        # Now invert (with upsampled version)
        name =  str(i)
        regenerateAudio(originalNSGT, targetLen = targetLen, iterations=it, curName= soundPath+name)        
        regenerateAudio(recNSGT, targetLen = targetLen, iterations=it, curName= soundPath + name + '_rec')