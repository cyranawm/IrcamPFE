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
import librosa
from nsgt.fscale import OctScale, LogScale, MelScale
from nsgt.cq import NSGT



def get_phase(filename, targetLen):
    sig, sr = librosa.load(filename)
    original_name = filename.split('/')[-1]
    print('** Computing for '+original_name + ' **')
    # Ensure signal is of rightful size
    if (len(sig) > targetLen):
        sig = sig[:int(targetLen)]
        # Otherwise pad with zeros
    else:
        sig = np.pad(sig, (0,int(np.round(targetLen)) - len(sig)), 'constant', constant_values = 0);
        
    # Create a frequency scale
    scl = OctScale(30, 11000, 48)
        
    # Create a NSGT object
    nsgt = NSGT(scl, sr, targetLen, real=True, matrixform=True, reducedform=1)
    
    #Forward pass
    nsgt_cplx = np.array(list(nsgt.forward(sig)))
    phase = np.angle(nsgt_cplx)
    
    return phase





def regenerate(VAE, dataset, nb, it, scale_param, scaling, log_scaling, downFactor, soundPath, crop = None, initPhase = True, nameExtension = ''):
    
    targetLen = 25486
    
    for i, raw_input in enumerate(dataset.data):
        if i>nb:
            break
        
        pre_process = torch.from_numpy(raw_input).float()
        if torch.cuda.is_available():
            pre_process = pre_process.cuda()
        pre_process = pre_process.unsqueeze(0)
        pre_process = pre_process.unsqueeze(0)#add 2 dimensions to forward into vae
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
        if initPhase:
            phase = get_phase(dataset.files[i], targetLen)
        else:
            phase = None
        
        filename =  str(i) + nameExtension #to test on various parameters sets
        regenerateAudio(originalNSGT, targetLen = targetLen, iterations=it, curName= soundPath + filename, initPhase = phase, crop = crop)        
        regenerateAudio(recNSGT, targetLen = targetLen, iterations=it, curName= soundPath + filename + '_rec', initPhase = phase, crop = crop)
        
        
def get_nn(coords, point):
    
    distances = []
    
    for data in coords:
        distances.append(np.linalg.norm(data-point))
        
    nn = np.argmin(distances)
    
    return nn
        
        
        
def create_line(a, b, nb_steps):
    """ From two points, creates a segment in a high dimensional space.
    Parameters
    ----------
    nb_steps : int
        The number of points generated
    a, b : np arrays
        The two datapoints that define the line
    Returns
    -------
    discrete_line : np array of dimension nb_chunks_mix x dim_embedd_space
        A set of nb_chunks_mix points in the latent space that belong to the segment [a,b].
        This defines a straight path between them in the latent space.
    Example
    -------
    a = embedded_data[idx_a,:]
    b = embedded_data[idx_b,:]
    discrete_line = create_line(a, b, 10)
    """
    if (a.shape != b.shape):
        raise(ValueError)
    

    # Sample nb_steps points from the line
    t = np.linspace(0,1,nb_steps)
    discrete_line = a.T*t + b.T*(1-t)
    discrete_line = np.transpose(discrete_line)
    return discrete_line        
        
    
    


