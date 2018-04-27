#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:32:06 2018

@author: cyranaouameur
"""
from outils.nsgt_inversion import regenerateAudio
import numpy as np
import librosa
from nsgt.cq import NSGT
from nsgt.fscale import OctScale, LogScale, MelScale
from skimage.transform import resize
import os


# Duration of output signal
tOut = 1

#Downsampling factors
downFactors = [1,2,3,5,10]

#number of Griffin-Lim iterations
nbIterations = [5,10,20,50,100]

for entry in os.scandir('./tests/data'):
    if entry.path.endswith('.wav'):
        # Import a signal
        sig, sr = librosa.load(entry.path)
        original_name = entry.path.split('/')[-1]
        print('** Computing for '+original_name + '**')
        # Ensure signal is of rightful size
        if ((len(sig) / sr) > tOut):
            sig = sig[:int((tOut * sr))]
            # Otherwise pad with zeros
        else:
            sig = np.pad(sig, (0,int(np.round(tOut * sr)) - len(sig)), 'constant', constant_values = 0);
            
        # Create a frequency scale
        scl = OctScale(30, 11000, 48)
        # Create a NSGT object
        nsgt = NSGT(scl, sr, tOut * sr, real=True, matrixform=True, reducedform=1)
        # Compute and turn into Numpy array
        finalDistribNSGT = np.array(list(nsgt.forward(sig)))
        # Checkout the number of frames 
        nbFreq, nbFrames = regenerateAudio(np.zeros((1, 1)), testSize = True, targetLen = int(tOut * sr))
        # ! WARNING ! NSGT AND INVERSION WORK ON FREQ x TIME
        # BUT TOOLBOX AND LEARNING ARE ON TIME x FREQ
        # Remove phase
        finalDistribNSGT = np.abs(finalDistribNSGT)
        for downFactor in downFactors:
            for it in nbIterations:
                # DOWNSAMPLE the distribution
                downsampNSGT = resize(finalDistribNSGT, (nbFreq, int(nbFrames / downFactor)), mode='constant')
                #
                # Here is learning and shit
                #
                # RE-UPSAMPLE the distribution
                downsampNSGT = resize(downsampNSGT, (nbFreq, nbFrames), mode='constant')
                # Now invert (with upsampled version)
                name =  original_name + '_down' + str(downFactor) + '_it' + str(it)
                regenerateAudio(downsampNSGT, targetLen = int(tOut * sr), iterations=it, curName='tests/res/'+name)