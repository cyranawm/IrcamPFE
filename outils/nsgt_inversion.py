#%%!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:08:53 2018

@author: cyranaouameur
"""
import numpy as np
import librosa
from nsgt.cq import NSGT
from nsgt.fscale import OctScale, LogScale, MelScale
from skimage.transform import resize

def pre_process_output():
    print('Mata ne')
    
def regenerateAudio(data, minFreq = 30, maxFreq = 11000, nsgtBins = 48, sr = 22050, scale = 'oct', targetLen = int(3 * 22050), iterations = 100, momentum=False, testSize=False, curName=None):
    # Create a scale
    if (scale == 'oct'):
        scl = OctScale(minFreq, maxFreq, nsgtBins)
    if (scale == 'mel'):
        scl = MelScale(minFreq, maxFreq, nsgtBins)
    if (scale == 'log'):
        scl = LogScale(minFreq, maxFreq, nsgtBins)
    # Create the NSGT object
    nsgt = NSGT(scl, sr, targetLen, real=True, matrixform=True, reducedform=1)
    # Run a forward test 
    if (testSize):
        testForward = np.array(list(nsgt.forward(np.zeros((targetLen)))))
        print(testForward.shape)
        targetFrames = testForward.shape[1]
        nbFreqs = testForward.shape[0]
        #assert(data.shape[0] == nbFreqs)
        #assert(data.shape[1] == targetFrames)
        return nbFreqs, targetFrames
    # Now Griffin-Lim dat
    print('Start Griffin-Lim')
    p = 2 * np.pi * np.random.random_sample(data.shape) - np.pi
    for i in range(iterations):
        S = data * np.exp(1j*p)
        inv_p = np.array(list(nsgt.backward(S)))#transformHandler(S, transformType, 'inverse', options)
        new_p = np.array(list(nsgt.forward(inv_p)))#transformHandler(inv_p, transformType, 'forward', options)
        new_p = np.angle(new_p)
        # Momentum-modified Griffin-Lim
        if (momentum):
            p = new_p + ((i > 0) * (0.99 * (new_p - p)))
        else:
            p = new_p
    # Save the output
    if (curName is not None):
        librosa.output.write_wav(curName + '.wav', inv_p, sr, norm = True)
        
if __name__ == '__main__':
    # Duration of output signal
    tOut = 3
    # Import a signal
    sig, sr = librosa.load('./tests/data/0001.wav')
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
    print(nbFreq)
    print(nbFrames)
    # Remove phase
    finalDistribNSGT = np.abs(finalDistribNSGT)
    # Downsample by a given factor
    downFactor = 100
    # DOWNSAMPLE the distribution
    downsampNSGT = resize(finalDistribNSGT, (nbFreq, int(nbFrames / downFactor)), mode='constant')
    #
    # Here is learning and shit
    #
    # RE-UPSAMPLE the distribution
    downsampNSGT = resize(downsampNSGT, (nbFreq, nbFrames), mode='constant')
    # Now invert (with upsampled version)
    regenerateAudio(downsampNSGT, targetLen = int(tOut * sr), iterations=100, curName='poulpe.wav')