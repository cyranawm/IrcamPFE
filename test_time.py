#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 18:40:22 2018

@author: cyranaouameur
"""

import numpy as np
import librosa
from nsgt.cq import NSGT
from nsgt.fscale import OctScale, LogScale, MelScale
from outils.nsgt_inversion import regenerateAudio
from matplotlib import pyplot as plt
import time

sig, fs = librosa.load('./tests/data/kick.wav')

lengths = [int(i) for i in np.linspace(16384, 32768, num = 100)]
times = []

mFreq = 30
maFreq = 11000
bins = 48

# Create a frequency scale
scl = OctScale(mFreq, maFreq, bins)

for l in lengths:
    padded_sig = np.pad(sig, (0, l-len(sig)), mode = 'constant')
    
    start = time.time()
        
    # Create a NSGT object
    nsgt = NSGT(scl, fs, len(padded_sig), real=True, matrixform=True, reducedform=1)
    sig_nsgt = np.abs(list(nsgt.forward(padded_sig)))
    regenerateAudio(sig_nsgt, sr=fs, targetLen = len(padded_sig), iterations=20, nsgtBins=bins, minFreq=mFreq, maxFreq=maFreq)
    
    end = time.time()
    elapsed = end - start
    times.append(elapsed)
    
    
#target time = 0.74304
#target 2 = 1.15583