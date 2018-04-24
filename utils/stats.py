#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 13:30:13 2018

@author: cyranaouameur
"""

import librosa
import os
import numpy as np

def sound_length(path):

    durations = []
    
    for entry in os.scandir(path):
        if entry.is_file():
            y,sr = librosa.load(entry.path)
            dur = librosa.get_duration(y=y, sr=sr)
            durations.append(dur)
            
    return durations
    
   
    
path = '/fast_1/DrumsDataset/data/Kicks'
dur = sound_length(path)

mean = np.mean(dur)
std = np.std(dur)
max = np.max(dur)

print("mean = %s s \n std = %s s \n max = %s s", (mean, std, max))