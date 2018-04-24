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

    name_dur = []
    durations = []
    
    for entry in os.scandir(path):
        if entry.is_file():
            y,sr = librosa.load(entry.path)
            name = entry.path.split('/')[-1]
            dur = librosa.get_duration(y=y, sr=sr)
            durations.append(dur)
            name_dur.append((name,dur))
            
    return durations, name_dur
    
   
def print_stats(path = '/fast-1/DrumsDataset/data/Kicks'): 

    dur, names_dur = sound_length(path)
    
    mean = np.mean(dur)
    std = np.std(dur)
    max = np.max(dur)
    
    
    print("mean = %s s \n std = %s s \n max = %s s" % (mean, std, max))
    return None