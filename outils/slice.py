#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:50:56 2018

@author: cyranaouameur
"""

import os
import librosa
from numpy.random import permutation
import numpy as np

def slice_and_split(dataDirectory, trainDirectory, testDirectory, sliceDur, trainRatio = 0.8, overlap = None):
    files = [i for i in os.listdir(dataDirectory) if i.endswith('.wav')]
    
    files = np.permutation(files)
    train = files[:int(trainRatio*len(files))]
    test = files[int(trainRatio*len(files)):]
    
    
    for file in train:
        y,sr = librosa.load(os.path.join(dataDirectory,file))
        name = file.split('.')[0]
        fileLen = len(y)
        sliceLen = int(sliceDur*sr)
        nbSlices = int(fileLen / sliceLen)
        
        if overlap is not None:
            i=0
            stepSize = (float(100 - overlap)/100) * sliceLen
            while (i+1)*stepSize <= fileLen :
                start = i*stepSize
                curSlice = y[start : start + sliceLen]
                path = os.path.join(trainDirectory, name + '_slice' + str(i) + '.wav')
                librosa.output.write_wav(path, curSlice, sr, norm = True)
                i += 1
           
        else: 
            for i in range(nbSlices):
                curSlice = y[i*sliceLen : (i+1)*sliceLen]
                path = os.path.join(trainDirectory, name + '_slice' + str(i) + '.wav')
                librosa.output.write_wav(path, curSlice, sr, norm = True)
                
                
    for file in test:
        y,sr = librosa.load(os.path.join(dataDirectory,file))
        name = file.split('.')[0]
        fileLen = len(y)
        sliceLen = int(sliceDur*sr)
        nbSlices = int(fileLen / sliceLen)
        
        if overlap is not None:
            i=0
            stepSize = (float(100 - overlap)/100) * sliceLen
            while (i+1)*stepSize <= fileLen :
                start = i*stepSize
                curSlice = y[start : start + sliceLen]
                path = os.path.join(testDirectory, name + '_slice' + str(i) + '.wav')
                librosa.output.write_wav(path, curSlice, sr, norm = True)
                i += 1
           
        else: 
            for i in range(nbSlices):
                curSlice = y[i*sliceLen : (i+1)*sliceLen]
                path = os.path.join(testDirectory, name + '_slice' + str(i) + '.wav')
                librosa.output.write_wav(path, curSlice, sr, norm = True)

def slice_data(dataDirectory, targetDirectory, sliceDur, overlap = None):
    
    for entry in os.scandir(dataDirectory):
        if entry.path.endswith('.wav'):
            y,sr = librosa.load(entry.path)
            name = entry.path.split('/')[-1]
            name = name.split('.')[0]
            fileLen = len(y)
            sliceLen = int(sliceDur*sr)
            nbSlices = int(fileLen / sliceLen)
            
            if overlap is not None:
                i=0
                stepSize = (float(100 - overlap)/100) * sliceLen
                while (i+1)*stepSize <= fileLen :
                    start = i*stepSize
                    curSlice = y[start : start + sliceLen]
                    path = os.path.join(targetDirectory, name + '_slice' + str(i) + '.wav')
                    librosa.output.write_wav(path, curSlice, sr, norm = True)
                    i += 1
               
            else: 
                for i in range(nbSlices):
                    curSlice = y[i*sliceLen : (i+1)*sliceLen]
                    path = os.path.join(targetDirectory, name + '_slice' + str(i) + '.wav')
                    librosa.output.write_wav(path, curSlice, sr, norm = True)
                



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Slice audio files into targetDirectory')


    parser.add_argument('dataDirectory', type=str,
                        help='Source directory')
    
    parser.add_argument('targetDirectory', type=str,
                        help='Destination')
    
    parser.add_argument('sliceDur', type=float,
                        help= 'Duration of a slice (in seconds)')
    
    args = parser.parse_args()
    
    slice_data(args.dataDirectory, args.targetDirectory, args.sliceDur)
    
    
    