#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 17:50:56 2018

@author: cyranaouameur
"""

import os
import librosa


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
                    path = os.path.join(targetDirectory, name + '_slice' + str(i+1) + '.wav')
                    librosa.output.write_wav(path, curSlice, sr, norm = True)
                    i += 1
               
            else: 
                for i in range(nbSlices):
                    curSlice = y[i*sliceLen : (i+1)*sliceLen]
                    path = os.path.join(targetDirectory, name + '_slice' + str(i+1) + '.wav')
                    librosa.output.write_wav(path, curSlice, sr, norm = True)
                



if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Slice audio files into targetDirectory')


    parser.add_argument('dataDirectory', type=str,
                        help='Source directory')
    
    parser.add_argument('targetDirectory', type=str,
                        help='Destination')
    
    parser.add_argument('sliceDur', type=float,
                        help='<Required> Dimension of the latent space')
    
    args = parser.parse_args()
    
    slice_data(args.dataDirectory, args.targetDirectory, args.sliceDur)
    
    
    