# -*- coding: utf-8 -*-
import torch


def scale_data(raw_data,log_scaling,scaling):
    
    if log_scaling:
        raw_data = torch.log(raw_data)
        
        
    if scaling == '0centering': # 0 centred
        print('scaling data to 0 centred 1 scale')
        
        data_mean = torch.mean(raw_data)
        raw_data -= data_mean
        
        data_weight = torch.max(torch.abs(raw_data))
        scaled_data = raw_data / data_weight # 0 centred in [-1,1]
        
        constants = [data_mean, data_weight]

         
    elif scaling == 'unitnorm': # stretched [in -1,1]
        print('scaling data to [-1,1] full range')
        
        data_min = torch.min(raw_data)
        raw_data -= data_min
        
        data_weight = 2 / torch.max(raw_data)
        scaled_data = raw_data * data_weight - 1
        
        constants = [data_min, data_weight]
        
        
    elif scaling == 'gaussian': # 0 mean and unit variance
        print('scaling data to 0 mean and unit variance')
        
        data_mean = torch.mean(raw_data)
        raw_data -= data_mean

        data_weight = torch.std(raw_data)
        scaled_data = raw_data/data_weight

        constants = [data_mean, data_weight]


    return scaled_data, constants