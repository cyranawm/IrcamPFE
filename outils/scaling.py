# -*- coding: utf-8 -*-
import numpy as np
import torch


def scale_array(raw_data,log_scaling,scaling):
    
    if log_scaling:
        raw_data = np.log(raw_data)
        
        
    if scaling == '0centering': # 0 centred    
        data_mean = np.mean(raw_data)
        raw_data -= data_mean
        data_weight = np.max(np.abs(raw_data))
        scaled_data = raw_data / data_weight # 0 centred in [-1,1]
        
        constants = [data_mean, data_weight]

         
    elif scaling == 'unitnorm': # stretched [in -1,1]       
        data_min = np.min(raw_data)
        raw_data -= data_min    
        data_weight = 2 / np.max(raw_data)
        scaled_data = raw_data * data_weight - 1
        
        constants = [data_min, data_weight]
        
        
    elif scaling == 'gaussian': # 0 mean and unit variance 
        data_mean = np.mean(raw_data)
        raw_data -= data_mean
        data_weight = np.std(raw_data)
        scaled_data = raw_data/data_weight

        constants = [data_mean, data_weight]


    return scaled_data, constants






def unscale_array(scaled_data,data_scales,scaling,log_scaling):
    if scaling == '0centering': # 0 centred
        scaled_data *= data_scales[1]
        unscaled_data = scaled_data+data_scales[0]
        
    elif scaling == 'unitnorm': # stretched [in -1,1]
        scaled_data = (scaled_data+1)/data_scales[1]
        unscaled_data = scaled_data+data_scales[0]
        
    elif scaling == 'gaussian': # 0 mean and unit variance
        scaled_data *= data_scales[1]
        unscaled_data = scaled_data+data_scales[0]
        
    if log_scaling:
        unscaled_data = np.exp(unscaled_data)
        
    return unscaled_data







#def scale_tensor(raw_data,log_scaling,scaling):
#    
#    if log_scaling:
#        raw_data = torch.log(raw_data)
#        
#        
#    if scaling == '0centering': # 0 centred
#        
#        data_mean = torch.mean(raw_data)
#        raw_data -= data_mean
#        
#        data_weight = torch.max(torch.abs(raw_data))
#        scaled_data = raw_data / data_weight # 0 centred in [-1,1]
#        
#        constants = [data_mean, data_weight]
#
#         
#    elif scaling == 'unitnorm': # stretched [in -1,1]
#        
#        data_min = torch.min(raw_data)
#        raw_data -= data_min
#        
#        data_weight = 2 / torch.max(raw_data)
#        scaled_data = raw_data * data_weight - 1
#        
#        constants = [data_min, data_weight]
#        
#        
#    elif scaling == 'gaussian': # 0 mean and unit variance
#        
#        data_mean = torch.mean(raw_data)
#        raw_data -= data_mean
#
#        data_weight = torch.std(raw_data)
#        scaled_data = raw_data/data_weight
#
#        constants = [data_mean, data_weight]
#
#
#    return scaled_data, constants