# -*- coding: utf-8 -*-
import numpy as np

class DataLoader(object):
    def __init__(self, dataset, batch_size, task=None, partition=None, transforms=[], *args, **kwargs):
        self.dataset = dataset
        self.batch_size = batch_size
        self.transforms = transforms
        if not issubclass(type(self.transforms), list):
            self.transforms = [self.transforms]
        if partition is None:
            random_indices = permutation(length(dataset.data)) 
        else:
            partition_ids = dataset.partitions[partition]
            random_indices = partition_ids[permutation(len(partition_ids))]
        n_batches = len(random_indices)//batch_size
        self.random_ids = np.split(random_indices[:n_batches*batch_size], len(random_indices)//batch_size)
        self.task = task
    
    def transform(self, data):
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def __iter__(self):
        for i in range(len(self.random_ids)):
            if issubclass(type(self.dataset.data), list):
                x = [d[self.random_ids[i]] for d in self.dataset.data]
            else:
                x = self.dataset.data[self.random_ids[i]]
            x = self.transform(x)
            if not self.task is None:
#                yield self.transform(self.dataset.data[self.random_ids[i]]), self.dataset.metadata[self.task][self.random_ids[i]]
                y = self.dataset.metadata[self.task][self.random_ids[i]]
            else:
                y = None
#                yield self.transform(self.dataset.data[self.random_ids[i]]), None
            yield x,y