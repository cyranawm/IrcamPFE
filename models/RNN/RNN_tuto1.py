# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import glob

def findFiles(path): return glob.glob(path)

print(findFiles('../datasets/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {} 
all_categories = []

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('../datasets/names/*.txt'):
    category = filename.split('/')[-1].split('.')[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

#%%
import torch
import torch.optim as optim


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

print(letterToTensor('J'))

print(lineToTensor('Jones').size())

#%%
import time

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


import math
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.criterion = nn.NLLLoss()
        
    def forward(self, datain, hidden):
        combined = torch.cat((datain, hidden),1)
        
        output = self.i2o(combined)
        output = self.softmax(output)
        hidden = self.i2h(combined)
        
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
    def loss(self, out, truth):
        return self.criterion(out,truth)
    
    def train(self, n_iters):
        
        print_every = 5000
        plot_every = 1000
        
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        all_losses = []
        start = time.time()
        
        hidden = Variable(self.init_hidden())
        self.zero_grad()
        
        current_loss = 0.0
        
        for iter in range(n_iters):
            
            self.zero_grad()
            
            hidden = Variable(hidden.data)

        
            category, line, category_tensor, line_tensor = randomTrainingExample()
            
            for i in range(line_tensor.size()[0]):
                output, hidden = self.forward(line_tensor[i], hidden)

            loss = self.criterion(output, category_tensor)
            
            loss.backward()
            optimizer.step()
            
            current_loss += loss.data[0]
            

            # Print iter number, loss, name and guess
            if iter % print_every == 0:
                guess, guess_i = categoryFromOutput(output)
                correct = '✓' if guess == category else '✗ (%s)' % category
                print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
    
            # Add current loss avg to list of losses
            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0
        
        return all_losses
    
#%%
        
n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

input1 = Variable(letterToTensor('A'))
hidden1 = Variable(rnn.init_hidden())

output1, next_Hidden = rnn(input1, hidden1)



def categoryFromOutput(output):
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    category_i = top_i[0][0]
    return all_categories[category_i], category_i

import random

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category, '/ line =', line)


#%%
    
all_losses = rnn.train(150000)

#%%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)

#%%

def evaluate(line_tensor):
    hidden = Variable(rnn.init_hidden())

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    return output

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    output = evaluate(Variable(lineToTensor(input_line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

#%%
predict('Gabanoka')




