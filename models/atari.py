#!/usr/bin/env python

"""
    models/atari.py
"""

import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable

from joint import SoftmaxPPO

class AtariPPO(SoftmaxPPO):
    
    def __init__(self, input_channels, input_height, input_width, n_outputs, 
        entropy_penalty=0.0, adam_lr=None, adam_eps=None, clip_eps=None, cuda=True):
        
        super(AtariPPO, self).__init__()
        
        self.input_shape = (input_channels, input_height, input_width)
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )
        
        self.conv_output_dim = self._compute_sizes(input_channels, input_height, input_width)
        self.fc = nn.Linear(self.conv_output_dim, 512)
        
        self.policy_fc = nn.Linear(512, n_outputs)
        self.value_fc = nn.Linear(512, 1)
        
        if adam_lr and adam_eps:
            self.opt = torch.optim.Adam(self.parameters(), lr=adam_lr, eps=adam_eps)
            self.clip_eps = clip_eps
            self.entropy_penalty = entropy_penalty
            
            self._old = AtariPPO(input_channels, input_height, input_width, n_outputs, cuda=cuda)
        
        self._cuda = cuda
    
    def _compute_sizes(self, input_channels, input_height, input_width):
        tmp = Variable(torch.zeros((1, input_channels, input_height, input_width)), volatile=True)
        tmp = self.conv_layers(tmp)
        return tmp.view(tmp.size(0), -1).size(-1)
    
    def forward(self, x):
        x = x / 255.0
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.policy_fc(x), self.value_fc(x)

