#!/usr/bin/env python

"""
    models/path.py
"""

import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable

from models.joint import MultiSoftmaxPPO, JointPPO

class SinglePathPPO(MultiSoftmaxPPO):
    def __init__(self, n_inputs=32, n_outputs=4, output_channels=2,
        entropy_penalty=0.0, adam_lr=None, adam_eps=None, clip_eps=None, cuda=True):
        
        super(SinglePathPPO, self).__init__(output_channels=output_channels)
        
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        self.policy_fc = nn.Linear(64, n_outputs * output_channels)
        self.value_fc = nn.Linear(64, 1)
        
        if adam_lr and adam_eps:
            self.opt = torch.optim.Adam(self.parameters(), lr=adam_lr, eps=adam_eps)
            self.clip_eps = clip_eps
            self.entropy_penalty = entropy_penalty
            
            self._old = SinglePathPPO(n_inputs, n_outputs, cuda=cuda)
        
        self._cuda = cuda
    
    def forward(self, x):
        x = self.layers(x)
        return self.policy_fc(x), self.value_fc(x)

# class SinglePathPPO(MultiSoftmaxPPO):
#     def __init__(self, n_inputs=32, n_outputs=4, output_channels=2,
#         entropy_penalty=0.0, adam_lr=None, adam_eps=None, clip_eps=None, cuda=True):
        
#         super(SinglePathPPO, self).__init__(output_channels=output_channels)
        
#         self.lstm1 = nn.LSTM(n_inputs, 64)
#         self.lstm2 = nn.LSTM(64, 64)
        
#         self.policy_fc = nn.Linear(64, output_channels)
#         self.value_fc = nn.Linear(64, 1)
        
#         if adam_lr and adam_eps:
#             self.opt = torch.optim.Adam(self.parameters(), lr=adam_lr, eps=adam_eps)
#             self.clip_eps = clip_eps
#             self.entropy_penalty = entropy_penalty
            
#             self._old = SinglePathPPO(n_inputs, n_outputs, cuda=cuda)
        
#         self._cuda = cuda
    
#     def forward(self, x):
#         x = x.unsqueeze(0).expand(n_outputs, x.shape[0], x.shape[1])
#         x = F.relu(self.lstm1(x)[0])
#         x = F.relu(self.lstm2(x)[0])
        
#         policy = self.policy_fc(x)
#         policy = policy.transpose(0, 1).contiguous().view(x.shape[0], -1)
        
#         value = self.value_fc(x)[-1]
        
#         return policy, value


