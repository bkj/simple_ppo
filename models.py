#!/usr/bin/env python

"""
    models.py
"""

import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable

torch.set_default_tensor_type('torch.DoubleTensor')

class ValueNetwork(nn.Module):
    
    def __init__(self, n_inputs, n_outputs=1, hidden_dim=64, adam_lr=None, adam_eps=None):
        super(ValueNetwork, self).__init__()
        
        assert n_outputs == 1
        
        self.value_fn = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.fc1 = nn.Linear(hidden_dim, n_outputs)
        self.fc1.weight.data.mul_(0.1) # !!
        self.fc1.bias.data.mul_(0.0) # !!
        
        if adam_lr and adam_eps:
            self.opt = torch.optim.Adam(self.parameters(), lr=adam_lr, eps=adam_eps)
        
    def forward(self, x):
        x = self.value_fn(x)
        return self.fc1(x).squeeze()
    
    def step(self, states, value_targets):
        self.opt.zero_grad()
        
        value_predictions = self(states).squeeze()
        value_loss = ((value_predictions - value_targets) ** 2).mean()
        value_loss.backward()
        
        self.opt.step()


class NormalPolicyNetwork(nn.Module):
    
    def __init__(self, n_inputs, n_outputs, hidden_dim=64, adam_lr=None, adam_eps=None, clip_eps=None):
        super(NormalPolicyNetwork, self).__init__()
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        self.policy_fn = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.action_mean = nn.Linear(hidden_dim, n_outputs)
        self.action_mean.weight.data.mul_(0.1) # !!
        self.action_mean.bias.data.mul_(0.0) # !!
        
        self.action_log_std = nn.Parameter(torch.zeros(1, n_outputs))
        
        if adam_lr and adam_eps:
            self.opt = torch.optim.Adam(self.parameters(), lr=adam_lr, eps=adam_eps)
            self.clip_eps = clip_eps
            
            self._old = NormalPolicyNetwork(n_inputs, n_outputs)
    
    def forward(self, x):
        x = self.policy_fn(x)
        action_mean = self.action_mean(x)
        return action_mean, self.action_log_std.expand_as(action_mean)
    
    def sample_action(self, state):
        state = Variable(torch.from_numpy(state).unsqueeze(0))
        action_mean, action_log_std = self(state)
        action = torch.normal(action_mean, torch.exp(action_log_std))
        return action.data.numpy().squeeze(axis=0)
    
    def log_prob(self, action, state):
        action_mean, action_log_std = self(state)
        return (
            - 0.5 * (action - action_mean) ** 2 / (torch.exp(action_log_std) ** 2)
            - 0.5 * np.log(2 * np.pi)
            - action_log_std
        ).sum(1)
    
    def backup(self):
        state_dict = self.state_dict()
        for k in state_dict.keys():
            if '_old.' in k:
                del state_dict[k]
        
        self._old.load_state_dict(state_dict)
    
    def step(self, states, actions, advantages):
        self.opt.zero_grad()
        
        log_prob = self.log_prob(actions, states)
        old_log_prob = self._old.log_prob(actions, states)
        ratio = torch.exp(log_prob - old_log_prob)
        
        advantages_normed = (advantages - advantages.mean()) / advantages.std()
        surr1 = ratio * advantages_normed
        surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_normed
        policy_surr = -torch.min(surr1, surr2).mean()
        
        policy_surr.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 40)
        
        self.opt.step()

