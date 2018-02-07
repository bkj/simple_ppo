#!/usr/bin/env python

"""
    models/continuous.py
"""

import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable

from helpers import to_numpy, BackupMixin

# --
# MLP Value + Policy networks (for mujoco)

class PPOWrapper(object):
    
    def __init__(self, value, policy):
        self.value = value
        self.policy = policy
    
    def predict_value(self, *args, **kwargs):
        return self.value.predict_value(*args, **kwargs)
    
    def backup(self):
        self.policy.backup()
    
    def sample_action(self, *args, **kwargs):
        return self.policy.sample_action(*args, **kwargs)
    
    def log_prob(self, *args, **kwargs):
        return self.policy.log_prob(*args, **kwargs)
    
    def step(self, states, actions, value_targets, advantages):
        self.value.step(states, value_targets)
        self.policy.step(states, actions, advantages)


class ValueMLP(nn.Module):
    
    def __init__(self, n_inputs, n_outputs=1, hidden_dim=64, adam_lr=None, adam_eps=None):
        super(ValueMLP, self).__init__()
        
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
        
    def predict_value(self, x):
        x = self.value_fn(x)
        return self.fc1(x).squeeze()
    
    def step(self, states, value_targets):
        self.opt.zero_grad()
        
        value_predictions = self.predict_value(states).squeeze()
        value_loss = ((value_predictions - value_targets) ** 2).mean()
        value_loss.backward()
        
        self.opt.step()


class NormalPolicyMLP(nn.Module, BackupMixin):
    
    def __init__(self, n_inputs, n_outputs, hidden_dim=64, adam_lr=None, adam_eps=None, clip_eps=None):
        super(NormalPolicyMLP, self).__init__()
        
        self.input_shape = (n_inputs,)
        
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
            
            self._old = NormalPolicyMLP(n_inputs, n_outputs)
    
    def _forward(self, x):
        x = self.policy_fn(x)
        action_mean = self.action_mean(x)
        return action_mean, self.action_log_std.expand_as(action_mean)
    
    def sample_action(self, state):
        state = Variable(torch.from_numpy(state).unsqueeze(0))
        action_mean, action_log_std = self._forward(state)
        action = torch.normal(action_mean, torch.exp(action_log_std))
        return to_numpy(action).squeeze(axis=0)
    
    def log_prob(self, action, state):
        action_mean, action_log_std = self._forward(state)
        return (
            - 0.5 * (action - action_mean) ** 2 / (torch.exp(action_log_std) ** 2)
            - 0.5 * np.log(2 * np.pi)
            - action_log_std
        ).sum(1)
    
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

