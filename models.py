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

# --
# Value networks

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

# --
# Policy networks

class PolicyNetworkMixin(object):
    def backup(self):
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
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


class NormalPolicyMLP(nn.Module, PolicyNetworkMixin):
    
    def __init__(self, n_inputs, n_outputs, hidden_dim=64, adam_lr=None, adam_eps=None, clip_eps=None):
        super(NormalPolicyMLP, self).__init__()
        
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
            
            self._old = NormalPolicyMLP(n_inputs, n_outputs)
    
    def _forward(self, x):
        x = self.policy_fn(x)
        action_mean = self.action_mean(x)
        return action_mean, self.action_log_std.expand_as(action_mean)
    
    def sample_action(self, state):
        state = Variable(torch.from_numpy(state).unsqueeze(0))
        action_mean, action_log_std = self._forward(state)
        action = torch.normal(action_mean, torch.exp(action_log_std))
        return action.data.numpy().squeeze(axis=0)
    
    def log_prob(self, action, state):
        action_mean, action_log_std = self._forward(state)
        return (
            - 0.5 * (action - action_mean) ** 2 / (torch.exp(action_log_std) ** 2)
            - 0.5 * np.log(2 * np.pi)
            - action_log_std
        ).sum(1)


class CategoricalPolicyCNN(nn.Module, PolicyNetworkMixin):
    
    def __init__(self, n_inputs, n_outputs, input_shape, adam_lr=None, adam_eps=None, clip_eps=None):
        super(CategoricalPolicyCNN, self).__init__()
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_inputs, 32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )
        
        self.conv_output_dim = self._compute_sizes(n_inputs, input_shape)
        self.fc1 = nn.Linear(self.conv_output_dim, 512)
        self.fc2 = nn.Linear(512, n_outputs)
        
        if adam_lr and adam_eps:
            self.opt = torch.optim.Adam(self.parameters(), lr=adam_lr, eps=adam_eps)
            self.clip_eps = clip_eps
            
            self._old = CategoricalPolicyCNN(n_inputs, n_outputs)

    def _compute_sizes(self, n_inputs, input_shape):
        tmp = Variable(torch.zeros((1, n_inputs) + input_shape), volatile=True)
        tmp = conv_layers(tmp)
        return tmp.view(tmp.size(0), -1).size(-1)
    
    def _forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
    def sample_action(self, state):
        state = Variable(torch.from_numpy(state).unsqueeze(0))
        logits = self._forward(state)
        gumbel = (logits - torch.log(-torch.log(torch.rand(logits.size()))))
        return gumbel.max(1)[1]
    
    def log_prob(self, action, state):
        logits = self._forward(state)
        return F.log_softmax(torch.FloatTensor(logits))[state]

