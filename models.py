#!/usr/bin/env python

"""
    models.py
"""

import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable

# torch.set_default_tensor_type('torch.DoubleTensor')

from helpers import to_numpy

# --
# Helpers

class BackupMixin(object):
    def backup(self):
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if '_old.' in k:
                del state_dict[k]
        
        self._old.load_state_dict(state_dict)

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


# --
# CNN joint network (for atari)

class JointPPO(nn.Module, BackupMixin):
    def step(self, states, actions, value_targets, advantages):
        
        value_predictions, log_prob, dist_entropy = self.evaluate_actions(states, actions)
        _, old_log_prob, _ = self._old.evaluate_actions(states, actions)
        
        ratio = torch.exp(log_prob - old_log_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = ((value_targets - value_predictions) ** 2).mean()
        
        self.opt.zero_grad()
        loss = (value_loss + policy_loss - dist_entropy * self.entropy_penalty)
        loss.backward()
        self.opt.step()
        return {
            "value_loss" : float(value_loss.data[0]),
            "policy_loss" : float(policy_loss.data[0]),
            "dist_entropy" : float(dist_entropy.data[0]),
        }


class SoftmaxPPO(JointPPO):
    def sample_actions(self, states):
        # print('--- sample_actions ---')
        states = Variable(torch.FloatTensor(states))
        if self._cuda:
            states = states.cuda()
        
        policy, value_predictions = self(states)
        
        # Sample action
        probs = F.softmax(policy, dim=1)
        action = probs.multinomial()
        # >>
        action_ = to_numpy(action).squeeze()
        action = np.zeros(policy.shape).astype(int)
        action[(np.arange(action_.shape[0]), action_)] = 1
        # <<
        
        return to_numpy(action), to_numpy(value_predictions)
    
    def evaluate_actions(self, states, actions):
        # print('--- evaluate_actions ---')
        policy, value_predictions = self(states)
        
        # Compute log prob of actions
        log_probs = F.log_softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions.max(dim=-1)[1].view(-1, 1))
        
        # Compute entropy
        probs = F.softmax(policy, dim=1)
        dist_entropy = -(log_probs * probs).sum(dim=-1).mean()
        
        return value_predictions, action_log_probs, dist_entropy


class MultiSoftmaxPPO(JointPPO):
    def sample_actions(self, states):
        states = Variable(torch.FloatTensor(states))
        if self._cuda:
            states = states.cuda()
        
        policy, value_predictions = self(states)
        
        # Sample action
        probs = F.softmax(policy.view(-1, 2), dim=1)
        action = probs.multinomial()
        action = action.view(policy.shape[0], -1)
        
        return to_numpy(action), to_numpy(value_predictions)
    
    def evaluate_actions(self, states, actions):
        # print('--- evaluate_actions ---')
        policy, value_predictions = self(states)
        
        # Compute log prob of actions
        log_probs = F.log_softmax(policy.view(-1, 2), dim=1)
        action_log_probs = log_probs.gather(1, actions.view(-1, 1))
        action_log_probs = action_log_probs.view(policy.shape[0], -1)
        
        # Compute entropy
        probs = F.softmax(policy.view(-1, 2), dim=1)
        dist_entropy = -(log_probs * probs).view(policy.shape[0], -1).sum(dim=-1).mean()
        
        return value_predictions, action_log_probs, dist_entropy


# class AtariPPO(SoftmaxPPO):
    
#     def __init__(self, input_channels, input_height, input_width, n_outputs, 
#         entropy_penalty=0.0, adam_lr=None, adam_eps=None, clip_eps=None, cuda=True):
        
#         super(AtariPPO, self).__init__()
        
#         self.input_shape = (input_channels, input_height, input_width)
        
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(input_channels, 32, kernel_size=(8, 8), stride=(4, 4)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1)),
#             nn.ReLU(),
#         )
        
#         self.conv_output_dim = self._compute_sizes(input_channels, input_height, input_width)
#         self.fc = nn.Linear(self.conv_output_dim, 512)
        
#         self.policy_fc = nn.Linear(512, n_outputs)
#         self.value_fc = nn.Linear(512, 1)
        
#         if adam_lr and adam_eps:
#             self.opt = torch.optim.Adam(self.parameters(), lr=adam_lr, eps=adam_eps)
#             self.clip_eps = clip_eps
#             self.entropy_penalty = entropy_penalty
            
#             self._old = AtariPPO(input_channels, input_height, input_width, n_outputs, cuda=cuda)
        
#         self._cuda = cuda
    
#     def _compute_sizes(self, input_channels, input_height, input_width):
#         tmp = Variable(torch.zeros((1, input_channels, input_height, input_width)), volatile=True)
#         tmp = self.conv_layers(tmp)
#         return tmp.view(tmp.size(0), -1).size(-1)
    
#     def forward(self, x):
#         x = x / 255.0
#         x = self.conv_layers(x)
#         x = x.view(x.size(0), -1)
#         x = F.relu(self.fc(x))
#         return self.policy_fc(x), self.value_fc(x)

# --
# Path PPO

class SinglePathPPO(MultiSoftmaxPPO):
    def __init__(self, n_inputs=32, n_outputs=4, 
        entropy_penalty=0.0, adam_lr=None, adam_eps=None, clip_eps=None, cuda=True):
        
        super(SinglePathPPO, self).__init__()
        
        self.trunk = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        self.policy_fc = nn.Linear(64, n_outputs * 2)
        self.value_fc = nn.Linear(64, 1)
        
        if adam_lr and adam_eps:
            self.opt = torch.optim.Adam(self.parameters(), lr=adam_lr, eps=adam_eps)
            self.clip_eps = clip_eps
            self.entropy_penalty = entropy_penalty
            
            self._old = SinglePathPPO(n_inputs, n_outputs, cuda=cuda)
        
        self._cuda = cuda
    
    def forward(self, x):
        x = self.trunk(x)
        return self.policy_fc(x), self.value_fc(x)


