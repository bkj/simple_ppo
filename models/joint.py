#!/usr/bin/env python

"""
    models/joint.py
"""

import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable

from helpers import to_numpy, BackupMixin

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
    def __init__(self, output_channels=None, **kwargs):
        super(MultiSoftmaxPPO, self).__init__(**kwargs)
        self.output_channels = output_channels
        
    def sample_actions(self, states):
        states = Variable(torch.FloatTensor(states))
        if self._cuda:
            states = states.cuda()
        
        policy, value_predictions = self(states)
        
        # Sample action
        probs = F.softmax(policy.view(-1, self.output_channels), dim=1)
        action = probs.multinomial()
        action = action.view(policy.shape[0], -1)
        
        return to_numpy(action), to_numpy(value_predictions)
    
    def evaluate_actions(self, states, actions):
        # print('--- evaluate_actions ---')
        policy, value_predictions = self(states)
        
        # Compute log prob of actions
        log_probs = F.log_softmax(policy.view(-1, self.output_channels), dim=1)
        action_log_probs = log_probs.gather(1, actions.view(-1, 1))
        action_log_probs = action_log_probs.view(policy.shape[0], -1)
        
        # Compute entropy
        probs = F.softmax(policy.view(-1, self.output_channels), dim=1)
        dist_entropy = -(log_probs * probs).view(policy.shape[0], -1).sum(dim=-1).mean()
        
        return value_predictions, action_log_probs, dist_entropy