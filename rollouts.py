#!/usr/bin/env python

"""
    rollouts.py
"""

import numpy as np
from collections import defaultdict

import torch
from torch.autograd import Variable

def stack0(x):
    shp = x.shape
    return x.view((shp[0] * shp[1],) + tuple(shp[2:]) )

class RunningStats(object):
    def __init__(self, shape, clip=5.0, epsilon=1e-4):
        
        self._sum = np.zeros(shape)
        self._sumsq = np.zeros(shape)
        self._count = epsilon
        
        self.shape = shape
        self.clip = clip
        self.epsilon = epsilon
        
    def __call__(self, x, update=True):
        if update:
            self.update(x)
        
        x -= self.mean
        x /= self.std
        return np.clip(x, -self.clip, self.clip)
    
    def update(self, x):
        self._count += 1
        self._sum += x
        self._sumsq += (x ** 2)
    
    @property
    def mean(self):
        return self._sum / self._count
    
    @property
    def std(self):
        return np.sqrt(np.maximum((self._sumsq / self._count) - self.mean ** 2, self.epsilon))

# --
# Environment

class RolloutGenerator(object):
    """ Specialized for Atari """
    def __init__(self, env, ppo, steps_per_batch, advantage_gamma, advantage_lambda, 
        num_workers=1, num_frames=4, rms=True, cuda=False, mode='lever'):
        
        self.env = env
        self.ppo = ppo
        self.steps_per_batch = steps_per_batch
        self.advantage_gamma = advantage_gamma
        self.advantage_lambda = advantage_lambda
        
        self.cuda = cuda
        self.num_workers = num_workers
        self.num_frames = num_frames
        self.mode = mode
        
        self.batch = []
        self.step_index = 0
        self.batch_index = 0
        self.episode_index = 0
        
        self._rollgen = self._make_rollgen()
        
    def _update_current_state(self, current_state, state):
        current_state[:, :-1] = current_state[:, 1:]
        current_state[:, -1:] = state
        return current_state
    
    def _make_rollgen(self):
        
        # Stack of frames
        if self.mode == 'atari':
            current_state = np.zeros((self.num_workers, self.num_frames, 84, 84))
            state = self.env.reset()
            current_state = self._update_current_state(current_state, state)
        else:
            current_state = self.env.reset()
        
        episode_buffer = defaultdict(list)
        episode_buffer['states']   = np.zeros((self.steps_per_batch, *current_state.shape))
        episode_buffer['actions']  = np.zeros((self.steps_per_batch, self.num_workers, 1)).astype(int)
        episode_buffer['values']   = np.zeros((self.steps_per_batch, self.num_workers, 1))
        episode_buffer['is_dones'] = np.zeros((self.steps_per_batch, self.num_workers, 1)).astype(int)
        episode_buffer['rewards']  = np.zeros((self.steps_per_batch, self.num_workers, 1))
        
        while True:
            for step in range(self.steps_per_batch):
                action, value = self.ppo.sample_actions(current_state)
                
                next_state, reward, is_done, _ = self.env.step(action)
                
                episode_buffer['states'][step]   = current_state.copy()
                episode_buffer['actions'][step]  = action.copy().reshape(-1, 1)
                episode_buffer['values'][step]   = value.copy().reshape(-1, 1)
                episode_buffer['is_dones'][step] = is_done.copy().reshape(-1, 1)
                episode_buffer['rewards'][step]  = reward.copy().reshape(-1, 1)
                
                if self.mode == 'atari':
                    current_state *= (1 - is_done).reshape(-1, 1, 1, 1)
                    current_state = self._update_current_state(current_state, next_state)
                else:
                    current_state *= 1 - is_done
                
                self.step_index += self.num_workers
            
            yield episode_buffer, current_state
    
    def _compute_targets(self, states, actions, values, is_dones, rewards, current_state):
        """ compute targets for value function """
        
        next_advantage = 0
        advantages = torch.Tensor(states.size(0), states.size(1), 1)
        if self.cuda:
            advantages = advantages.cuda()
        
        _, next_value = self.ppo(Variable(current_state, volatile=True))
        next_value = next_value.data.view(-1, 1)
        
        for i in reversed(range(rewards.size(0))):
            nonterminal = (1 - is_dones[i]).double()
            delta = rewards[i] + self.advantage_gamma * next_value * nonterminal - values[i]
            advantages[i] = delta + self.advantage_gamma * self.advantage_lambda * next_advantage * nonterminal
            next_value = values[i].view(-1, 1)
            next_advantage = advantages[i].view(-1, 1)
        
        value_targets = advantages + values
        return value_targets, advantages
    
    def next(self):
        
        # Run simulations
        self.batch_index += 1
        self.batch, current_state = next(self._rollgen)
        
        # "Transpose" batch
        self.batch = {
            "states"   : torch.Tensor(self.batch['states']),
            "values"   : torch.Tensor(self.batch['values']),
            "actions"  : torch.LongTensor(self.batch['actions']),
            "is_dones" : torch.LongTensor(self.batch['is_dones']),
            "rewards"  : torch.Tensor(self.batch['rewards']),
            
            "current_state" : torch.Tensor(current_state),
        }
        
        if self.cuda:
            self.batch = dict(zip(self.batch.keys(), map(lambda x: x.cuda(), self.batch.values())))
        
        # Predict value
        self.batch['value_targets'], self.batch['advantages'] = self._compute_targets(**self.batch)
        
        # Normalize advantages
        self.batch['advantages'] = (self.batch['advantages'] - self.batch['advantages'].mean()) / (self.batch['advantages'].std() + 1e-5)
        
        self.batch = {
            "states" : stack0(self.batch['states']),
            
            "actions"       : self.batch['actions'].view(-1, 1),
            "is_dones"      : self.batch['is_dones'].view(-1, 1),
            "rewards"       : self.batch['rewards'].view(-1, 1),
            
            # "current_state" : self.batch['current_state'].view(-1, 1),
            "advantages"    : self.batch['advantages'].view(-1, 1),
            "value_targets" : self.batch['value_targets'].view(-1, 1),
        }
    
    def iterate_batch(self, batch_size=64, seed=0):
        idx = torch.LongTensor(np.random.RandomState(seed).permutation(self.n_steps))
        if self.cuda:
            idx = idx.cuda()
        
        for chunk in torch.chunk(idx, idx.size(0) // batch_size):
            yield {
                "states" : Variable(self.batch['states'][chunk]),
                "actions" : Variable(self.batch['actions'][chunk]),
                "advantages" : Variable(self.batch['advantages'][chunk]),
                "value_targets" : Variable(self.batch['value_targets'][chunk]),
            }
    
    @property
    def episodes_in_batch(self):
        return self.batch['is_dones'].sum()
        
    @property
    def total_reward(self):
        return self.batch['rewards'].sum()
    
    @property
    def n_steps(self):
        return self.batch['states'].size(0)
    