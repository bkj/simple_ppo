#!/usr/bin/env python

"""
    rollouts.py
"""

import numpy as np

import torch
from torch.autograd import Variable

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
    
    def __init__(self, env, ppo, steps_per_batch, advantage_gamma, advantage_lambda, rms=True, cuda=False):
        
        self.env = env
        self.ppo = ppo
        self.steps_per_batch = steps_per_batch
        self.advantage_gamma = advantage_gamma
        self.advantage_lambda = advantage_lambda
        
        self.rms = rms
        if rms:
            self.running_stats = RunningStats(ppo.policy.input_shape, clip=5.0)
        
        self.cuda = cuda
        self.step_index = 0
    
    def _do_rollout(self):
        """ yield a batch of experiences """
        
        state = self.env.reset()
        # >>
        # !! ATARI
        state = state.astype('float') / 255 - 0.5
        if len(state.shape) == 3:
            state = np.expand_dims(state, 0)
        # << 
        if self.rms:
            state = self.running_stats(state)
        
        batch = {}
        is_done = np.array([False])
        while (len(batch) < self.steps_per_batch) and (not is_done.any()):
            
            action = self.ppo.sample_action(state)
            next_state, reward, is_done, _ = self.env.step(action)
            # >>
            # !! ATARI
            next_state = state.astype('float') / 255 - 0.5
            if len(next_state.shape) == 3:
                next_state = np.expand_dims(next_state, 0)
            # << 
            if self.rms:
                next_state = self.running_stats(next_state)
            
            for i in range(next_state.shape[0]):
                batch[i].append({
                    "state"   : state[i],
                    "action"  : action[i],
                    "is_done" : is_done[i],
                    "reward"  : reward[i],
                })
            state = next_state
            
            self.step_index += next_state.shape[0]
        
        return batch
    
    def _compute_targets(self, states, actions, is_dones, rewards):
        """ compute targets for value function """
        value_predictions = self.ppo.predict_value(Variable(states))
        advantages = torch.Tensor(states.size(0))
        
        if self.cuda:
            value_predictions = value_predictions.cuda()
            advantages = advantages.cuda()
        
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            nonterminal = 1 - is_dones[i]
            delta = rewards[i] + self.advantage_gamma * prev_value * nonterminal - value_predictions.data[i]
            advantages[i] = delta + self.advantage_gamma * self.advantage_lambda * prev_advantage * nonterminal
            prev_value = value_predictions.data[i]
            prev_advantage = advantages[i]
        
        value_targets = advantages + value_predictions.data
        
        return value_targets, advantages
    
    def next(self):
        # Run simulations
        self.batch = self._do_rollout()
        
        # "Transpose" batch
        self.tbatch = {
            "states"   : torch.from_numpy(np.array([b['state'] for b in self.batch])),
            "actions"  : torch.from_numpy(np.array([b['action'] for b in self.batch])),
            "is_dones" : torch.from_numpy(np.array([b['is_done'] for b in self.batch]).astype('int')),
            "rewards"  : torch.from_numpy(np.array([b['reward'] for b in self.batch])),
        }
        
        for k in self.tbatch.keys():
            print(k, self.tbatch[k].size())
        
        if self.cuda:
            self.tbatch = dict(zip(self.tbatch.keys(), map(lambda x: x.cuda(), self.tbatch.values())))
        
        # Predict value
        self.tbatch['value_targets'], self.tbatch['advantages'] = self._compute_targets(**self.tbatch)
    
    def iterate_batch(self, batch_size=64, seed=0):
        if batch_size > 0:
            idx = torch.LongTensor(np.random.RandomState(seed).permutation(self.n_steps))
            if self.cuda:
                idx = idx.cuda()
            
            for chunk in torch.chunk(idx, idx.size(0) // batch_size):
                yield {
                    "states" : Variable(self.tbatch['states'][chunk]),
                    "actions" : Variable(self.tbatch['actions'][chunk]),
                    "advantages" : Variable(self.tbatch['advantages'][chunk]),
                    "value_targets" : Variable(self.tbatch['value_targets'][chunk]),
                }
        else:
            yield {
                "states" : Variable(self.tbatch['states']),
                "actions" : Variable(self.tbatch['actions']),
                "advantages" : Variable(self.tbatch['advantages']),
                "value_targets" : Variable(self.tbatch['value_targets']),
            }
    
    @property
    def episodes_in_batch(self):
        return self.tbatch['is_dones'].sum()
    
    @property
    def total_reward(self):
        return self.tbatch['rewards'].sum()
    
    @property
    def n_steps(self):
        return self.tbatch['states'].size(0)