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
    
    def __init__(self, env, policy_net, value_net, steps_per_batch, advantage_gamma, advantage_lambda, rms=True):
        
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        
        self.steps_per_batch = steps_per_batch
        self.advantage_gamma = advantage_gamma
        self.advantage_lambda = advantage_lambda
        
        self.steps_so_far = 0
        
        self.rms = rms
        if rms:
            self.running_stats = RunningStats((policy_net.n_inputs,), clip=5.0)
    
    def _do_rollout(self):
        """ yield a batch of experiences """
        
        batch = []
        
        batch_steps = 0
        while batch_steps < self.steps_per_batch:
            state = self.env.reset()
            if self.rms:
                state = self.running_stats(state)
            
            episode = []
            is_done = False
            while not is_done:
                action = self.policy_net.sample_action(state)
                
                next_state, reward, is_done, _ = self.env.step(action)
                if self.rms:
                    next_state = self.running_stats(next_state)
                
                episode.append({
                    "state" : state,
                    "action" : action,
                    "is_done" : is_done,
                    "reward" : reward
                })
                state = next_state
            
            batch.append(episode)
            batch_steps += len(episode)
        
        self.steps_so_far += batch_steps
        
        return batch
    
    def _compute_targets(self, states, actions, is_dones, rewards):
        """ compute targets for value function """
        
        value_predictions = self.value_net(Variable(states))
        
        advantages = torch.Tensor(states.size(0))
        
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
            "states"   : torch.from_numpy(np.vstack([[e['state'] for e in episode] for episode in self.batch])),
            "actions"  : torch.from_numpy(np.vstack([[e['action'] for e in episode] for episode in self.batch])),
            "is_dones" : torch.from_numpy(np.hstack([[e['is_done'] for e in episode] for episode in self.batch]).astype('int')),
            "rewards"  : torch.from_numpy(np.hstack([[e['reward'] for e in episode] for episode in self.batch])),
        }
        
        # Predict value
        self.tbatch['value_targets'], self.tbatch['advantages'] = self._compute_targets(**self.tbatch)
    
    def iterate_batch(self, batch_size=64, seed=0):
        if batch_size > 0:
            idx = torch.LongTensor(np.random.RandomState(seed).permutation(self.n_steps))
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
    def n_episodes(self):
        return self.tbatch['is_dones'].sum()
    
    @property
    def total_reward(self):
        return self.tbatch['rewards'].sum()
    
    @property
    def n_steps(self):
        return self.tbatch['states'].size(0)