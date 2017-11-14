#!/usr/bin/env python

"""
    rollouts.py
"""

import numpy as np
from collections import defaultdict

import torch
from torch.autograd import Variable

# def summary(x):
#     print(x.size())
#     print("no norm sum", x.cpu().numpy().squeeze().sum())
#     xx = (x - x.mean()) / (x.std() + 1e-5)
#     print("norm abs sum", (xx).abs().cpu().numpy().squeeze().sum())
#     print("norm sq sum", (xx ** 2).cpu().numpy().squeeze().sum())
#     print("norm sum", (xx).cpu().numpy().squeeze().sum())


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
            self.running_stats = RunningStats(ppo.input_shape, clip=5.0)
        
        self.cuda = cuda
        self.batch = []
        self.step_index = 0
        self.batch_index = 0
        self.episode_index = 0
        
        self._rollgen = self._make_rollgen()
    
    def _make_rollgen(self):
        """ yield a batch of experiences """
        
        episode_buffer = defaultdict(list)
        
        num_workers = 1
        steps_per_batch = 64
        
        current_state = np.zeros((num_workers, 4, 84, 84))
        
        def update_current_state(state):
            current_state[:, :-1] = current_state[:, 1:]
            current_state[:, -1:] = state
        
        state = self.env.reset()
        
        # >>
        # !! ATARI
        state = state.astype(np.float64)
        update_current_state(state)
        # << 
        
        if self.rms:
            state = self.running_stats(state)
        
        counter = 0
        while True:
            counter += 1
            
            action = self.ppo.sample_action(current_state)
            next_state, reward, is_done, _ = self.env.step(action)
            # >>
            # !! ATARI
            next_state = next_state.astype(np.float64)
            # << 
            
            if self.rms:
                next_state = self.running_stats(next_state)
            
            for i in range(next_state.shape[0]):
                episode_buffer[i].append({
                    "state"      : current_state[i].copy(),
                    "action"     : action[i],
                    "is_done"    : is_done[i],
                    "reward"     : reward[i],
                    "step_index" : self.step_index,
                })
                
                if is_done[i]:
                    current_state[i] *= 0
            
            if counter > steps_per_batch:
                for k,v in episode_buffer.items():
                    yield v
                    episode_buffer[k] = [episode_buffer[k][-1]]
                    
                counter = 1
            
            self.step_index += next_state.shape[0]
            update_current_state(next_state)
            print(current_state.shape)
    
    def _compute_targets(self, states, actions, is_dones, rewards):
        """ compute targets for value function """
        value_predictions = self.ppo.predict_value(Variable(states))
        
        advantages = torch.Tensor(states.size(0))
        
        if self.cuda:
            value_predictions = value_predictions.cuda()
            advantages = advantages.cuda()
        
        prev_advantage = 0
        prev_value = value_predictions[-1].data.cpu().numpy()[0]
        
        for i in reversed(range(rewards.size(0) - 1)):
            nonterminal = 1 - is_dones[i]
            delta = rewards[i] + self.advantage_gamma * prev_value * nonterminal - value_predictions.data[i]
            advantages[i] = delta + self.advantage_gamma * self.advantage_lambda * prev_advantage * nonterminal
            prev_value = value_predictions.data[i]
            prev_advantage = advantages[i]
        
        value_targets = advantages + value_predictions.data
        return value_targets, advantages.view(-1, 1)
    
    def next(self):
        
        # Run simulations
        self.batch = []
        self.batch_index += 1
        steps_in_batch = 0
        while steps_in_batch < self.steps_per_batch:
            episode = next(self._rollgen)
            print('episode', len(episode))
            steps_in_batch += len(episode)
            self.batch.append(episode)
        
        # "Transpose" batch
        self.tbatch = {
            "states"   : torch.from_numpy(np.vstack([[e['state'] for e in episode] for episode in self.batch])),
            "actions"  : torch.from_numpy(np.vstack([[e['action'] for e in episode] for episode in self.batch])),
            "is_dones" : torch.from_numpy(np.hstack([[e['is_done'] for e in episode] for episode in self.batch]).astype('int')),
            "rewards"  : torch.from_numpy(np.hstack([[e['reward'] for e in episode] for episode in self.batch])),
        }
        
        print("self.tbatch['states'].size()", self.tbatch['states'].size())
        
        if self.cuda:
            self.tbatch = dict(zip(self.tbatch.keys(), map(lambda x: x.cuda(), self.tbatch.values())))
        
        # Predict value
        self.tbatch['value_targets'], self.tbatch['advantages'] = self._compute_targets(**self.tbatch)
        
        for k in self.tbatch.keys():
            self.tbatch[k] = self.tbatch[k][:-1]
        
        self.tbatch['advantages'] = (self.tbatch['advantages'] - self.tbatch['advantages'].mean()) / (self.tbatch['advantages'].std() + 1e-5)
        
    
    def iterate_batch(self, batch_size=64, seed=0):
        if batch_size > 0:
            idx = np.random.RandomState(seed).permutation(self.n_steps)
            idx = np.sort(idx)
            idx = torch.LongTensor(idx)
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