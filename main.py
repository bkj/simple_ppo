#!/usr/bin/env python

"""
    main.py
    
    # !! Add entropy penalty to loss
"""

import gym
import json
import argparse
import itertools
import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable

torch.set_default_tensor_type('torch.DoubleTensor')

# --
# Helpers

def copy_model(source, target):
    target.load_state_dict(source.state_dict())


def set_seeds(seed):
    _ = env.seed(seed)
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed)
    _ = torch.cuda.manual_seed(seed)


class RunMeanStd(object):
    def __init__(self, shape, clip=5.0):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
        
        self.clip = clip
        
    def __call__(self, x, update=True):
        if update:
            self.__push(x)
        
        x -= self.mean
        x /= self.std + 1e-8
        return np.clip(x, -self.clip, self.clip)
    
    def __push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            prev_m = self._M.copy()
            self._M[...] += (x - prev_m) / self._n
            self._S[...] += (x - prev_m) * (x - self._M)
    
    @property
    def mean(self):
        return self._M

    @property
    def std(self):
        if self._n > 1:
            return np.sqrt(self._S / (self._n - 1))
        else:
            return np.abs(self._M)

# --
# Environment

class RolloutGenerator(object):
    
    def __init__(self, env, policy_net, steps_per_batch, total_steps):
        
        self.env = env
        self.policy_net = policy_net
        self.steps_per_batch = steps_per_batch
        
        self.total_step = 0
        self.total_steps = total_steps
        
        self.running_state = RunMeanStd((policy_net.n_inputs,), clip=5.0)
    
    def _next(self):
        """ yield a batch of experiences """
        
        batch = []
        
        batch_steps = 0
        while batch_steps < self.steps_per_batch:
            state = self.running_state(env.reset())
            
            episode = []
            is_done = False
            while not is_done:
                action = policy_net.sample_action(state)
                
                next_state, reward, is_done, _ = env.step(action)
                next_state = self.running_state(next_state)
                episode.append({
                    "state" : state,
                    "action" : action,
                    "is_done" : is_done,
                    "reward" : reward,
                    "next_state" : next_state,
                })
                state = next_state
            
            batch.append(episode)
            batch_steps += len(episode)
        
        return batch, batch_steps
    
    def next(self):
        """ Iterate, until number of steps is exceeed """
        if self.total_step < self.total_steps:
            batch, batch_steps = self._next()
            self.total_step += batch_steps
            return batch
        else:
            return None

# --
# Networks

class ValueNetwork(nn.Module):
    
    def __init__(self, n_inputs, n_outputs=1, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        
        assert n_outputs == 1
        
        self.value_fn = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        self.fc1 = nn.Linear(hidden_dim, n_outputs)
        self.fc1.weight.data.mul_(0.1)
        self.fc1.bias.data.mul_(0.0)
    
    def forward(self, x):
        x = self.value_fn(x)
        return self.fc1(x).squeeze()


class NormalPolicyNetwork(nn.Module):
    
    def __init__(self, n_inputs, n_outputs, hidden_dim=64):
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
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        
        self.action_log_std = nn.Parameter(torch.zeros(1, n_outputs))
    
    def forward(self, x):
        x = self.policy_fn(x)
        action_mean = self.action_mean(x)
        return action_mean, self.action_log_std.expand_as(action_mean)
    
    def sample_action(self, state):
        state = Variable(torch.from_numpy(state).unsqueeze(0))
        action_mean, action_log_std = self(state)
        action = torch.normal(action_mean, torch.exp(action_log_std))
        return action.data.numpy().squeeze()
    
    def log_prob(self, action, state):
        action_mean, action_log_std = self(state)
        action_std = torch.exp(action_log_std)
        
        out = - 0.5 * (action - action_mean) ** 2 / (action_std ** 2)
        out -= 0.5 * np.log(2 * np.pi)
        out -= action_log_std
        
        return out.sum(1)


class TrainBatch(object):
    
    def __init__(self, batch):
        self.states      = torch.from_numpy(np.vstack([[e['state'] for e in episode] for episode in batch]))
        self.actions     = torch.from_numpy(np.vstack([[e['action'] for e in episode] for episode in batch]))
        self.is_dones    = torch.from_numpy(np.hstack([[e['is_done'] for e in episode] for episode in batch]).astype('int'))
        self.rewards     = torch.from_numpy(np.hstack([[e['reward'] for e in episode] for episode in batch]))
        self.next_states = torch.from_numpy(np.vstack([[e['next_state'] for e in episode] for episode in batch]))
        
        self.n_episodes = self.is_dones.sum()
        self.total_reward = self.rewards.sum()
        
    def compute_targets(self, value_net, gamma=0.99, tau=0.95):
        """ compute targets for value function """
        
        value_predictions = value_net(Variable(self.states))
        
        self.advantages = torch.Tensor(self.states.size(0))
        
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(self.rewards.size(0))):
            nonterminal = 1 - self.is_dones[i]
            delta = self.rewards[i] + gamma * prev_value * nonterminal - value_predictions.data[i]
            self.advantages[i] = delta + gamma * tau * prev_advantage * nonterminal
            prev_value = value_predictions.data[i]
            prev_advantage = self.advantages[i]
        
        self.value_targets = self.advantages + value_predictions.data
    
    def iterate(self, batch_size=64):
        if batch_size > 0:
            idx = torch.LongTensor(np.random.RandomState(0).permutation(self.states.size(0)))
            for chunk in torch.chunk(idx, 1):
                yield {
                    "states" : Variable(self.states[chunk]),
                    "actions" : Variable(self.actions[chunk]),
                    "advantages" : Variable(self.advantages[chunk]),
                    "value_targets" : Variable(self.value_targets[chunk]),
                }
        else:
            yield {
                "states" : Variable(self.states),
                "actions" : Variable(self.actions),
                "advantages" : Variable(self.advantages),
                "value_targets" : Variable(self.value_targets),
            }

# --
# Params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v1')
    
    parser.add_argument('--total-steps', type=int, default=int(1e6))
    parser.add_argument('--steps-per-batch', type=int, default=2048)
    parser.add_argument('--epochs-per-batch', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=-1)
    
    parser.add_argument('--clip-epsilon', type=float, default=0.2)
    
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

args = parse_args()


# --
# Initialize

args = parse_args()

env = gym.make(args.env)
set_seeds(args.seed)

value_net = ValueNetwork(n_inputs=env.observation_space.shape[0])

policy_net, old_policy_net = [NormalPolicyNetwork(
    n_inputs=env.observation_space.shape[0],
    n_outputs=env.action_space.shape[0]
) for _ in range(2)]
copy_model(policy_net, old_policy_net)

opt_policy = torch.optim.Adam(policy_net.parameters(), lr=0.001)
opt_value = torch.optim.Adam(value_net.parameters(), lr=0.001)

# --
# Run

set_seeds(args.seed)

rollout_generator = RolloutGenerator(env, policy_net, args.steps_per_batch, args.total_steps)

for batch_index in itertools.count(0):
    
    # Compute rollouts
    batch = rollout_generator.next()
    if not batch:
        break
    
    train_batch = TrainBatch(batch)
    train_batch.compute_targets(value_net)
    
    for _ in range(args.epochs_per_batch):
        minibatch_generator = train_batch.iterate(batch_size=args.batch_size)
        for minibatch_idx, minibatch in enumerate(minibatch_generator):
            
            # Update value function
            opt_value.zero_grad()
            value_predictions = value_net(minibatch['states']).squeeze()
            value_loss = ((value_predictions - minibatch['value_targets']) ** 2).mean()
            value_loss.backward()
            opt_value.step()
            
            # Update policy
            opt_policy.zero_grad()
            
            log_prob = policy_net.log_prob(minibatch['actions'], minibatch['states'])
            old_log_prob = old_policy_net.log_prob(minibatch['actions'], minibatch['states'])
            ratio = torch.exp(log_prob - old_log_prob)
            
            if minibatch_idx == 0:
                copy_model(policy_net, old_policy_net)
            
            advantages_normed = (minibatch['advantages'] - minibatch['advantages'].mean()) / minibatch['advantages'].std()
            surr1 = ratio * advantages_normed
            surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * advantages_normed
            policy_surr = -torch.min(surr1, surr2).mean()
            policy_surr.backward()
            torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
            opt_policy.step()
    
    # Logging
    print {
        "batch_index" : batch_index,
        "n_episodes" : train_batch.n_episodes,
        "avg_reward" : train_batch.total_reward / train_batch.n_episodes
    }