#!/usr/bin/env python

"""
    main.py
    
    !! May want to take a look at how `RunningStats` are computed
    !! Also, not sure about gradient clipping
    !! Also, should look into initializations
    !! Also, should implement clipping + learning rate annealing
"""

from __future__ import print_function

import sys
import gym
import json
import argparse
import itertools
import numpy as np
from time import time
from collections import OrderedDict

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
    
    def __init__(self, env, policy_net, steps_per_batch, rms=True):
        
        self.env = env
        self.policy_net = policy_net
        self.steps_per_batch = steps_per_batch
        
        self.steps_so_far = 0
        
        self.rms = rms
        if rms:
            self.running_stats = RunningStats((policy_net.n_inputs,), clip=5.0)
    
    def next(self):
        """ yield a batch of experiences """
        
        batch = []
        
        batch_steps = 0
        while batch_steps < self.steps_per_batch:
            state = env.reset()
            if self.rms:
                state = self.running_stats(state)
            
            episode = []
            is_done = False
            while not is_done:
                action = policy_net.sample_action(state)
                
                next_state, reward, is_done, _ = env.step(action)
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
        self.fc1.weight.data.mul_(0.1) # !!
        self.fc1.bias.data.mul_(0.0) # !!
    
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
        self.action_mean.weight.data.mul_(0.1) # !!
        self.action_mean.bias.data.mul_(0.0) # !!
        
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
        return (
            - 0.5 * (action - action_mean) ** 2 / (torch.exp(action_log_std) ** 2)
            - 0.5 * np.log(2 * np.pi)
            - action_log_std
        ).sum(1)


class TrainBatch(object):
    
    def __init__(self, batch):
        self.states = torch.from_numpy(np.vstack([[e['state'] for e in episode] for episode in batch]))
        self.actions = torch.from_numpy(np.vstack([[e['action'] for e in episode] for episode in batch]))
        self.is_dones = torch.from_numpy(np.hstack([[e['is_done'] for e in episode] for episode in batch]).astype('int'))
        self.rewards = torch.from_numpy(np.hstack([[e['reward'] for e in episode] for episode in batch]))
        
        self.n_episodes = self.is_dones.sum()
        self.total_reward = self.rewards.sum()
        
    def compute_targets(self, value_net, advantage_gamma, advantage_lambda):
        """ compute targets for value function """
        
        value_predictions = value_net(Variable(self.states))
        
        self.advantages = torch.Tensor(self.states.size(0))
        
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(self.rewards.size(0))):
            nonterminal = 1 - self.is_dones[i]
            delta = self.rewards[i] + advantage_gamma * prev_value * nonterminal - value_predictions.data[i]
            self.advantages[i] = delta + advantage_gamma * advantage_lambda * prev_advantage * nonterminal
            prev_value = value_predictions.data[i]
            prev_advantage = self.advantages[i]
        
        self.value_targets = self.advantages + value_predictions.data
    
    def iterate(self, batch_size=64, seed=0):
        if batch_size > 0:
            idx = torch.LongTensor(np.random.RandomState(seed).permutation(self.states.size(0)))
            for chunk in torch.chunk(idx, idx.size(0) // batch_size):
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
    """ parameters mimic openai/baselines """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v1')
    
    parser.add_argument('--total-steps', type=int, default=int(1e6))
    parser.add_argument('--steps-per-batch', type=int, default=2048)
    parser.add_argument('--epochs-per-batch', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    
    parser.add_argument("--advantage-gamma", type=float, default=0.99)
    parser.add_argument("--advantage-lambda", type=float, default=0.95)
    
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--adam-eps', type=float, default=1e-5)
    parser.add_argument('--adam-lr', type=float, default=3e-4)
    
    parser.add_argument('--seed', type=int, default=123)
    
    parser.add_argument('--rms', action="store_true")
    return parser.parse_args()



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

opt_policy = torch.optim.Adam(policy_net.parameters(), lr=args.adam_lr, eps=args.adam_eps)
opt_value = torch.optim.Adam(value_net.parameters(), lr=args.adam_lr, eps=args.adam_eps)

# --
# Run

set_seeds(args.seed)

start_time = time()
rollout_generator = RolloutGenerator(env, policy_net, steps_per_batch=args.steps_per_batch, rms=args.rms)
for batch_index in itertools.count(0):
    
    # --
    # Simulation
    
    if rollout_generator.steps_so_far > args.total_steps:
        break
    
    batch = rollout_generator.next()
    
    train_batch = TrainBatch(batch)
    train_batch.compute_targets(
        value_net, 
        advantage_gamma=args.advantage_gamma,
        advantage_lambda=args.advantage_lambda
    )
    
    # --
    # Logging
    
    for episode_index in range(len(batch)):
        print(json.dumps(OrderedDict([
            ("elapsed_time", time() - start_time),
            ("n_steps", rollout_generator.steps_so_far),
            ("batch_index", batch_index),
            ("episode_index", episode_index),
            ("episode_length", len(batch[episode_index])),
            ("reward", sum([r['reward'] for r in batch[episode_index]])),
        ])))
    sys.stdout.flush()
    print(json.dumps(OrderedDict([
        ("elapsed_time", time() - start_time),
        ("n_steps", rollout_generator.steps_so_far),
        ("batch_index", batch_index),
        ("n_episodes", train_batch.n_episodes),
        ("avg_reward", train_batch.total_reward / train_batch.n_episodes),
    ])), file=sys.stderr)
    
    # --
    # Learning
    
    copy_model(policy_net, old_policy_net)
    
    for epoch in range(args.epochs_per_batch):
        minibatch_generator = train_batch.iterate(batch_size=args.batch_size, seed=(epoch, batch_index))
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
            
            advantages_normed = (minibatch['advantages'] - minibatch['advantages'].mean()) / minibatch['advantages'].std()
            surr1 = ratio * advantages_normed
            surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * advantages_normed
            policy_surr = -torch.min(surr1, surr2).mean()
            
            policy_surr.backward()
            torch.nn.utils.clip_grad_norm(policy_net.parameters(), 40)
            opt_policy.step()
    
