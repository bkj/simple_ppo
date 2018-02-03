#!/usr/bin/env python

"""
    path-main.py
"""

from __future__ import print_function

import os
import sys
import json
import argparse
import itertools
import numpy as np
from time import time
from collections import OrderedDict

import torch
from torch import nn
from torch.autograd import Variable

import gym
from gym.spaces.box import Box

from models import SinglePathPPO
from rollouts import RolloutGenerator
from external.monitor import Monitor
from external.subproc_vec_env import SubprocVecEnv

from helpers import set_seeds

# torch.set_default_tensor_type('torch.DoubleTensor') # Necessary?

from rsub import *
from matplotlib import pyplot as plt

# --
# Params

def parse_args():
    """ parameters mimic openai/baselines """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    
    parser.add_argument('--total-steps', type=int, default=int(40e6))
    parser.add_argument('--steps-per-batch', type=int, default=64)
    parser.add_argument('--epochs-per-batch', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--num-frames', type=int, default=4)
    
    parser.add_argument('--advantage-gamma', type=float, default=0.99)
    parser.add_argument('--advantage-lambda', type=float, default=0.95)
    
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--adam-eps', type=float, default=1e-5)
    parser.add_argument('--adam-lr', type=float, default=1e-2)
    parser.add_argument('--entropy-penalty', type=float, default=0.001)
    
    parser.add_argument('--seed', type=int, default=123)
    
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--log-dir', type=str, default="./logs")
    
    return parser.parse_args()

# --
# Initialize

# softmax
N = 6
class SimpleEnv(object):
    def __init__(self, n_levers=N, seed=123):
        self._payouts = np.arange(n_levers)
        self._counter = 0
        self.rs = np.random.RandomState(seed=seed)
        
        self._counts = np.zeros(n_levers)
    
    def reset(self):
        return self.rs.normal(0, 1, 32)
    
    def step(self, action):
        
        if (self._counter + 1) % 5000 == 0:
            print('reversal')
            self._payouts = self._payouts[::-1]
        
        self._counter += 1
        self._counts += (action == 1)
        
        payout = 0
        if (action == 1).sum() > 0:
            active = self._payouts[action == 1]
            payout = np.max(active) - action.sum()
        
        # payout = action
        is_done = self._counter % 10 == 0
        state = self.rs.normal(0, 1, 32)
        return np.hstack([state]), np.array([payout]), np.array([is_done]), None


def f(seed):
    def f_():
        return SimpleEnv(seed=seed)
    
    return f_

args = parse_args()

set_seeds(args.seed)

env = SubprocVecEnv([f(seed=i) for i in range(args.num_workers)])


ppo = SinglePathPPO(
    n_outputs=N,
    adam_lr=args.adam_lr,
    adam_eps=args.adam_eps,
    entropy_penalty=args.entropy_penalty,
    clip_eps=args.clip_eps,
    cuda=args.cuda,
)
print(ppo, file=sys.stderr)

if args.cuda:
    ppo = ppo.cuda()

roll_gen = RolloutGenerator(
    env=env,
    ppo=ppo,
    steps_per_batch=args.steps_per_batch,
    rms=False,
    advantage_gamma=args.advantage_gamma,
    advantage_lambda=args.advantage_lambda,
    cuda=args.cuda,
    num_workers=args.num_workers,
    num_frames=args.num_frames,
    action_dim=N,
    mode='lever'
)

# --
# Run

start_time = time()
all_action_counts = []
while roll_gen.step_index < args.total_steps:
    
    # --
    # Sample a batch of rollouts
    
    roll_gen.next()
    
    # action_counts = np.bincount(roll_gen.batch['actions'].cpu().numpy().ravel(), minlength=4)
    # print(roll_gen.batch['rewards'].cpu().numpy().mean())
    action_counts = roll_gen.batch['actions'].cpu().numpy().sum(axis=0)
    print(roll_gen.step_index, action_counts)
    # all_action_counts.append(action_counts)
    
    # if not roll_gen.step_index % 100:
    #     z = np.vstack(all_action_counts)
    #     for r in z.T:
    #         _ = plt.plot(r)
        
    #     show_plot()
    
    # --
    # Logging
    
    # print(json.dumps(OrderedDict([
    #     ("step_index",        roll_gen.step_index),
    #     ("batch_index",       roll_gen.batch_index),
    #     ("elapsed_time",      time() - start_time),
    #     ("episodes_in_batch", roll_gen.episodes_in_batch),
    #     ("total_reward",      roll_gen.total_reward),
    # ])))
    
    # for episode in roll_gen.batch:
    #     print(json.dumps(OrderedDict([
    #         ("step_index",     roll_gen.step_index),
    #         ("batch_index",    roll_gen.batch_index),
    #         ("elapsed_time",   time() - start_time),
    #     ])))
    
    # sys.stdout.flush()
    
    # --
    # Update model parameters
    
    ppo.backup()
    for epoch in range(args.epochs_per_batch):
        minibatches = roll_gen.iterate_batch(
            batch_size=args.batch_size * args.num_workers,
            seed=(epoch, roll_gen.step_index)
        )
        for minibatch in minibatches:
            losses = ppo.step(**minibatch)
            # print(json.dumps(losses))
