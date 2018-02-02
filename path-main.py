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

from models import PathPPO
from rollouts import RolloutGenerator
from external.monitor import Monitor
from external.subproc_vec_env import SubprocVecEnv

from helpers import set_seeds

torch.set_default_tensor_type('torch.DoubleTensor') # Necessary?

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
    parser.add_argument('--adam-lr', type=float, default=7e-4)
    parser.add_argument('--entropy-penalty', type=float, default=0.01)
    
    parser.add_argument('--seed', type=int, default=123)
    
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--log-dir', type=str, default="./logs")
    
    return parser.parse_args()

# --
# Initialize

class SimpleEnv(object):
    def __init__(self, n_levers=4, seed=123):
        self._payouts = np.arange(n_levers)
        self._counter = 0
    
    def reset(self):
        return np.random.normal(0, 1, 32)
    
    def step(self, action):
        if self._counter % 5000 == 0:
            print('------------- reverse -------------')
            self._payouts = self._payouts[::-1]
        
        self._counter += 1
        payout = self._payouts[action.squeeze()].sum()
        is_done = self._counter % 10 == 0
        state = np.random.normal(0, 1, 32)
        return np.array([state]), np.array([payout]), np.array([is_done]), None


args = parse_args()

set_seeds(args.seed)

env = SubprocVecEnv([SimpleEnv for i in range(args.num_workers)])


ppo = PathPPO(
    n_outputs=4,
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
)

roll_gen.next()

# --
# Run

start_time = time()
while roll_gen.step_index < args.total_steps:
    
    # --
    # Sample a batch of rollouts
    
    roll_gen.next()
    
    print(np.bincount(roll_gen.batch['actions'].cpu().numpy().ravel(), minlength=4))
    
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
    
    sys.stdout.flush()
    
    # --
    # Update model parameters
    
    ppo.backup()
    for epoch in range(args.epochs_per_batch):
        for minibatch in roll_gen.iterate_batch(batch_size=args.batch_size * args.num_workers, seed=(epoch, roll_gen.step_index)):
            losses = ppo.step(**minibatch)
            # print(json.dumps(losses))
