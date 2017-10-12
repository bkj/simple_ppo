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
from torch.autograd import Variable

from rollouts import RolloutGenerator
from models import ValueNetwork, NormalPolicyNetwork

torch.set_default_tensor_type('torch.DoubleTensor')

# --
# Helpers

def set_seeds(seed):
    _ = env.seed(seed)
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed)
    _ = torch.cuda.manual_seed(seed)

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
    
    parser.add_argument('--advantage-gamma', type=float, default=0.99)
    parser.add_argument('--advantage-lambda', type=float, default=0.95)
    
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

value_net = ValueNetwork(
    n_inputs=env.observation_space.shape[0],
    adam_lr=args.adam_lr,
    adam_eps=args.adam_eps,
)

policy_net = NormalPolicyNetwork(
    n_inputs=env.observation_space.shape[0],
    n_outputs=env.action_space.shape[0],
    adam_lr=args.adam_lr,
    adam_eps=args.adam_eps,
    clip_eps=args.clip_eps,
)

roll_gen = RolloutGenerator(
    env=env,
    policy_net=policy_net,
    value_net=value_net, 
    steps_per_batch=args.steps_per_batch,
    rms=args.rms,
    advantage_gamma=args.advantage_gamma,
    advantage_lambda=args.advantage_lambda,
)

# --
# Run

set_seeds(args.seed)

start_time = time()
for batch_index in itertools.count(0):
    
    # --
    # Do rollouts
    
    if roll_gen.steps_so_far > args.total_steps:
        break
    
    roll_gen.next()
    
    # --
    # Logging
    
    for episode_index in range(len(roll_gen.batch)):
        print(json.dumps(OrderedDict([
            ("elapsed_time", time() - start_time),
            ("n_steps", roll_gen.steps_so_far),
            ("batch_index", batch_index),
            ("episode_index", episode_index),
            ("episode_length", len(roll_gen.batch[episode_index])),
            ("reward", sum([r['reward'] for r in roll_gen.batch[episode_index]])),
        ])))
    sys.stdout.flush()
    print(json.dumps(OrderedDict([
        ("elapsed_time", time() - start_time),
        ("n_steps", roll_gen.steps_so_far),
        ("batch_index", batch_index),
        ("n_episodes", roll_gen.n_episodes),
        ("avg_reward", roll_gen.total_reward / roll_gen.n_episodes),
    ])), file=sys.stderr)
    
    # --
    # Update model parameters
    
    policy_net.backup()
    
    for epoch in range(args.epochs_per_batch):
        minibatch_generator = roll_gen.iterate_batch(batch_size=args.batch_size, seed=(epoch, batch_index))
        for minibatch_idx, minibatch in enumerate(minibatch_generator):
            value_net.step(minibatch['states'], minibatch['value_targets'])
            policy_net.step(minibatch['states'], minibatch['actions'], minibatch['advantages'])
