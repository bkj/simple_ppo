#!/usr/bin/env python

"""
    atari-main.py
    
    Todo:
        !! Same as in `mlp-main.py`
        
        !! Also need to add entropy penaly
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
from models import AtariPPO

from external.atari_wrappers import make_atari, wrap_deepmind
from external.subproc_vec_env import SubprocVecEnv

torch.set_default_tensor_type('torch.DoubleTensor')

# --
# Helpers

def set_seeds(subenvs, seed):
    _ = [subenv.seed(seed + 100 * i) for i,subenv in enumerate(subenvs)]
    _ = np.random.seed(seed)
    _ = torch.manual_seed(seed)
    _ = torch.cuda.manual_seed(seed)

# --
# Params

def parse_args():
    """ parameters mimic openai/baselines """
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    
    parser.add_argument('--total-steps', type=int, default=int(40e6))
    parser.add_argument('--steps-per-batch', type=int, default=256)
    parser.add_argument('--epochs-per-batch', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    
    parser.add_argument('--advantage-gamma', type=float, default=0.99)
    parser.add_argument('--advantage-lambda', type=float, default=0.95)
    
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--adam-eps', type=float, default=1e-5)
    parser.add_argument('--adam-lr', type=float, default=1e-3)
    
    parser.add_argument('--seed', type=int, default=123)
    
    parser.add_argument('--rms', action="store_true")
    
    parser.add_argument('--cuda', action="store_true")
    
    return parser.parse_args()

# --
# Initialize

args = parse_args()

def make_env(env_id, seed, rank):
    def _thunk():
        env = make_atari(env_id)
        env.seed(seed + rank)
        env = wrap_deepmind(env)
        return env
    
    return _thunk

print('envs')
# env = SubprocVecEnv([
#     make_env(args.env, args.seed, i) for i in range(2)
# ])
env = make_env(args.env, args.seed, 0)()
# <<

print('ppo')
ppo = AtariPPO(
    input_channels=env.observation_space.shape[2],
    input_height=env.observation_space.shape[0],
    input_width=env.observation_space.shape[1],
    n_outputs=env.action_space.n,
    adam_lr=args.adam_lr,
    adam_eps=args.adam_eps,
    clip_eps=args.clip_eps,
    cuda=args.cuda
)
print(ppo)

if args.cuda:
    ppo = ppo.cuda()

roll_gen = RolloutGenerator(
    env=env,
    ppo=ppo,
    steps_per_batch=args.steps_per_batch,
    rms=args.rms,
    advantage_gamma=args.advantage_gamma,
    advantage_lambda=args.advantage_lambda,
    cuda=args.cuda
)

# --
# Run

start_time = time()
while roll_gen.step_index < args.total_steps:
    
    # --
    # Sample a batch of rollouts
    print('roll_gen.next()')
    roll_gen.next()
    
    # --
    # Logging
    
    print(json.dumps(OrderedDict([
        ("step_index",        roll_gen.step_index),
        ("batch_index",       roll_gen.batch_index),
        ("episode_index",     roll_gen.episode_index),
        ("elapsed_time",      time() - start_time),
        ("episodes_in_batch", roll_gen.episodes_in_batch),
        ("avg_reward",        roll_gen.total_reward / roll_gen.episodes_in_batch),
    ])), file=sys.stderr)
    
    for episode in roll_gen.batch:
        print(json.dumps(OrderedDict([
            ("step_index",     episode[0]['step_index']),
            ("batch_index",    episode[0]['batch_index']),
            ("episode_index",  episode[0]['episode_index']),
            ("elapsed_time",   time() - start_time),
            ("episode_length", len(episode)),
            ("reward",         sum([r['reward'] for r in episode])),
        ])))
    
    sys.stdout.flush()
    
    # --
    # Update model parameters
    
    ppo.backup()
    for epoch in range(args.epochs_per_batch):
        for minibatch in roll_gen.iterate_batch(batch_size=args.batch_size, seed=(epoch, roll_gen.step_index)):
            ppo.step(**minibatch)

