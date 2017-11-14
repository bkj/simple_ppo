#!/bin/bash

mkdir -p logs
CUDA_VISIBLE_DEVICES=1 python atari-main.py  \
    --cuda \
    --env BreakoutNoFrameskip-v4 \
    --num-workers 8 \
    --steps-per-batch 256