#!/bin/bash

# run.sh

# --
# Run Atari

mkdir -p logs
CUDA_VISIBLE_DEVICES=1 python atari-main.py  \
    --cuda \
    --env BreakoutNoFrameskip-v4 \
    --num-workers 8 \
    --steps-per-batch 256 \
    --log-dir /dev/null

CUDA_VISIBLE_DEVICES=0 python atari-main.py  \
    --cuda \
    --env BeamRiderNoFrameskip-v4 \
    --num-workers 8 \
    --steps-per-batch 256 \
    --log-dir ./logs-beam

# >>

CUDA_VISIBLE_DEVICES=1 python atari-main.py  \
    --cuda \
    --env BreakoutNoFrameskip-v4 \
    --num-workers 8 \
    --steps-per-batch 256 \
    --log-dir ./logs/tmp

# <<