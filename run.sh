#!/bin/bash

mkdir -p results/refactor

python mlp-main.py --total-steps 1000000 --env Reacher-v1 > ./results/refactor/Reacher-constant-no_rms
python mlp-main.py --total-steps 1000000 --env Reacher-v1 --rms > ./results/refactor/Reacher-constant-rms

python mlp-main.py --total-steps 1000000 --env Hopper-v1 > ./results/refactor/Hopper-constant-no_rms
python mlp-main.py --total-steps 1000000 --env Hopper-v1 --rms > ./results/refactor/Hopper-constant-rms

python mlp-main.py --total-steps 1000000 --env HalfCheetah-v1 > ./results/refactor/HalfCheetah-constant-no_rms
python mlp-main.py --total-steps 1000000 --env HalfCheetah-v1 --rms > ./results/refactor/HalfCheetah-constant-rms

python mlp-main.py --total-steps 1000000 --env InvertedPendulum-v1 > ./results/refactor/InvertedPendulum-constant-no_rms
python mlp-main.py --total-steps 1000000 --env InvertedPendulum-v1 --rms > ./results/refactor/InvertedPendulum-constant-rms
