#!/bin/bash

python main.py --total-steps 1000000 --env HalfCheetah-v1 > ./results/HalfCheetah-constant-no_rms
python main.py --total-steps 1000000 --env HalfCheetah-v1 --rms > ./results/HalfCheetah-constant-rms

python main.py --total-steps 1000000 --env Hopper-v1 > ./results/Hopper-constant-no_rms
python main.py --total-steps 1000000 --env Hopper-v1 --rms > ./results/Hopper-constant-rms

python main.py --total-steps 1000000 --env Reacher-v1 > ./results/Reacher-constant-no_rms
python main.py --total-steps 1000000 --env Reacher-v1 --rms > ./results/Reacher-constant-rms

python main.py --total-steps 1000000 --env InvertedPendulum-v1 > ./results/InvertedPendulum-constant-no_rms
python main.py --total-steps 1000000 --env InvertedPendulum-v1 --rms > ./results/InvertedPendulum-constant-rms
