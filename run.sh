#!/bin/bash

python main.py --total-steps 1000000 --env Reacher-v1 > ./results-ref/Reacher-constant-no_rms
python main.py --total-steps 1000000 --env Reacher-v1 --rms > ./results-ref/Reacher-constant-rms

python main.py --total-steps 1000000 --env Hopper-v1 > ./results-ref/Hopper-constant-no_rms
python main.py --total-steps 1000000 --env Hopper-v1 --rms > ./results-ref/Hopper-constant-rms

python main.py --total-steps 1000000 --env HalfCheetah-v1 > ./results-ref/HalfCheetah-constant-no_rms
python main.py --total-steps 1000000 --env HalfCheetah-v1 --rms > ./results-ref/HalfCheetah-constant-rms

python main.py --total-steps 1000000 --env InvertedPendulum-v1 > ./results-ref/InvertedPendulum-constant-no_rms
python main.py --total-steps 1000000 --env InvertedPendulum-v1 --rms > ./results-ref/InvertedPendulum-constant-rms
