#!/bin/bash

python main.py --total-steps 500000 --env HalfCheetah-v1 > ./results/hc-constant-500k-no_rms
python main.py --total-steps 500000 --env HalfCheetah-v1 --rms > ./results/hc-constant-500k-rms

