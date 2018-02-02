import os
import argparse

import numpy as np
import pandas as pd
from glob import glob

from rsub import *
from matplotlib import pyplot as plt

def load_data(indir='./logs'):
    infiles = glob(os.path.join(indir, '*.monitor.csv'))
    df = pd.concat([pd.read_csv(f, skiprows=1) for f in infiles], axis=0)
    df = df.sort_values('t').reset_index(drop=True)
    
    df['timesteps'] = df.l.cumsum()
    df['timesteps'] = np.concatenate([[0], df.timesteps])[:-1]
    
    return np.array(df.timesteps), np.array(df.r.rolling(100).mean())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=-1)
    return parser.parse_args()

# --

args = parse_args()

N = args.n if args.n > 0 else float('inf')

for p in ['logs', '/tmp/gym']:
# for p in ['logs/space']:
    x, y = load_data(p)
    sel = x < N
    _ = plt.plot(x[sel], y[sel], alpha=0.5)

show_plot()