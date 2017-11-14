import os
import numpy as np
import pandas as pd
from glob import glob

from rsub import *
from matplotlib import pyplot as plt

def smooth_reward_curve(x, y):
    k      = min(31, int(np.ceil(len(x) / 30)))
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def load_data(indir='./logs'):
    infiles = glob(os.path.join(indir, '*.monitor.csv'))
    df = pd.concat([pd.read_csv(f, skiprows=1) for f in infiles], axis=0)
    df = df.sort_values('t').reset_index(drop=True)
    df['timesteps'] = df.l.cumsum()
    df['timesteps'] = np.concatenate([[0], df.timesteps])[:-1]
    
    # return np.array(df.timesteps), np.array(df.r.rolling(10).mean())
    return smooth_reward_curve(
        np.array(df.timesteps),
        np.array(df.r),
    )


x, y = load_data('./logs')

_ = plt.plot(x, y)
# _ = plt.xlim(0, 200000)
show_plot()