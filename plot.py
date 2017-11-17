import os
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
    
    return np.array(df.timesteps), np.array(df.r.rolling(10).mean())

# --

x, y = load_data('./logs')
sel = x < 1.5e6
_ = plt.plot(x[sel], y[sel])

xlim = x.max()

x, y = load_data('/tmp/gym')
sel = x < 1.5e6
_ = plt.plot(x[sel], y[sel])

show_plot()