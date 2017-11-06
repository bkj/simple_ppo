#!/usr/bin/env python

"""
    helpers.py
"""

import numpy as np

from torch.autograd import Variable

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    
    if isinstance(x, Variable):
        return to_numpy(x.data)
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()

