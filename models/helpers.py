#!/usr/bin/env python

"""
    helpers.py
"""

import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.autograd import Variable

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    
    if isinstance(x, Variable):
        return to_numpy(x.data)
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class BackupMixin(object):
    def backup(self):
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if '_old.' in k:
                del state_dict[k]
        
        self._old.load_state_dict(state_dict)

