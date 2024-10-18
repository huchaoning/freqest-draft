from math import *
import numpy as np
import os

from copy import deepcopy as cp
from datetime import datetime

from .core import *
from ..cache import cache

date = datetime.now().strftime('%Y%m%d')

empty_dic = {
    'spade': [],
    'di': []
}

@cache
def load(files, lengths, mirrors=None, save=f'~/Desktop/{date}'):
    if set([key.lower() for key in files.keys()]) != set(['spade', 'di']):
        raise ValueError('files must be a dict and files.keys() must be spade and di')

    raw, photons, noise = [cp(empty_dic) for _ in range(3)]
    
    for k, v in files.items():
        raw[k], photons[k], noise[k] = DataReader(k).load(v, lengths, mirrors) 

    if save is not False:
        save = os.path.expanduser(save)
        np.savez_compressed(save+'_raw.npz', **raw)
        np.savez_compressed(save+'_photons.npz', **photons)
        np.savez_compressed(save+'_noise.npz', **noise)
        print(f'.npz files are saved in {os.path.dirname(os.path.abspath(save))}')

    return raw, photons, noise
