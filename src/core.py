import os
from math import *
import numpy as np
from dataclasses import dataclass

from .estimator import freq_estimator


__all__ = [
    'qCMOS',
    'DMD',

    'SPADE',
    'DI',

    'MetaData',
    'FrequencyEstmation'
]


#################### 
#    Equipments    #
####################
class qCMOS:
    # The camera pixel size is 4.6 um per pixel.
    PIXEL_SIZE = 4.6 #um
    CONVERSION_FACTOR = 0.107



class DMD:
    PIXEL_SIZE = 19.374725804511403 #um

    TRIANGLE_SEQ = np.ravel((np.array([np.arange(-5, 6), np.arange(-5, 6)])).T)[::-1][1:-1]
    TRIANGLE_SEQ = np.concatenate([TRIANGLE_SEQ, TRIANGLE_SEQ[::-1]])



######################
#    Measurements    #
######################
class _Share:
    def __init__(self, raw):
        self.raw = raw.astype(float)

    def est_pn(self):
        self.pn = (self.cropped.sum(-1) - 400) * qCMOS.CONVERSION_FACTOR

    def est_lse(self):
        self.lse = np.array([freq_estimator(sample) for sample in self.td])

    def est_all(self):
        self.crop()
        self.est_td()
        self.est_lse()
        self.est_pn()



class SPADE(_Share):
    X_AXIS = 89
    POINT_1 = 406
    POINT_2 = 116

    ROI = {'X0': 2128, 'Y0': 720, 'W': 180, 'H': 500}

    def crop(self):
        self.cropped = self.raw[..., (self.POINT_1, self.POINT_2), self.X_AXIS]


    def est_td(self):
        self.td = self.cropped[..., 1] - self.cropped[..., 0]


class DI(_Share):
    pass




######################
#      Main Cls      #
######################
@dataclass
class MetaData:
    measurement: str
    ground_truth: float
    amplitude: int
    timestamp: np.ndarray



@dataclass
class FrequencyEstmation:
    raw: np.ndarray
    metadata: MetaData

    def run(self):
        if self.metadata.measurement.upper() == 'SPADE':
            expt = SPADE(self.raw)
        elif self.metadata.measurement.upper() == 'DI':
            expt = DI(self.raw)
        else:
            raise ValueError
        
        expt.est_all()
        del self.raw # Delete raw data to save memory

        self.cropped_data = expt.cropped
        self.time_domain = expt.td
        self.frequency_estmates = expt.lse[..., 0]
        self.phase_estmates = expt.lse[..., 1]
        self.photons = expt.pn


    def savez(self, filename):
        filename = os.path.expanduser(filename)
        if not os.path.exists(filename): 
            np.savez_compressed(filename, **self.__dict__)
        else:
            print(f'ERROR: File {filename} already exists.')
            



