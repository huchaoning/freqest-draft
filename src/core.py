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
    'Estmates',
    'LoadEstmates',
    'FrequencyEstmation',

    'FIM_CRB'
]


#################### 
#    Equipments    #
####################
class qCMOS:
    # The camera pixel size is 4.6 um per pixel.
    PIXEL_SIZE = 4.6 #um
    CONVERSION_FACTOR = 0.107
    OFFSET = 200



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

        temp = self.raw[..., :-4, :]
        noise = (temp[..., :5,   :5].mean((-1, -2)) + temp[..., -5:,   :5].mean((-1, -2))  + 
                 temp[..., :5, -5: ].mean((-1, -2)) + temp[..., -5:, -5: ].mean((-1, -2))) / 4
        self.noise = (noise - qCMOS.OFFSET) * qCMOS.CONVERSION_FACTOR
        del raw, temp, noise

    def est_pn(self):
        self.pn = (self.cropped - qCMOS.OFFSET).sum(-1) * qCMOS.CONVERSION_FACTOR

    def est_lse(self):
        self.lse = np.array([freq_estimator(sample) for sample in self.td])

    def est_w(self):
        self.w = self.noise / (self.cropped - qCMOS.OFFSET).mean(-1) * qCMOS.CONVERSION_FACTOR

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
class Estmates:
    frequency_estmates: np.ndarray
    phase_estmates: np.ndarray

    cropped_data: np.ndarray
    time_domain: np.ndarray
    photons: np.ndarray

    noise: np.ndarray
    noise_weight: np.ndarray

    metadata: MetaData = None

    def savez(self, filename):
        filename = os.path.expanduser(filename)

        if os.path.exists(filename): 
            override = input(f'File {filename} already exists, override? [y/N]')
            if not override.lower() in ('yes', 'y'):
                return

        np.savez_compressed(filename, **self.__dict__)



def LoadEstmates(file):
    file = os.path.expanduser(file)
    npz = np.load(file, allow_pickle=True)
    dic = {}
    for k in npz.files:
        dic[k] = npz[k]
        if k.lower() == 'metadata':
            dic[k] = npz[k].item()
    return Estmates(**dic)



def FrequencyEstmation(raw: np.ndarray, measurement: str, metadata: MetaData = None):
    if measurement.upper() == 'SPADE':
        expt = SPADE(raw)
    elif measurement.upper() == 'DI':
        expt = DI(raw)
    else:
        raise ValueError
    
    expt.est_all()
    return Estmates(expt.lse[..., 0], 
                    expt.lse[..., 1], 

                    expt.cropped, 
                    expt.td, 
                    expt.pn, 

                    expt.noise,
                    expt.w,
                
                    metadata)



######################
#     FI and CRB     #
######################
class FIM_CRB:
    def __init__(self, 
                 N: int = 50,
                 phi: float = 0, 
                 sigma: float = 103,

                 multi: bool = True, 
                 approx: bool = False,

                 waveform: str = 'sin'):
        
        self.N = N
        self.phi = phi
        self.sigma = sigma

        self.multi = multi
        self.approx = approx

        if waveform.lower() in ('sin', 'cos'):
            self.waveform = waveform
        else:
            raise ValueError('Waveform must be sin or cos.')
        
        
    def fim(self, A, f):
        n = tau * np.arange(self.N)
        if self.approx:
            fi11 = (n**2).sum() / 2
            fi12 = (n**1).sum() / 2
            fi22 = (n**0).sum() / 2
        elif self.waveform.lower() == 'sin':
            fi11 = (n**2 * np.cos(f*n + self.phi)**2).sum()
            fi12 = (n**1 * np.cos(f*n + self.phi)**2).sum()
            fi22 = (n**0 * np.cos(f*n + self.phi)**2).sum()
        elif self.waveform.lower() == 'cos':
            fi11 = (n**2 * np.sin(f*n + self.phi)**2).sum()
            fi12 = (n**1 * np.sin(f*n + self.phi)**2).sum()
            fi22 = (n**0 * np.sin(f*n + self.phi)**2).sum()
        else:
            raise ValueError

        if self.multi:
            return (A/self.sigma)**2 * np.array([[fi11, fi12], [fi12, fi22]])
        elif not self.multi:
            return (A/self.sigma)**2 * fi11
        else:
            raise ValueError
        

    def crb(self, A, f):
        matrix = self.fim(A, f)
        if self.multi:
            return np.linalg.inv(matrix)[0, 0]
        elif not self.multi:
            return 1 / matrix
        else:
            raise ValueError


    def crb_list(self, A, f_list):
        return np.array([self.crb(A, f) for f in f_list])

