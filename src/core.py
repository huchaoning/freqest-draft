import os
from math import *
import numpy as np
from dataclasses import dataclass

from .estimator import freq_estimator, td_estimator


__all__ = [
    'qCMOS',
    'DMD',

    'SPADE',
    'DI',

    'MetaData',
    'Estimates',
    'LoadEstimates',
    'FrequencyEstimation',

    'FisherInformation',
    'ApproxFisherInformation'
]


#################### 
#    Equipments    #
####################
class qCMOS:
    # The camera pixel size is 4.6 um per pixel.
    PIXEL_SIZE = 4.6 #um
    CONVERSION_FACTOR = 0.11
    OFFSET = 200
    QUANTUM_EFFICIENCY = 0.5528



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
        self.noise = (temp[..., :5,   :5].mean((-1, -2)) + temp[..., -5:,   :5].mean((-1, -2))  + 
                      temp[..., :5, -5: ].mean((-1, -2)) + temp[..., -5:, -5: ].mean((-1, -2))) / 4
        del raw, temp

    def est_all(self):
        self.crop()
        self.pn = ((self.cropped - qCMOS.OFFSET).sum(-1)) * qCMOS.CONVERSION_FACTOR
        self.w = self.noise / self.cropped.mean(-1)
        self.td = td_estimator(self.__class__.__name__, self.cropped)
        self.lse = np.array([freq_estimator(sample) for sample in self.td])




class SPADE(_Share):
    X_AXIS = 89
    POINT_1 = 406
    POINT_2 = 116

    ROI = {'X0': 2128, 'Y0': 720, 'W': 180, 'H': 500}

    def crop(self):
        self.cropped = self.raw[..., (self.POINT_1, self.POINT_2), self.X_AXIS]



class DI(_Share):
    SIGMA = 103 #um

    X_AXIS = 86
    CENTER = 113
    
    ROI = {'X0': 1440, 'Y0': 876, 'W': 160, 'H': 228}

    def __init__(self, raw, amplitude):
        super().__init__(raw)
        self.amplitude = amplitude
        self.upper_bound = int(np.ceil(self.CENTER - (2*amplitude + 4*self.SIGMA) / qCMOS.PIXEL_SIZE))
        self.lower_bound = int(np.ceil(self.CENTER + 4*self.SIGMA / qCMOS.PIXEL_SIZE))
        self.detectors = self.lower_bound - self.upper_bound

    def crop(self):
        self.cropped = self.raw[..., self.upper_bound:self.lower_bound, self.X_AXIS]




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
class Estimates:
    '''
        NOTE: Only `photons` is in unit of photon number, other data related to photon count are in unit of Adu.

        EXAMPLE: If you want to know the photon number of the noise, you needs to 
        ```
        (Estimates.noise - qCMOS.OFFSET) * qCMOS.CONVERSION_FACTOR
        ```

        The `noise_weight` is just for MLE needs.
    '''

    frequency_estimates: np.ndarray

    cropped_data: np.ndarray
    time_domain: np.ndarray
    photons: np.ndarray

    noise: np.ndarray
    noise_weight: np.ndarray

    metadata: MetaData = None

    def savez(self, filename):
        filename = os.path.expanduser(filename)

        if os.path.exists(filename): 
            raise FileExistsError(f'{filename} already exists')

        np.savez_compressed(filename, **self.__dict__)

    @classmethod
    def load(cls, file):
        return LoadEstimates(file)



def LoadEstimates(file):
    file = os.path.expanduser(file)
    npz = np.load(file, allow_pickle=True)
    dic = {}
    for k in npz.files:
        dic[k] = npz[k]
        if k.lower() == 'metadata':
            dic[k] = npz[k].item()
    return Estimates(**dic)



class FrequencyEstimation:
    @classmethod
    def FromRaw(cls, raw: np.ndarray, metadata: MetaData):
        if metadata.measurement.upper() == 'SPADE':
            expt = SPADE(raw)
        elif metadata.measurement.upper() == 'DI':
            expt = DI(raw, metadata.amplitude)
        else:
            raise ValueError
        
        del raw
        expt.est_all()
        return Estimates(frequency_estimates = expt.lse,

                         cropped_data = expt.cropped, 
                         time_domain = expt.td, 
                         photons = expt.pn, 

                         noise = expt.noise,
                         noise_weight = expt.w,
                    
                         metadata = metadata)
    
    @classmethod
    def FromEstimates(cls, Estimates_instance: Estimates):
        c: Estimates = np.copy(Estimates_instance).item()

        c.photons = (c.cropped_data - qCMOS.OFFSET).sum(-1) * qCMOS.CONVERSION_FACTOR
        c.noise_weight = c.noise / (c.cropped_data - qCMOS.OFFSET).mean(-1) * qCMOS.CONVERSION_FACTOR

        c.time_domain = td_estimator(c.metadata.measurement, c.cropped_data, c.noise_weight)
        c.frequency_estimates = np.array([freq_estimator(sample) for sample in c.time_domain])

        return c



######################
#     FI and CRB     #
######################
def FisherInformation(A_list: np.ndarray, freq_list: np.ndarray, sigma=DI.SIGMA, N=50):
    results_1, results_2 = [], []
    n = np.arange(N)
    for A in A_list:
        for f in freq_list:
            _temp = ((tau*n) * np.sin(tau*f*n))**2
            results_1.append((A/sigma)**2 * _temp.sum())
        results_2.append(results_1)
    return np.array(results_2)

def ApproxFisherInformation(A_list: np.ndarray, sigma=DI.SIGMA, N=50):
    n = np.arange(N)
    results = [2*(A*pi/sigma)**2 * (n**2).sum() for A in A_list]
    return np.array(results)

