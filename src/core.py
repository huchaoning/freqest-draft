import os
from math import *
import numpy as np
import scipy as sp
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
    'ApproxFisherInformation',

    'Simulator'
]


#################### 
#    Equipments    #
####################
class qCMOS:
    # The camera pixel size is 4.6 um per pixel.
    PIXEL_SIZE = 4.6 #um
    CONVERSION_FACTOR = 0.11
    OFFSET = 200
    QUANTUM_EFFICIENCY_770 = 0.5528 # @770nm

    @classmethod
    def quantum_efficiency(cls, wavelength):
        fx = sp.interpolate.interp1d(np.linspace(250, 1100, 8501), 
                                     np.load(os.path.join(os.path.dirname(__file__), 'quantum_efficiency.npy')))
        return fx(wavelength)


class DMD:
    PIXEL_SIZE = 19.374725804511403 #um

    TRIANGLE_SEQ = np.ravel((np.array([np.arange(-5, 6), np.arange(-5, 6)])).T)[::-1][1:-1]
    TRIANGLE_SEQ = np.concatenate([TRIANGLE_SEQ, TRIANGLE_SEQ[::-1]])


class SLM:
    PIXEL_SIZE = 8 #um
    RESOLUTION = (1920, 1080)


######################
#    Measurements    #
######################
class _Share:
    SIGMA = 103 #um

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
    X_AXIS = 86
    CENTER = 113
    
    ROI = {'X0': 1440, 'Y0': 876, 'W': 160, 'H': 228}

    def __init__(self, raw, amplitude):
        super().__init__(raw)
        (self.lower_bound, self.upper_bound), self.detectors = self.crop_bound(amplitude)


    def crop(self):
        self.cropped = self.raw[..., self.upper_bound:self.lower_bound, self.X_AXIS]


    @classmethod
    def crop_bound(cls, amplitude):
        lower_bound = int(np.ceil(cls.CENTER + 4*cls.SIGMA / qCMOS.PIXEL_SIZE))
        upper_bound = int(np.ceil(cls.CENTER - (2*amplitude + 4*cls.SIGMA) / qCMOS.PIXEL_SIZE))
        detectors = lower_bound - upper_bound
        return (lower_bound, upper_bound), detectors




######################
#      Main Cls      #
######################
@dataclass
class MetaData:
    measurement: str
    ground_truth: float
    amplitude: int
    pwm_duty: int
    timestamp: np.ndarray = None



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

    metadata: MetaData

    def savez(self, dirname):
        dirname = os.path.expanduser(dirname)
        
        m = self.metadata.measurement.lower()
        px = round(self.metadata.amplitude / DMD.PIXEL_SIZE * 2)
        f = self.metadata.ground_truth
        d = self.metadata.pwm_duty

        filename = os.path.join(dirname, f'{m}_{px}px_f{f}_d{d}.npz')

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



#####################
#     Simulator     #
#####################
class Simulator:
    def __init__(self, metadata: MetaData):
        self.meta = metadata


    def loc(self, n):
        return self.meta.amplitude * (1 + np.sign(np.sin(tau * self.meta.ground_truth * n + 0.001)))


    def gen(self, photons, sample_length=50):
        if self.meta.measurement.lower() == 'spade':
            _sig = SPADE.SIGMA

            p1 = lambda s: (s+2*_sig)**2*np.exp(-s**2/(4*_sig**2))/(8*_sig**2)
            p2 = lambda s: (s-2*_sig)**2*np.exp(-s**2/(4*_sig**2))/(8*_sig**2)

            data = [np.histogram(np.random.uniform(0, 1, photons), 
                    bins=[0, p1(self.loc(n)), p1(self.loc(n))+p2(self.loc(n))])[0] for n in range(sample_length)]

            return np.array(data).astype(float)
        

        if self.meta.measurement.lower() == 'di':
            def _gen_one(n):
                # Convert length units to camera pixel size to match experimental data.
                _loc = (self.loc(n) - self.meta.amplitude) / qCMOS.PIXEL_SIZE
                _sig = DI.SIGMA / qCMOS.PIXEL_SIZE

                (lower_bound, upper_bound), detectors = DI.crop_bound(self.meta.amplitude)

                return np.histogram(np.random.normal(detectors/2+_loc, _sig, photons), 
                                    bins=detectors, range=(upper_bound, lower_bound))[0]
                

            return np.array([_gen_one(n) for n in range(sample_length)]).astype(float)