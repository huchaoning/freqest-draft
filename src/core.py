import os
from math import tau, pi
import numpy as np
import scipy as sp
import gc

from dataclasses import dataclass

from .estimator import *


__all__ = [
    'qCMOS',
    'DMD',
    'SLM',

    'SPADE',
    'DI',

    'MetaData',
    'Estimates',
    'LoadEstimates',
    'Estimation',

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
    
    @classmethod
    def convert2photons(cls, img, smoothing=1e-10):
        # If the ADU value is less than the qCMOS offset, means the signal here is zero.
        # 1e-10 for smoothing
        photons = (img - cls.OFFSET) * cls.CONVERSION_FACTOR
        photons = np.clip(photons, smoothing, np.inf)
        return photons


class DMD:
    PIXEL_SIZE = 19.374725804511403 #um

    TRIANGLE_SEQ = np.ravel((np.array([np.arange(-5, 6), np.arange(-5, 6)])).T)[::-1][1:-1]
    TRIANGLE_SEQ = np.concatenate([TRIANGLE_SEQ, TRIANGLE_SEQ[::-1]])


class SLM:
    PIXEL_SIZE = 8 #um
    RESOLUTION = (1920, 1080)




######################
#      Data Cls      #
######################
class _Repr:
    def __repr__(self):
        return '\n'.join(f'{attribute}: {value}' for attribute, value in self.__dict__.items())
    
@dataclass
class MetaData(_Repr):
    '''
    Data class to store metadata for measurements.

    Parameters:
        measurement (str): The type of measurement, 'SPADE' or 'DI'.
        ground_truth (float): The ground truth value, frequency (frames/s) or velocity (um/s).
        estimating (str): Deciding to estimate frequency or velocity. 

        amplitude (float, optional): The amplitude value. Keep it None if estimating velocity.
        pwm_duty (int, optional): The PWM duty cycle. Defaults to 0.
        methods (str, optional): Estimation algorithm. Keep it None, and the program will decide automatically.
        timestamp (np.ndarray, optional): Keep it None if simulating.
    '''
    measurement: str
    ground_truth: float
    estimating: str

    amplitude: float = None
    pwm_duty: int = 0
    methods: str = None
    timestamp: np.ndarray = None

    def __post_init__(self):
        if self.methods is None:
            if self.estimating.lower() == 'frequency':
                if self.measurement.lower() == 'di':
                    self.methods = ('mle', 'lse') if self.pwm_duty == 0 else ('lse', 'lse')
                elif self.measurement.lower() == 'spade':
                    self.methods = ('sub', 'lse') if self.pwm_duty == 0 else ('sub', 'lse')
                else:
                    raise ValueError('measurement must be DI or SPADE')
            
            elif self.estimating.lower() == 'velocity':
                if self.measurement.lower() == 'di':
                    self.methods = ('mle', 'lse') if self.pwm_duty == 0 else ('lse', 'lse')
                elif self.measurement.lower() == 'spade':
                    self.methods = ('zhou2023', 'lse') if self.pwm_duty == 0 else ('zhou2023', 'lse')
                else:
                    raise ValueError('measurement must be DI or SPADE')
            
            else:
                raise ValueError('estimating must be frequency or velocity')

    def __repr__(self):
        return super().__repr__()


@dataclass
class Estimates(_Repr):
    '''
        NOTE: Only `photons` is in unit of photon number, other data related to photon count are in unit of ADU.

        EXAMPLE: If you want to know the photon number of the noise, you needs to 
        ```
        (Estimates.noise - qCMOS.OFFSET) * qCMOS.CONVERSION_FACTOR
        ```
        or
        ```
        qCMOS.convert2photons(Estimates.noise)
        ```
    '''
    cropped_data: np.ndarray
    metadata: MetaData
    background: np.ndarray

    estimates_a: np.ndarray
    estimates_b: np.ndarray = None

    time_domain: np.ndarray = None
    photons: np.ndarray = None


    def savez(self, dirname):
        dirname = os.path.expanduser(dirname)
        truth = self.metadata.ground_truth

        m = self.metadata.measurement.lower()
        d = self.metadata.pwm_duty

        if self.metadata.estimating.lower() == 'frequency':
            px = round(self.metadata.amplitude / DMD.PIXEL_SIZE * 2)
            filename = os.path.join(dirname, f'{m}_{px}px_f{truth}_d{d}.npz')
        elif self.metadata.estimating.lower() == 'velocity':
            filename = os.path.join(dirname, f'{m}_v{truth}_d{d}.npz')

        if os.path.exists(filename): 
            raise FileExistsError(f'{filename} already exists')

        np.savez_compressed(filename, **self.__dict__)

    @classmethod
    def load(cls, file):
        return LoadEstimates(file)
    
    def __repr__(self):
        return super().__repr__()



def LoadEstimates(file):
    file = os.path.expanduser(file)
    npz = np.load(file, allow_pickle=True)
    dic = {}
    for k in npz.files:
        dic[k] = npz[k]
        if k.lower() == 'metadata':
            dic[k] = npz[k].item()
    return Estimates(**dic)




######################
#    Measurements    #
######################
class _Share:
    SIGMA = 103 #um
    def __init__(self, raw=None, cropped=None, background=None, metadata: MetaData=None):
        self.meta = metadata
        if raw is not None:
            self.raw = raw.astype(float)

            temp = self.raw[..., :-4, :]
            self.background = (temp[..., :5,   :5].mean((-1, -2)) + temp[..., -5:,   :5].mean((-1, -2))  + 
                          temp[..., :5, -5: ].mean((-1, -2)) + temp[..., -5:, -5: ].mean((-1, -2))) / 4
            del raw, temp
            self.crop()

        elif (cropped is not None) and (background is not None):
            self.cropped = cropped
            self.background = background
        
        else:
            raise ValueError('When raw is not given, cropped and background must be given. When raw is given, cropped and background will be ignored.')


    def est_all(self):
        if self.meta.estimating.lower() == 'velocity':
            self.td = td_estimator(self.__class__.__name__, self.cropped, self.background, self.meta.methods[0], standardize=False)
            _result = np.array([velocity_estimator(sample) for sample in self.td])
            self.theta_a, self.theta_a = _result[:, 0], _result[:, 1]

        elif self.meta.estimating.lower() == 'frequency':
            self.td = td_estimator(self.__class__.__name__, self.cropped, self.background, self.meta.methods[0], standardize=False)
            self.theta_a = np.array([freq_estimator(sample) for sample in self.td])
            self.theta_b = None

        self.estimates = Estimates( cropped_data = self.cropped, 
                                    metadata = self.meta,
           
                                    estimates_a = self.theta_a,
                                    estimates_b = self.theta_b,

                                    time_domain = self.td, 
                                    photons = qCMOS.convert2photons(self.cropped, 0), 

                                    background = self.background)



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

    def __init__(self, *args, **kwargs):
        meta: MetaData = kwargs['metadata']
        if not meta.estimating.lower() == 'frequency':
            self.lower_bound = int(np.ceil(self.CENTER + 4*self.SIGMA / qCMOS.PIXEL_SIZE))
            self.upper_bound = int(np.ceil(self.CENTER - (2*meta.amplitude + 4*self.SIGMA) / qCMOS.PIXEL_SIZE))  
        elif not meta.estimating.lower() == 'velocity':
            self.lower_bound = int(np.ceil(self.CENTER + (5*DMD.PIXEL_SIZE + 4*self.SIGMA) / qCMOS.PIXEL_SIZE))
            self.upper_bound = int(np.ceil(self.CENTER - (5*DMD.PIXEL_SIZE + 4*self.SIGMA) / qCMOS.PIXEL_SIZE))

        self.detectors = self.lower_bound - self.upper_bound
        super().__init__(*args, **kwargs)

    def crop(self):
        self.cropped = self.raw[..., self.upper_bound:self.lower_bound, self.X_AXIS]




######################
#     Estimation     #
######################
class Estimation:
    @classmethod
    def FromRaw(cls, raw: np.ndarray, metadata: MetaData):
        if metadata.measurement.upper() == 'SPADE':
            expt = SPADE(raw=raw, metadata=metadata)
        elif metadata.measurement.upper() == 'DI':
            expt = DI(raw=raw, metadata=metadata)
        else:
            raise ValueError('measurement must be SPADE or DI')
        
        del raw

        gc.collect()
        expt.est_all()
        return expt.estimates


    @classmethod
    def FromEstimates(cls, Estimates_instance: Estimates):
        c: Estimates = np.copy(Estimates_instance).item()
        del Estimates_instance

        if c.metadata.measurement.upper() == 'SPADE':
            expt = SPADE(cropped=c.cropped_data, background=c.background, metadata=c.metadata)
        elif c.metadata.measurement.upper() == 'DI':
            expt = DI(cropped=c.cropped_data, background=c.background, metadata=c.metadata)
        else:
            raise ValueError('measurement must be SPADE or DI')

        gc.collect()
        expt.est_all()
        return expt.estimates




######################
#     FI and CRB     #
######################
def FisherInformation(A_list: np.ndarray, freq_list: np.ndarray, sigma=_Share.SIGMA, N=50):
    results_1, results_2 = [], []
    n = np.arange(N)
    for A in A_list:
        for f in freq_list:
            _temp = ((tau*n) * np.cos(tau*f*n))**2
            results_1.append((A/sigma)**2 * _temp.sum())
        results_2.append(results_1)
    return np.array(results_2)

def ApproxFisherInformation(A_list: np.ndarray, sigma=_Share.SIGMA, N=50):
    n = np.arange(N)
    results = [2*(A*pi/sigma)**2 * (n**2).sum() for A in A_list]
    return np.array(results)

def VelocityFisherInformationMatrix(sampling_rate=20, sigma=_Share.SIGMA, N=10):
    t = np.arange(N)/sampling_rate
    fi11 = 1/sigma**2 * np.sum(t**2)
    fi12 = 1/sigma**2 * np.sum(t**1)
    fi22 = 1/sigma**2 * np.sum(t**0)
    return np.array([[fi11, fi12], [fi12, fi22]])



#####################
#     Simulator     #
#####################
class Simulator:
    def __init__(self, 
                 metadata: MetaData, 
                 waveform: str = 'sign',
                 sampling_rate = 20, 
                 delay = 0):
        ''' 
        Parameters:
            metadata (MetaData): MetaData instance
            waveform (str): 'sign', 'sin', or 'linear', 
            sampling_rate (int): default is 20 Hz
            delay (float): default is 0
        '''

        self.meta = metadata
        self.waveform = waveform
        self.sampling_rate = sampling_rate
        self.delay = delay


    def loc(self, n):

        fs = self.sampling_rate
        t = n / fs
        fo = fs * self.meta.ground_truth

        if self.waveform.lower() == 'sign':
            _k = np.sign(np.sin(tau * fo * (t + self.delay)))
            if _k == 0:
                _k = 1
            return self.meta.amplitude * (1 + _k)
        elif self.waveform.lower() == 'sin':
            return self.meta.amplitude * (1 + np.sin(tau * fo * (t + self.delay)))
        elif self.waveform.lower() == 'linear':
            return self.meta.ground_truth * (t + 1e-6 + self.delay) - 5 * DMD.PIXEL_SIZE
        else:
            raise ValueError('waveform must be sign or sin')


    def gen(self, photons=None, noise=0, sample_length=None):
        '''
        Generate simulated data using a statistical histogram method.

        Parameters:
            photons (int): Number of photons to generate for each sample, default is 400 (DI) or 60 (SPADE).
            noise (int): The lambda parameter of the poisson, default is 0.
            sample_length (int): Length of the generated data, default is 50.

        Returns:
            np.ndarray: Simulated data array.
        '''
        photons = photons or (400 if self.meta.measurement.lower() == 'di' else 60)
        sample_length = sample_length or (10 if self.waveform.lower() == 'linear' else 50)

        if self.meta.measurement.lower() == 'spade':
            _sig = SPADE.SIGMA

            p1 = lambda s: (s-2*_sig)**2*np.exp(-s**2/(4*_sig**2))/(8*_sig**2)
            p2 = lambda s: (s+2*_sig)**2*np.exp(-s**2/(4*_sig**2))/(8*_sig**2)

            data = [np.histogram(np.random.uniform(0, 1, photons), 
                    bins=[0, p1(self.loc(n)), p1(self.loc(n))+p2(self.loc(n))])[0] for n in range(sample_length)]
            
            data = np.array(data).astype(float)

        if self.meta.measurement.lower() == 'di':
            def _gen_one(n):
                # Convert length units to camera pixel size to match experimental data.
                _loc = self.loc(n) / qCMOS.PIXEL_SIZE
                _sig = DI.SIGMA / qCMOS.PIXEL_SIZE

                # (lower_bound, upper_bound), detectors = DI.crop_bound(self.meta.amplitude)
                detectors = 230

                return np.histogram(np.random.normal(detectors/2+_loc, _sig, photons), 
                                    bins=detectors, range=(0, detectors))[0]

            data = np.array([_gen_one(n) for n in range(sample_length)]).astype(float)

        noise = np.random.poisson(noise, size=data.shape)
        return np.round((data + noise)/qCMOS.CONVERSION_FACTOR + qCMOS.OFFSET), np.round((noise)/qCMOS.CONVERSION_FACTOR + qCMOS.OFFSET)
