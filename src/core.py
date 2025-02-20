import os
from math import tau, pi
import numpy as np
import scipy as sp

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
    'NewEstimates',

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
        ground_truth (float): The ground truth value of frequency.

        amplitude (float, unit: um): The amplitude value ().
        pwm_duty (int, optional): The PWM duty cycle. Defaults to 0.
    '''
    measurement: str
    ground_truth: float
    amplitude: float

    pwm_duty: int = 0

    def __post_init__(self):
        if self.measurement.lower() not in ('spade', 'di'):
            raise ValueError("measurement must be 'SPADE' or 'DI'")

    def __repr__(self):
        return super().__repr__()


@dataclass
class Estimates(_Repr):
    '''
    A data class to store all data and estimates.

    Parameters:
        cropped_data (np.ndarray, unit: ADU): Raw data is a 2D image, we used only 1D data cropped from raw data.
        metadata (MetaData): MetaData instance.

        background (float, unit: photons): Averaged background photons noise per pixel.
        photons (float, unit: photons): Averaged total signal photons used for estimation.

        time_domain (np.ndarray): Time domain signal estimated by MLE localization algorithm.

        estimates_a (np.ndarray, unit: um): The A estimates. A * sin(2 * pi * f * n + phi)
        estimates_b (np.ndarray): The f estimates. A * sin(2 * pi * f * n + phi)
        estimates_c (np.ndarray): The phi estimates. A * sin(2 * pi * f * n + phi)
    '''
    cropped_data: np.ndarray
    metadata: MetaData

    background: float
    photons: float

    time_domain: np.ndarray = None

    estimates_a: np.ndarray = None
    estimates_b: np.ndarray = None
    estimates_c: np.ndarray = None

    def est(self):
        return freq_est(td_est(self))


    def savez(self, dirname):
        dirname = os.path.expanduser(dirname)
        truth = self.metadata.ground_truth

        m = self.metadata.measurement.lower()
        d = self.metadata.pwm_duty

        px = round(self.metadata.amplitude / DMD.PIXEL_SIZE * 2)
        filename = os.path.join(dirname, f'{m}_{px}px_f{truth}_d{d}.npz')

        if os.path.exists(filename): 
            raise FileExistsError(f'{filename} already exists')

        np.savez_compressed(filename, **self.__dict__)


    def __repr__(self):
        return super().__repr__()



def LoadEstimates(file) -> Estimates:
    file = os.path.expanduser(file)
    npz = np.load(file, allow_pickle=True)
    dic = {}
    for k in npz.files:
        dic[k] = npz[k]
        if k.lower() == 'metadata':
            dic[k] = npz[k].item()
    return Estimates(**dic)



def NewEstimates(raw_path: str, metadata: MetaData, photons: float = None) -> Estimates:
    if os.path.exists(raw_path):
        raw = np.load(raw_path)
    else:
        raise FileNotFoundError(f".npy file '{raw_path}' not found")

    raw = raw.astype(float)

    temp = raw[..., :-4, :]
    background = (temp[..., :5,   :5].mean((-1, -2)) + temp[..., -5:,   :5].mean((-1, -2))  + 
                  temp[..., :5, -5: ].mean((-1, -2)) + temp[..., -5:, -5: ].mean((-1, -2))) / 4
    background = qCMOS.convert2photons(background).mean()
    
    if metadata.measurement.lower() == 'di':
        lower_bound = int(np.ceil(DI.CENTER + 4*DI.SIGMA / qCMOS.PIXEL_SIZE))
        upper_bound = int(np.ceil(DI.CENTER - (2*metadata.amplitude + 4*DI.SIGMA) / qCMOS.PIXEL_SIZE))  
        cropped = raw[..., upper_bound:lower_bound, DI.X_AXIS]

    elif metadata.measurement.lower() == 'spade':
        cropped = raw[..., (SPADE.POINT_1, SPADE.POINT_2), SPADE.X_AXIS]

    if (metadata.pwm_duty == 0) and (photons is None):
        photons_ = qCMOS.convert2photons(cropped).sum(-1).mean()

    elif photons is not None:
        photons_ = photons

    else:
        raise ValueError('PWM duty is not 0, photons is needed.')

    return Estimates(cropped, metadata, background, photons_)





######################
#    Measurements    #
######################
class _Share:
    SIGMA = 103 #um


class SPADE(_Share): # with PM-mode
    X_AXIS = 89
    POINT_1 = 406
    POINT_2 = 116

    ROI = {'X0': 2128, 'Y0': 720, 'W': 180, 'H': 500}

    @classmethod
    def gamma(cls, s, b):
        xi = s / (2 * cls.SIGMA)
        uk = lambda k: 1 / 2 * (xi + k)**2 * np.exp(-xi**2) + b
        duk = lambda k: - 1 / (2 * cls.SIGMA) * (xi + k) * (xi**2 + k * xi - 1) * np.exp(-xi**2)

        return np.array([1 / uk(k) * duk(k)**2 for k in (-1, 1)]).sum(0)


class DI(_Share):
    X_AXIS = 86
    CENTER = 113
    
    ROI = {'X0': 1440, 'Y0': 876, 'W': 160, 'H': 228}

    @classmethod
    def gamma(cls, s, b, a=qCMOS.PIXEL_SIZE, regin=np.inf):
        s = np.array(s)
        ndim = s.ndim

        s = np.array([s]) if ndim == 0 else s

        from scipy.special import erf
        if regin == np.inf:
            k = np.arange(-10*cls.SIGMA, 10*cls.SIGMA + a, a)
        else:
            k = np.arange(-regin, regin + a, a)

        zp = np.array([(k - _s + 0.5*a) / (cls.SIGMA * (2**0.5)) for _s in np.asarray(s)])
        zn = np.array([(k - _s - 0.5*a) / (cls.SIGMA * (2**0.5)) for _s in np.asarray(s)])
        
        uk = erf(zp)/2 - erf(zn)/2 + b
        duk = 1/(cls.SIGMA*(tau**0.5)) * (-np.exp(-zp**2) + np.exp(-zn**2))

        if ndim == 0:
            return (1 / uk * duk**2).sum()
        else:
            return (1 / uk * duk**2).sum(-1)





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
            return self.meta.amplitude * (_k - 1)
        elif self.waveform.lower() == 'sin':
            return self.meta.amplitude * (np.sin(tau * fo * (t + self.delay)) - 1)
        else:
            raise ValueError("waveform must be 'sign' or 'sin'")


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

                detectors = 230

                return np.histogram(np.random.normal(detectors/2+_loc, _sig, photons), 
                                    bins=detectors, range=(0, detectors))[0]

            data = np.array([_gen_one(n) for n in range(sample_length)]).astype(float)

        noise = np.random.poisson(noise, size=data.shape)
        return np.round((data + noise)/qCMOS.CONVERSION_FACTOR + qCMOS.OFFSET), np.round((noise)/qCMOS.CONVERSION_FACTOR + qCMOS.OFFSET)
