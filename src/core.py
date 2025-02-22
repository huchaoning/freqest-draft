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

    @classmethod
    def A(cls, px):
        return px * cls.PIXEL_SIZE / 2


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
        else:
            self.measurement = self.measurement.upper()

    def convert2str(self):
        return f'{self.measurement.lower()}_{round(self.amplitude*2/DMD.PIXEL_SIZE)}px_f{self.ground_truth}_d{self.pwm_duty}'

    def __repr__(self):
        return super().__repr__()


@dataclass
class Estimates(_Repr):
    '''
    A data class to store all data and estimates.

    Parameters:
        metadata (MetaData): MetaData instance.
        
        cropped_data (np.ndarray, unit: ADU): Raw data is a 2D image, we used only 1D data cropped from raw data.
        background (float, unit: photons): Averaged background photons noise per pixel.
        photons (float, unit: photons): Averaged total signal photons used for estimation.

        time_domain (np.ndarray): Time domain signal estimated by MLE localization algorithm.
        estimates (np.ndarray): The frequency estimates. A * sin(2 * pi * f * n + phi)
    '''

    metadata: MetaData

    cropped_data: np.ndarray
    background: float
    photons: float

    time_domain: np.ndarray = None
    estimates: np.ndarray = None

    def est(self):
        return freq_est(td_est(self))


    def savez(self, dirname):
        dirname = os.path.expanduser(dirname)
        filename = os.path.join(dirname, self.metadata.convert2str() + '.npz')

        if os.path.exists(filename): 
            raise FileExistsError(f'{filename} already exists')

        self.cropped_data = self.cropped_data.astype(np.uint16)
        np.savez_compressed(filename, **self.__dict__)


    def __repr__(self):
        return super().__repr__()



def LoadEstimates(file) -> Estimates:
    file = os.path.expanduser(file)
    npz = np.load(file, allow_pickle=True)
    dic = {}
    for k in npz.files:
        dic[k] = npz[k]
        if k.lower() == 'cropped_data':
            dic[k] = npz[k].astype(float)
        if k.lower() == 'metadata':
            dic[k] = npz[k].item()
    return Estimates(**dic)



def NewEstimates(raw_path: str, metadata: MetaData, photons = None) -> Estimates:
    if os.path.exists(raw_path):
        raw = np.load(raw_path)
    else:
        raise FileNotFoundError(f".npy file '{raw_path}' not found")

    raw = raw.astype(float)

    # Method 1
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

    photons = (qCMOS.convert2photons(cropped).mean() - background) * cropped.shape[-1]

    # if metadata.pwm_duty == 0:
    #     photons_ = qCMOS.convert2photons(cropped).sum(-1).mean()

    # elif photons is not None:
    #     photons_ = photons

    # else:
    #     raise ValueError('PWM duty is not 0, photons is needed.')

    # Method 2
    # ERROR
    # background = (qCMOS.convert2photons(cropped).sum(-1).mean() - photons_) / cropped.shape[-1]

    return Estimates(metadata, cropped, background, photons)





######################
#    Measurements    #
######################
class _Share:
    SIGMA = 103 #um
    SAMPLE_LENGTH = 50

    @classmethod
    def CFI(cls, b, A, f, nu=1):
        b = np.clip(b, 1e-10, np.inf) # smoothing
        n = np.arange(cls.SAMPLE_LENGTH)

        def _cal(b, f):
            alpha = tau * f * n
            s  = A * np.sin(alpha)
            ds = A*tau*n * np.cos(alpha)
            return (cls.gamma(s, b/nu) * ds**2).sum(-1)

        if np.array(f).ndim != 0:
            return np.array([_cal(b, _f) for _f in f])
        elif np.array(b).ndim != 0:
            return np.array([_cal(_b, f) for _b in b])
        else:
            return _cal(b, f)
        
    @classmethod
    def ApproxCFI(cls, A):
        sum_n = (np.arange(cls.SAMPLE_LENGTH)**2).sum()
        return 2*(A*pi/cls.SIGMA)**2 * sum_n



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
class PSF:
    def __init__(self, sigma):
        self.sigma = sigma

    def abs_sq(self, x):
        return np.abs(self.psf(x))**2

    def pixelized(self, j_th, pixel_size=1):
        from scipy.integrate import quad
        return quad(self.abs_sq, pixel_size*j_th - pixel_size/2, pixel_size*j_th + pixel_size/2, limit=1000)[0]
    
    def prob_table(self, lower_limit: int, upper_limit: int, pixel_size=1):
        return np.array([self.pixelized(j, pixel_size) for j in np.arange(lower_limit, upper_limit + 1, 1)])


class SincPSF(PSF):
    def psf(self, x):
        return np.sinc(x / (self.sigma*np.pi)) / (self.sigma*np.pi)**0.5
    

class GausPSF(PSF):
    def psf(self, x):
        return np.exp(-(x**2) / (4*self.sigma**2)) / (2*np.pi*self.sigma**2)**0.25


class Simulator:
    def __init__(self, 
                 metadata: MetaData, 
                 waveform: str = 'sign',
                 psf: str = 'gaus',
                 sampling_rate = 20, 
                 repeat = 200,
                 sample_length = _Share.SAMPLE_LENGTH):
        ''' 
        Parameters:
            metadata (MetaData): MetaData instance
            waveform (str): 'sign' or 'sin' 
            psf (str): 'gaus' or 'sinc', affect to DI only
            sampling_rate (int): default is 20 Hz
            delay (float): default is 0
        '''

        self.meta = metadata
        self.waveform = waveform.lower()
        self.psf = psf.lower()
        self.sampling_rate = sampling_rate
        self.repeat = repeat
        self.N = sample_length

        self.meta.pwm_duty = 'SIM'

    def loc(self, n, delay):
        fs = self.sampling_rate
        t = n / fs
        fo = fs * self.meta.ground_truth

        if self.waveform == 'sign':
            _k = np.sign(np.sin(tau * fo * (t + delay)))
            if _k == 0:
                _k = 1
            return self.meta.amplitude * (_k)

        elif self.waveform == 'sin':
            return self.meta.amplitude * (np.sin(tau * fo * (t + delay)))

        else:
            raise ValueError("waveform must be 'sign' or 'sin'")


    def gen(self, noise=0, delay=0, photons=None):
        '''
        Generate simulated data using a statistical histogram method.

        Parameters:
            photons (int): Number of photons to generate for each sample, default is 400 (DI) or 60 (SPADE).
            noise (int): The lambda parameter of the poisson, default is 0.
            sample_length (int): Length of the generated data, default is SAMPLE_LENGTH.

        Returns:
            np.ndarray: Simulated data array.
        '''
        photons = photons or (400 if self.meta.measurement.lower() == 'di' else 60)

        if self.meta.measurement.lower() == 'spade':
            _sig = SPADE.SIGMA

            p1 = lambda s: (s-2*_sig)**2*np.exp(-s**2/(4*_sig**2))/(8*_sig**2)
            p2 = lambda s: (s+2*_sig)**2*np.exp(-s**2/(4*_sig**2))/(8*_sig**2)

            data = [np.histogram(np.random.uniform(0, 1, photons), 
                    bins=[0, p1(self.loc(n, delay)), p1(self.loc(n, delay)) + p2(self.loc(n, delay))])[0] for n in range(self.N)]

            data = np.array(data).astype(float)

        elif self.meta.measurement.lower() == 'di':
            def _gen_one(n):
                # Convert length units to camera pixel size to match experimental data.
                _loc = self.loc(n, delay) / qCMOS.PIXEL_SIZE
                _sig = DI.SIGMA / qCMOS.PIXEL_SIZE

                detectors = round((2*self.meta.amplitude + 8*DI.SIGMA) / qCMOS.PIXEL_SIZE)

                if self.psf == 'gaus':
                    # Use NumPy random number generator for better performence.
                    outcomes = np.random.normal(detectors/2+_loc, _sig, photons)
                elif self.psf == 'sinc':
                    prob_table = SincPSF(_sig).prob_table(-detectors/2, detectors/2)
                    outcomes = _loc + np.random.choice(len(prob_table), photons, p=prob_table/prob_table.sum())
                else:
                    raise ValueError("psf must be 'gaus' for 'sinc'")

                return np.histogram(outcomes, bins=detectors, range=(0, detectors))[0]

            data = []
            for _ in range(self.repeat):
                for n in range(self.N):
                    data.append(_gen_one(n))
            data = np.array(data).astype(float).reshape(self.repeat, self.N, -1)

        return Estimates(self.meta, 
                         np.round((data + np.random.poisson(noise, size=data.shape)) / qCMOS.CONVERSION_FACTOR + qCMOS.OFFSET),
                         noise, photons)