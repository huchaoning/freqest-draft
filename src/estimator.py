from math import *
import numpy as np
from scipy.optimize import curve_fit, minimize


__all__ = ['velocity_estimator', 'freq_estimator', 'td_estimator']


def velocity_estimator(sample: np.ndarray, sampling_rate=20):
    def waveform(t, v, b):
        return v * t + b

    l = len(sample)

    (v_est, b_est), _ = curve_fit(waveform, np.arange(l)/sampling_rate, sample, p0=[0, 0])
    return v_est, b_est



def freq_estimator(sample: np.ndarray, method='lse'):
    def waveform(n, f):
        return np.sin(tau * f * n)

    # Use FFT as pre-estimator, if method is fft, retrun FFT estimates.
    l = len(sample)

    fft = np.abs(np.fft.fft(sample)[:l // 2])
    freq = np.fft.fftfreq(l, 1)[:l // 2]

    peak_index = np.argmax(fft[1:]) + 1
    pre_freq_est = freq[peak_index]


    if method.lower() == 'fft':
        return pre_freq_est


    elif method.lower() == 'lse':
        # Use LSE as frequency estimator. 
        # Note that LSE and MLE are the same in WGN.
        n = np.arange(len(sample), dtype=np.float64)
        popt, _= curve_fit(f=waveform, xdata=n, ydata=sample, p0=pre_freq_est)
        return popt.item()

    else:
        raise ValueError('method must be lse or fft')




# The camera pixels are used as the length unit, so grid size is 1.
# In that case, the function values of the Gaussian distribution can be used to approximate the integral values.
class _MLE:
    def _p(self, x, s, w):
        return 0

    def _nll(self, data, w, detectors):
        axis = np.arange(detectors)
        return lambda s: - data.T @ np.log(self._p(axis, s, w)) / data.sum()

    def run(self, data, w):
        # Use BFGS algorithm to minimize negative log-likelihood function.
        result = minimize(self._nll(data, w, len(data)), x0=len(data)/2)
        if result.success:
            return result.x[0]
        else:
            raise RuntimeError('not converged')


def _preprocess(sample: np.ndarray):
    origin_shape = sample.shape
    detectors = origin_shape[-1]
    works = np.prod(origin_shape[:-1])
    flatten = sample.reshape(-1, detectors)
    return works, flatten


def _standardize(time_domain: np.ndarray, standardize: bool):
    if standardize:
        std = time_domain.std()
        mean = time_domain.mean()
        time_domain = (time_domain - mean) / std
    return time_domain


def _spade_td(sample: np.ndarray, w: np.ndarray, method: str):
    from .core import SPADE, qCMOS
    if method == 'sub':
        return sample[..., 1] - sample[..., 0]
    elif method == 'zhou2023':
        # If the ADU value is less than the qCMOS offset, means the signal here is zero.
        # Add a small offset to avoid division by zero errors.
        _sample = np.clip(sample - qCMOS.OFFSET, 0, np.inf) + 1e-12
        k = _sample[..., 0] / _sample[..., 1]
        time_domain = 2 * SPADE.SIGMA * (1 - np.sqrt(k)) / (1 + np.sqrt(k))
        return time_domain
    elif method == 'mle':
        class MLE(_MLE):
            def _p(self, k, s, w):
                return (1-w)/8 * np.exp(-(s/(2*SPADE.SIGMA/qCMOS.PIXEL_SIZE))**2) * (s / (SPADE.SIGMA/qCMOS.PIXEL_SIZE) + 2*(-1)**(k + 1))**2 + w/sample.shape[-1]
        mle = MLE()
        works, flatten_data = _preprocess(sample)
        if w is None:
            time_domain = [qCMOS.PIXEL_SIZE * mle.run(flatten_data[i], 0) for i in range(works)]
        else:
            time_domain = [qCMOS.PIXEL_SIZE * mle.run(flatten_data[i], w.mean()) for i in range(works)]
        return np.array(time_domain)
    else:
        raise ValueError('spade_method must be sub, zhou2023, or mle')


def _di_td(sample: np.ndarray, w: np.ndarray, method: str):
    from .core import DI, qCMOS
    if method == 'simple':
        temp = sample.reshape(-1, sample.shape[-1]) / sample.reshape(-1, sample.shape[-1]).sum(axis=-1).reshape(-1, 1)
        x_axis = np.arange(sample.shape[-1])
        time_domain = (temp @ x_axis) * qCMOS.PIXEL_SIZE
        return time_domain
    elif method == 'mle':
        class MLE(_MLE):
            def _p(self, x, s, w):
                return (1-w) / np.sqrt(tau*(DI.SIGMA/qCMOS.PIXEL_SIZE)**2) * np.exp(-(x-s)**2 / (2*(DI.SIGMA/qCMOS.PIXEL_SIZE)**2)) + w/sample.shape[-1]
        mle = MLE()
        works, flatten = _preprocess(sample)
        if w is None:
            time_domain = [qCMOS.PIXEL_SIZE * mle.run(flatten[i], 0) for i in range(works)]
        else:
            time_domain = [qCMOS.PIXEL_SIZE * mle.run(flatten[i], w.mean()) for i in range(works)]
        return np.array(time_domain)
    else:
        raise ValueError('di_method must be simple or mle')


def td_estimator(measurement: str, 
                 data: np.ndarray, 
                 w: np.ndarray = None, 
                 di_method: str = 'mle', 
                 spade_method: str = 'sub', 
                 standardize: bool = True):
    
    if measurement.lower() == 'spade':
        time_domain = _spade_td(data, w, spade_method)
    elif measurement.lower() == 'di':
        time_domain = _di_td(data, w, di_method)
    else:
        raise ValueError('measurement must be spade or di')

    time_domain = _standardize(time_domain, standardize)
    return time_domain.reshape(*data.shape[:-1])