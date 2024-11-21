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




def td_estimator(measurement, sample: np.ndarray, w=None, di_method='mle', spade_method='sub', standardize=True):
        from .core import DI, SPADE, qCMOS

        origin_shape = sample.shape

        detectors = origin_shape[-1]
        works = np.prod(origin_shape[:-1])

        flatten_data = sample.reshape(-1, detectors)


        if measurement.lower() == 'spade':
            if spade_method.lower() == 'sub':
                time_domain = sample[..., 1] - sample[..., 0]
            elif spade_method.lower() == 'zhou2023':
                k = (sample[..., 0] - qCMOS.OFFSET) / (sample[..., 1] - qCMOS.OFFSET)
                time_domain = 2*SPADE.SIGMA * (1-np.sqrt(k)) / (1+np.sqrt(k))
                time_domain[np.isnan(time_domain)] = -2*SPADE.SIGMA 


        elif measurement.lower() == 'di':

            if di_method.lower() == 'simple':
                temp = flatten_data / flatten_data.sum(axis=-1).reshape(-1, 1)
                x_axis = np.arange(detectors)
                time_domain = (temp @ x_axis)

            elif di_method.lower() == 'mle':
                def _p(x, s, w):
                    _sigma = DI.SIGMA / qCMOS.PIXEL_SIZE
                    return (1-w)/np.sqrt(tau*_sigma**2)*np.exp(-(x-s)**2/(2*_sigma**2)) + w/detectors

                def _nll(data, w):
                    axis = np.arange(detectors)
                    return lambda s: - data.T @ np.log(_p(axis, s, w)) / data.sum()

                def _run_mle(data, w):
                    # Use BFGS algorithm to minimize negative log-likelihood function.
                    result = minimize(_nll(data, w), x0=detectors/2)
                    
                    if result.success:
                        return result.x[0]
                    else:
                        raise RuntimeError('not converged')
                
                if w is None:
                    time_domain = [_run_mle(flatten_data[i], 0) for i in range(works)]
                else:
                    w_set = np.ravel(w)
                    time_domain = [_run_mle(flatten_data[i], w_set[i]) for i in range(works)]

        time_domain = np.array(time_domain)
        
        if standardize:
            # Standardize
            std = time_domain.std()
            mean = time_domain.mean()
            return ((time_domain - mean) / std).reshape(*origin_shape[:-1])
        else:
            return time_domain.reshape(*origin_shape[:-1]) * [qCMOS.PIXEL_SIZE if measurement.lower() == 'DI' else 1][0]