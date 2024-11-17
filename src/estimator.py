from math import *
import numpy as np
from scipy.optimize import curve_fit, minimize


__all__ = ['freq_estimator', 'td_estimator']


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





def td_estimator(measurement, sample: np.ndarray, w=None, di_method='mle'):
        from .core import DI, qCMOS

        origin_shape = sample.shape

        detectors = origin_shape[-1]
        works = np.prod(origin_shape[:-1])

        flatten_data = sample.reshape(-1, detectors)


        if measurement.lower() == 'spade':
            time_domain = sample[..., 1] - sample[..., 0]


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


        # Standardize
        time_domain = np.array(time_domain)
        std = time_domain.std()
        mean = time_domain.mean()

        return ((time_domain - mean) / std).reshape(*origin_shape[:-1])
