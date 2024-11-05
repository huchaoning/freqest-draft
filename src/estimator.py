from math import *
import numpy as np
from scipy.optimize import curve_fit, minimize


__all__ = ['freq_estimator', 'td_estimator']


def freq_estimator(sample: np.ndarray, sampling_rate=1, method='lse'):
    # Standardize sample
    sample = (sample - sample.mean()) / sample.std()


    def normalize_phase(phase):
        # Normalize phase to be within [0, 2*pi)
        phase = phase % tau
        # Adjust phase to be within [-pi, pi)
        if phase > pi:
            phase -= tau
        return phase


    def waveform(n, f, phi):
        return np.sin(tau * f * n / sampling_rate + phi)


    # Use FFT as pre-estimator, if method is fft, retrun FFT estimates.
    def pre_estimator():
        l = len(sample)

        fft = np.fft.fft(sample)[:l // 2]
        freq = np.fft.fftfreq(l, 1)[:l // 2]
        
        amplitude = np.abs(fft)
        phi = np.angle(fft)

        peak_index = np.argmax(amplitude[1:]) + 1
        pre_freq_est = freq[peak_index]
        pre_phi_est = phi[peak_index]

        return pre_freq_est, normalize_phase(pre_phi_est + pi/2)


    if method.lower() == 'fft':
        return pre_estimator()


    elif method.lower() == 'lse':
        # Use LSE as frequency estimator. 
        # Note that LSE and MLE are the same in WGN.
        n = np.arange(len(sample), dtype=np.float64)
        popt, _= curve_fit(f=waveform, xdata=n, ydata=sample, p0=pre_estimator())
        return popt

    else:
        raise ValueError('method must be lse or fft')



def td_estimator(measurement, sample: np.ndarray, w=None, method='mle'):
        from .core import DI, qCMOS

        origin_shape = sample.shape

        detectors = origin_shape[-1]
        works = np.prod(origin_shape[:-1])

        flatten_data = sample.reshape(-1, detectors)

        if method.lower() == 'simple':

            if measurement.lower() == 'di':
                temp = flatten_data / flatten_data.sum(axis=-1).reshape(-1, 1)
                x_axis = np.arange(detectors)
                time_domain = (temp @ x_axis)

            if measurement.lower() == 'spade':
                time_domain = sample[..., 1] - sample[..., 0]


        elif method.lower() == 'mle':
            _sigma = DI.SIGMA / qCMOS.PIXEL_SIZE

            if measurement.lower() == 'di':
                def _p(x, s, w):
                    return (1-w)/np.sqrt(tau*_sigma**2)*np.exp(-(x-s)**2/(2*_sigma**2)) + w/detectors

            elif measurement.lower() == 'spade':
                def _p(k, s, w):
                    return (1-w)/8*np.exp(-(s/(2*_sigma))**2)*(s/_sigma+2*(-1)**(k+1))**2 + w/detectors

            def _nll(data, w):
                axis = np.arange(detectors)
                return lambda s: - data.T @ np.log(_p(axis, s, w)) / data.sum()

            def _run_mle(data, w):
                # Use BFGS algorithm to minimize negative log-likelihood function.
                result = minimize(_nll(data, w), x0=detectors/2, method='BFGS')
                
                if result.success:
                    return result.x[0]
                else:
                    raise RuntimeError('not converged')
            
            if w is None:
                time_domain = [_run_mle(flatten_data[i], 0) for i in range(works)]
            else:
                w_set = np.ravel(w)
                time_domain = [_run_mle(flatten_data[i], w_set[i]) for i in range(works)]  

        return np.array(time_domain).reshape(*origin_shape[:-1])







# def di_td_estimator(sample: np.ndarray, w = None, method='mle'):
#         from .core import DI, qCMOS
        
#         origin_shape = sample.shape

#         detectors = origin_shape[-1]
#         works = np.prod(origin_shape[:-1])

#         flatten_data = sample.reshape(-1, detectors)

#         if method.lower() == 'simple':
#             temp = flatten_data / flatten_data.sum(axis=-1).reshape(-1, 1)
#             x_axis = np.arange(detectors)
#             time_domain = (temp @ x_axis)

#         elif method.lower() == 'mle':
#             def _p(x, s, w):
#                 _sigma = DI.SIGMA / qCMOS.PIXEL_SIZE
#                 return (1-w)/np.sqrt(tau*_sigma**2)*np.exp(-(x-s)**2/(2*_sigma**2)) + w/detectors
            
#             def _negative_ll(data, w):
#                 x_axis = np.arange(detectors)
#                 return lambda s: - data.T @ np.log(_p(x_axis, s, w)) / data.sum()
            
#             def _run_mle(data, w):
#                 # Use BFGS algorithm to minimize negative log-likelihood function.
#                 result = minimize(_negative_ll(data, w), x0=detectors/2, method='BFGS')
                
#                 if result.success:
#                     return result.x[0]
#                 else:
#                     raise RuntimeError('not converged')
            
#             if w is None:
#                 time_domain = [_run_mle(flatten_data[i], 0) for i in range(works)]
#             else:
#                 w_set = np.ravel(w)
#                 time_domain = [_run_mle(flatten_data[i], w_set[i]) for i in range(works)]  

#         return np.array(time_domain).reshape(*origin_shape[:-1])
