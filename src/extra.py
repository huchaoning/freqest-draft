from math import tau
import numpy as np
from scipy.optimize import curve_fit, minimize, brute
from scipy.special import erf
from scipy.stats import norm


__all__ = ['extra_estimator']


def freq_estimator(estimates_instance, 
                   method='lse', 
                   
                   zero_padding=0, 
                   window_type='none'):
    

    from .core import Estimates
    c: Estimates = estimates_instance



    def _waveform(n, f):
        return np.sin(tau * f * n)
    
    def _grad_waveform(n, f):
        return tau * n * np.cos(tau * f * n)
    
    def _residual(f):
        n = np.arange(len(sample), dtype=np.float64)
        return np.sum((sample - _waveform(n, f)) ** 2)

    l = len(sample)


    # Use FFT as pre-estimator, if method is fft, retrun FFT estimates.
    # Apply windowing function
    window_type = window_type.lower()
    if window_type == 'hanning':
        window = np.hanning(l)
    elif window_type == 'hamming':
        window = np.hamming(l) 
    elif window_type == 'blackman':
        window = np.blackman(l)
    elif window_type == 'none':
        window = 1
    else:
        raise ValueError("window type must be one of 'hanning', 'hamming', or 'blackman'.")

    # Apply the window to the signal
    windowed_sample = _standardize(sample, True) * window

    # Zero-padding to increase frequency resolution
    n = l + zero_padding
    fft = np.abs(np.fft.fft(windowed_sample, n=n))[:n // 2]
    freq = np.fft.fftfreq(n, 1)[:n // 2]

    # Find the peak in the FFT
    peak_index = np.argmax(fft[1:]) + 1
    pre_freq_est = freq[peak_index]


    if method.lower() == 'fft':
        return pre_freq_est
    

    if method.lower() == 'brute':
        # Brute-force search for the initial frequency
        search_range = (0.05, 0.45)
        grid_points = 256

        search_grid = (slice(search_range[0], search_range[1], (search_range[1] - search_range[0]) / grid_points),)
        brute_freq = brute(_residual, ranges=search_grid, finish=None)
        return brute_freq


    elif method.lower() == 'lse':
        # Use LSE as frequency estimator. 
        # Note that this LSE is an non-linear LSE.
        n = np.arange(len(sample), dtype=np.float64)
        popt, _= curve_fit(_waveform, xdata=n, ydata=sample, p0=pre_freq_est, maxfev=1000, jac=_grad_waveform)
        return popt.item()
    
    elif method.lower() == 'mle':
        # Use MLE as frequency estimator. (Kay1993)
        N = len(sample)
        n = np.arange(N, dtype=np.float64)
        # _I = lambda f: - np.abs((sample * np.exp(-2j*np.pi*f*n)).sum())**2 / N
        def _I(f):
            exr1 = np.sum(sample * np.cos(2*np.pi * f * n))
            exr2 = np.sum(sample * np.sin(2*np.pi * f * n) * n)

            exr3 = np.sum(sample * np.sin(2*np.pi * f * n))
            exr4 = np.sum(sample * np.cos(2*np.pi * f * n) * n)

            return - np.abs((sample * np.exp(-2j*np.pi*f*n)).sum())**2 / N, \
                    2*np.pi/N * (exr1*exr2 - exr3*exr4)

        result = minimize(_I, 
                          pre_freq_est,
                          bounds = [(0.05, 0.45)], 
                          tol = 1e-8, 
                          options = {'maxls': 100},
                          jac=True)

        if result.success:
            return result.x.item()
        else:
            raise RuntimeError(f'not converged: {result.message}')

    else:
        raise ValueError('method must be lse or fft')



# The camera pixels are used as the length unit, so the pixel size is 1.
def _standardize(time_domain: np.ndarray, standardize: bool):
    if standardize:
        std = time_domain.std()
        mean = time_domain.mean()
        time_domain = (time_domain - mean) / std
    return time_domain