from math import tau
import numpy as np
from scipy.optimize import curve_fit, minimize, brute
from scipy.special import erf
from scipy.stats import norm


__all__ = ['velocity_estimator', 'freq_estimator', 'td_estimator']


def velocity_estimator(sample: np.ndarray, sampling_rate=20):
    def waveform(t, v, b):
        return v * t + b

    l = len(sample)

    (v_est, b_est), _ = curve_fit(waveform, np.arange(l)/sampling_rate, sample, p0=[0, 0])
    return v_est, b_est



def freq_estimator(sample: np.ndarray, method='lse', zero_padding=0, window_type='none'):
    def _waveform(n, f):
        return np.sin(tau * f * n)
    
    def _grad_waveform(n, f):
        return tau * n * np.cos(tau * f * n)
    
    def _residual(f):
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
        search_range = (0.08, 0.42)
        grid_points = 256

        search_grid = (slice(search_range[0], search_range[1], (search_range[1] - search_range[0]) / grid_points),)
        brute_freq = brute(_residual, ranges=search_grid, finish=None)
        return brute_freq


    elif method.lower() == 'lse':
        # Use LSE as frequency estimator. 
        # Note that LSE and MLE are the same in WGN.
        n = np.arange(len(sample), dtype=np.float64)
        popt, _= curve_fit(_waveform, xdata=n, ydata=sample, p0=pre_freq_est, maxfev=1000, jac=_grad_waveform)
        return popt.item()

    else:
        raise ValueError('method must be lse or fft')



# The camera pixels are used as the length unit, so the pixel size is 1.
def _standardize(time_domain: np.ndarray, standardize: bool):
    if standardize:
        std = time_domain.std()
        mean = time_domain.mean()
        time_domain = (time_domain - mean) / std
    return time_domain



def _spade_td(data: np.ndarray, background: np.ndarray, method: str):
    from .core import qCMOS, SPADE

    temp = qCMOS.convert2photons(data)
    k = temp[..., 0] / temp[..., 1]
    zhou2023 = 2 * SPADE.SIGMA * (1 - np.sqrt(k)) / (1 + np.sqrt(k))

    if method.lower() == 'sub':
        return data[..., 1] - data[..., 0]
    
    elif method.lower() == 'zhou2023':
        return zhou2023
    
    elif method.lower() in ('lse', 'mle'):
        origin_shape = data.shape
        flatten = data.reshape(-1, origin_shape[-1])
        works = flatten.shape[0]

        pn = qCMOS.convert2photons(flatten)
        b = qCMOS.convert2photons(background).mean()
        I0 = (pn - b).sum(-1).mean(0)

        def _uk(k, theta):
            xi = theta / (2 * SPADE.SIGMA)
            return I0/2 * (xi + k)**2 * np.exp(-xi**2) + b
        
        def _grad_nll(frame):
            def wrapper(theta):
                xi = theta / (2 * SPADE.SIGMA)
                k = np.array([-1, 1])
                return - np.sum(frame * np.log(_uk(k, theta)) - _uk(k, theta) - (frame * np.log(frame) - frame)), \
                         np.sum(I0/(2*SPADE.SIGMA) * np.exp(-xi**2)*((xi + k) * (xi**2 + k*xi - 1) * (frame/_uk(k, theta) - 1)))
            return wrapper

        time_domain = []
        for i in range(works):
            if method.lower() == 'mle':
                opt = minimize(_grad_nll(pn[i]), 
                               x0 = np.ravel(zhou2023)[i], 
                               jac = True, 
                               bounds = [(-6*SPADE.SIGMA, 6*SPADE.SIGMA)], 
                               tol = 1e-8, 
                               options = {'maxls': 100})

                result = [opt.x.item(), None, opt.success, opt.message]

            elif method.lower() == 'lse':
                result = curve_fit(_uk, xdata=np.array([-1, 1]), ydata=pn[i], p0=np.ravel(zhou2023)[i], full_output=True, maxfev=800, xtol=1e-8)
            
            if result[2]:
                time_domain.append(result[0])
            else:
                raise RuntimeError(f'Time-Domain estimator: {method.upper()} is not converged, {result[3]}')
            
        return np.array(time_domain).reshape(*origin_shape[:-1])

    else:
        raise ValueError('spade_method must be sub, zhou2023, lse, or mle')




def _di_td(data: np.ndarray, background: np.ndarray, method: str):
    from .core import DI, qCMOS
    origin_shape = data.shape
    flatten = data.reshape(-1, origin_shape[-1])
    works = flatten.shape[0]
    pixels = origin_shape[-1]

    pn = qCMOS.convert2photons(flatten)
    mass_center = pn @ np.arange(pixels) / pn.sum(-1)

    if method.lower() == 'simple':
        time_domain = mass_center

    elif method.lower() in ('mle', 'lse'):
        b = qCMOS.convert2photons(background).mean()
        I0 = (pn - b).sum(-1).mean(0)
        _sig = DI.SIGMA / qCMOS.PIXEL_SIZE

        def _uk(x_, theta):
            z1 = (x_ - theta + 0.5) / (_sig * (2**0.5))
            z2 = (x_ - theta - 0.5) / (_sig * (2**0.5))
            DeltaE = erf(z1)/2 - erf(z2)/2
            return I0 * DeltaE + b

        def _grad_nll(frame):
            x = np.arange(pixels).astype(float)
            def wrapper(theta):
                z1 = (x - theta + 0.5) / (_sig * (2**0.5))
                z2 = (x - theta - 0.5) / (_sig * (2**0.5))
                DeltaE = erf(z1)/2 - erf(z2)/2
                uk = I0 * DeltaE + b
                return - np.sum(frame * np.log(uk) - uk - (frame * np.log(frame) - frame)), \
                       - I0/(_sig*tau**0.5) * np.sum((-np.exp(-z1**2) + np.exp(-z2**2)) * (frame/uk - 1))
            return wrapper

        time_domain = []
        for i in range(works):
            if method.lower() == 'mle':
                opt = minimize(_grad_nll(pn[i]), 
                               x0 = mass_center[i], 
                               jac = True, 
                               bounds = [(0, pixels)], 
                               tol = 1e-8, 
                               options = {'maxls': 100})

                result = [opt.x.item(), None, opt.success, opt.message]

            elif method.lower() == 'lse':
                result = curve_fit(_uk, xdata=np.arange(pixels), ydata=pn[i], p0=mass_center[i], full_output=True, maxfev=800, xtol=1e-8)
            
            if result[2]:
                time_domain.append(result[0])
            else:
                raise RuntimeError(f'Time-Domain estimator: {method.upper()} is not converged, {result[3]}')

    else:
        raise ValueError('di_method must be simple, lse or mle')
    
    return (np.array(time_domain).reshape(*origin_shape[:-1]) - pixels/2) * qCMOS.PIXEL_SIZE


def td_estimator(measurement: str, 
                 data: np.ndarray, 
                 background: np.ndarray, 
                 method: str,
                 standardize: bool):
    
    if measurement.lower() == 'spade':
        time_domain = _spade_td(data, background, method)
    elif measurement.lower() == 'di':
        time_domain = _di_td(data, background, method)
    else:
        raise ValueError('measurement must be SPADE or DI')

    return _standardize(time_domain, standardize)