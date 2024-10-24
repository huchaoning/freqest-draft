from math import *
import numpy as np
from scipy.optimize import curve_fit


__all__ = ['freq_estimator']


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

