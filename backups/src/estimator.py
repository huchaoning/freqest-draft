#####################################################
# 注意: FFT 返回的估计值中的相位估计值, 是针对 cos 的. #
#####################################################


from math import *
import numpy as np
from scipy.optimize import curve_fit


__all__ = ['freq_estimator']


def freq_estimator(sample: np.ndarray, sampling_rate=1, method='lse'):
    # Standardize sample
    sample = (sample - sample.mean()) / sample.std()

    def waveform(n, f):
        return np.sin(tau * f * n / sampling_rate)


    # Use FFT as pre-estimator, if method is fft, retrun FFT estimates.
    def pre_estimator():
        l = len(sample)

        fft = np.fft.fft(sample)[:l // 2]
        freq = np.fft.fftfreq(l, 1 / sampling_rate)[:l // 2]
        amplitude = np.abs(fft)

        peak_index = np.argmax(amplitude[1:]) + 1
        pre_freq_est = freq[peak_index]

        return pre_freq_est


    if method.lower() == 'fft':
        return pre_estimator()


    elif method.lower() == 'lse':
        # Use LSE as frequency estimator. 
        # Note that LSE and MLE are the same in WGN.
        n = np.arange(len(sample), dtype=np.float64)
        popt, _ = curve_fit(f=waveform, xdata=n, ydata=sample, p0=pre_estimator())
        return popt

    else:
        raise ValueError('method must be lse or fft')






'''
# 使用多参数估计的方法, 信号标准化为自己瞎写的一个算法. 拟合的波形是 cos


# START OF OLD FREQ EST
def freq_estimator(sample, sampling_rate=1, method='lse'):
    # Normalize
    temp = sample - sample.mean()
    scale = (temp[temp>0].mean() - temp[temp<0].mean()) / 2
    sample = temp / scale

    def normalize_phase(phase):
        # Normalize phase to be within [0, 2*pi)
        phase = phase % tau
        # Adjust phase to be within [-pi, pi)
        if phase > pi:
            phase -= tau
        return phase


    def waveform(n, f, phi):
        return np.cos(tau * f * n / sampling_rate + phi)


    # Use FFT as pre-estimator, if method is fft, retrun FFT estimates.
    def pre_estimator():
        l = len(sample)

        fft = np.fft.fft(sample)[:l // 2]
        freq = np.fft.fftfreq(l, 1 / sampling_rate)[:l // 2]
        
        amplitude = np.abs(fft)
        phi = np.angle(fft)

        peak_index = np.argmax(amplitude[1:]) + 1
        pre_freq_est = freq[peak_index]
        pre_phi_est = phi[peak_index]

        return pre_freq_est, normalize_phase(pre_phi_est)


    if method.lower() == 'fft':
        return pre_estimator()


    elif method.lower() == 'lse':
        # Use LSE as frequency estimator. 
        # Note that LSE and MLE are the same in WGN.
        n = np.arange(len(sample), dtype=np.float64)
        popt, pcov = curve_fit(f=waveform, xdata=n, ydata=sample, p0=pre_estimator())

        if np.linalg.cond(pcov) <= 5e5:
            freq_est, phi_est = popt
        else:
            raise RuntimeError('np.linalg.cond(pcov) > 50000, the algorithm may not converge')

        return freq_est, phi_est


    else:
        raise ValueError('method must be lse or fft')

        
# END OF OLD FREQ EST
'''