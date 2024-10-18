from math import *
import numpy as np

from scipy.optimize import minimize, curve_fit

from .params import *
from ..macro import aviread
from ..cache import cache



__all__ = [
    'DataReader',
    'TDEst',
    'FreqEst'
]


class _ChooseMeasurement:
    def __init__(self, measure):
        if measure.lower() in ('spade', 'di'):
            self.measure = measure.lower()
        else:
            raise ValueError('measure must be one of spade or di')


class DataReader(_ChooseMeasurement):
    def _spade_reader(self, data):
        return data[:, SPADE.point1[0], SPADE.point1[1]], data[:, SPADE.point2[0], SPADE.point2[1]]


    def _di_roi(self, mirrors):
        if mirrors is None:
            raise ValueError('type(mirrors) must be int, ' + 
                             'mirrors stands for how many DMD pixels')
        
        s = mirrors * DMD.pixel_size / qCMOS.pixel_size
        _sigma = Expt.sigma / qCMOS.pixel_size
        
        return np.ceil([DI.center - s - 3*_sigma, DI.center + 3*_sigma]).astype(int)


    def _cropping(self, data, mirrors=None):
        temp = data[:, :-4, :]
        noise = (temp[:, :5,   :5].mean((1, 2)) + temp[:, -5:,   :5].mean((1, 2))  + 
                 temp[:, :5, -5: ].mean((1, 2)) + temp[:, -5:, -5: ].mean((1, 2))) / 4
        
        if self.measure == 'spade':
            cropped = np.array(self._spade_reader(temp)).T
        if self.measure == 'di':
            cropped = temp[:, self._di_roi(mirrors)[0]:self._di_roi(mirrors)[1], DI.ax]
        
        return cropped, noise


    def _reshaping(self, data, noise, length, mirrors=None):
        detectors = 2 if self.measure == 'spade' else - np.subtract(*self._di_roi(mirrors))
        reshaped_data = data[:length, :].reshape(Expt.repeat_times, -1, detectors)[:, :Expt.samples_length, :]
        reshaped_noise = noise[:length].reshape(Expt.repeat_times, -1)[:, :Expt.samples_length]
        return reshaped_data, reshaped_noise


    def _processing(self, avi, length, mirrors=None):
        avi = aviread(avi)
        cropped, noise = self._cropping(avi, mirrors)
        return self._reshaping(cropped, noise, length, mirrors)
    

    def load(self, files, lengths, mirrors=None):
        if len(files) != len(lengths):
            raise ValueError('len(files) and len(lengths) must be the same')
        else:
            l = len(files)

        processed, noise = zip(*[
            self._processing(files[i], lengths[i], mirrors) for i in range(l)])
        
        raw, noise = np.array(processed), np.array(noise)
        photons = raw.sum(-1)

        return raw, photons, noise




class TDEst(_ChooseMeasurement):
    def _normalize(self, data):
        data = np.array(data)
        temp = data - data.mean()
        scale = (temp[temp>0].mean() - temp[temp<0].mean()) / 2
        return temp / scale
    

    @cache
    def estimator(self, dataset, noise=None, di_method='mle'):
        origin_shape = dataset.shape

        detectors = origin_shape[-1]
        works = np.prod(origin_shape[:-1])

        flatten_data = dataset.reshape(-1, detectors)

        if self.measure == 'di':
            if di_method.lower() == 'simple':
                temp = flatten_data / flatten_data.sum(axis=-1).reshape(-1, 1)
                x_axis = np.arange(detectors)
                time_domain = (temp @ x_axis)

            elif di_method.lower() == 'mle':
                def _p(x, s, w):
                    _sigma = Expt.sigma / qCMOS.pixel_size
                    return (1-w)/np.sqrt(tau*_sigma**2)*np.exp(-(x-s)**2/(2*_sigma**2)) + w/detectors
                
                def _negative_ll(data, w):
                    x_axis = np.arange(detectors)
                    return lambda s: - data.T @ np.log(_p(x_axis, s, w)) / data.sum()
                
                def _run_mle(data, w):
                    # Use BFGS algorithm to minimize negative log-likelihood function.
                    result = minimize(_negative_ll(data, w), x0=detectors/2, method='BFGS')
                    
                    if result.success:
                        return result.x[0]
                    else:
                        raise RuntimeError('not converged')
                
                if noise is None:
                    time_domain = [_run_mle(flatten_data[i], 0) for i in range(works)]
                else:
                    w_set = np.ravel(noise) / flatten_data.mean(-1)
                    time_domain = [_run_mle(flatten_data[i], w_set[i]) for i in range(works)]  

        if self.measure == 'spade':
            # SPADE's time domain estmator is extremely simple.
            time_domain = flatten_data[:, 0] - flatten_data[:, 1]

        return self._normalize(time_domain).reshape(*origin_shape[:-1])




class FreqEst:
    def __init__(self, est_phi=False, phi=None):
        if type(est_phi) == bool:
            self.est_phi = est_phi

        if not self.est_phi:
            if type(phi) in (int, float):
                self.phi = phi
            else:
                raise ValueError('phi must be given')


    def _normalize_phase(self, phase):
        # Normalize phase to be within [0, 2*pi)
        phase = phase % tau
        # Adjust phase to be within [-pi, pi)
        if phase > pi:
            phase -= tau
        return phase
    

    def waveform(self, n, f, phi, a):
        return a*np.cos(tau * f * n + phi)


    # Use FFT as pre-estimator.
    def fft_estimator(self, signal):
        l = len(signal)

        fft = np.fft.fft(signal)[:l // 2]
        freq = np.fft.fftfreq(l, Expt.expo_time)[:l // 2]
        
        amplitude = np.abs(fft)
        phase = np.angle(fft)

        peak_index = np.argmax(amplitude[1:]) + 1
        freq_est = freq[peak_index]
        phase_est = phase[peak_index]
        
        return freq_est / Expt.sampling_rate, self._normalize_phase(phase_est)
    

    # Use LSE as frequency estimator. 
    # Note that LSE and MLE are the same in WGN.
    def lse_estimator(self, dataset):
        N = dataset.shape[-1]
        n = np.arange(N, dtype=np.float64)
        
        if self.est_phi:
            s = lambda n, f, phi: self.waveform(n, f, phi, 1)
        else:
            s = lambda n, f: self.waveform(n, f, self.phi, 1)

        freq_est, phi_est = [], []
        for group in dataset:
            for data in group:
                f_init, phi_init = self.fft_estimator(data)
                init = [f_init, phi_init] if self.est_phi else [f_init]
                popt, pcov = curve_fit(s, n, data, init)

                if np.linalg.cond(pcov) <= 5e5:
                    if self.est_phi:
                        freq_est.append(popt[0])
                        phi_est.append(popt[1])
                    else:
                        freq_est.append(popt.item())
                        phi_est.append(phi_init)
                else:
                    raise RuntimeError('np.linalg.cond(pcov) > 50000, the algorithm may not converge')

        return np.array(freq_est).reshape(*dataset.shape[:-1]), np.array(phi_est).reshape(*dataset.shape[:-1])


    def fim(self, f, phi, a):
        n = np.arange(Expt.samples_length)
        fi11 = (n**2 * np.sin(tau * f * n + phi)**2).sum()
        fi12 = (n**1 * np.sin(tau * f * n + phi)**2).sum()
        fi22 = (n**0 * np.sin(tau * f * n + phi)**2).sum()

        if self.est_phi:
            matrix = np.array([[fi11, fi12],
                               [fi12, fi22]])
        else:
            matrix = fi11

        return tau**2 * a**2 * matrix


    def fim_approxed(self, a):
        n = np.arange(Expt.samples_length)
        fi11 = (n**2).sum()
        fi12 = (n**1).sum()
        fi22 = (n**0).sum()

        if self.est_phi:
            matrix = np.array([[fi11, fi12],
                               [fi12, fi22]])
        else:
            matrix = fi11

        return tau**2 * a**2 * matrix / 2
